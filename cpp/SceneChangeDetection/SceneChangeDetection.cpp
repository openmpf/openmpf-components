/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

#include "SceneChangeDetection.h"

#include <algorithm>
#include <map>
#include <utility>

#include <opencv2/imgproc.hpp>

#include <detectionComponentUtils.h>
#include <MPFVideoCapture.h>
#include <Utils.h>


using namespace MPF::COMPONENT;
using namespace cv;


std::string SceneChangeDetection::GetDetectionType() {
    return "SCENE";
}

bool SceneChangeDetection::Init() {

    // Determine where the executable is running.
    std::string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    std::string plugin_path = run_dir + "/SceneChangeDetection";
    std::string config_path = plugin_path + "/config";

    logger_ = log4cxx::Logger::getLogger("SceneChangeDetection");

    LOG4CXX_DEBUG(logger_, "Plugin path: " << plugin_path);
    LOG4CXX_INFO(logger_, "Initializing SceneChangeDetection");

    // Initialize dilateKernel.
    dilateKernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(5, 5));

    LOG4CXX_INFO(logger_, "INITIALIZED COMPONENT" );
    return true;
}



bool SceneChangeDetection::Close() {
    return true;
}

/*
 * Calculates the difference in edge pixels between the last two frames.
 * Returns true when the difference exceeds edge_thresh.
 */
bool SceneChangeDetection::DetectChangeEdges(const cv::Mat &frameGray,
                                             cv::Mat &lastFrameEdgeFinal) const {
    cv::Mat frameEdges, frameEdgeFinal, edgeDst;
    blur(frameGray, frameEdges, Size(3, 3));
    Canny(frameEdges, frameEdges, 90, 270, 3);
    frameGray.copyTo(frameEdgeFinal, frameEdges);
    dilate(frameEdgeFinal, frameEdgeFinal, dilateKernel);
    absdiff(frameEdgeFinal, lastFrameEdgeFinal, edgeDst);
    double sumEdges = sum(edgeDst).val[0];
    int frame_pixels = edgeDst.size().width * edgeDst.size().height;
    double deltaEdges = sumEdges / frame_pixels;

    frameEdges.copyTo(lastFrameEdgeFinal);
    return deltaEdges > edge_thresh;
}

/*
 * Performs histogram comparison between the last two frames.
 * Returns true when correlation falls below hist_thresh.
 */
bool SceneChangeDetection::DetectChangeHistogram(const cv::Mat &frame, cv::Mat &lastHist) const
{
    MatND hist;
    const float* ranges[] = { hranges, sranges };


    calcHist( &frame, 1, (const int*) &channels[0], Mat(), // Do not use mask.
             hist, 2, histSize, ranges,
             true, // The histogram is uniform.
             false );
    double val = compareHist(hist, lastHist, cv::HISTCMP_CORREL);
    hist.copyTo(lastHist);
    return val < hist_thresh;
}

/*
 * Calculates average difference in HSV values between the last two frames.
 * Returns true when total average difference exceeds cont_thresh.
 */
bool SceneChangeDetection::DetectChangeContent(const cv::Mat &frame, cv::Mat &lastFrameHSV) const
{
    Mat frameHSV, dst;
    cvtColor(frame, frameHSV, COLOR_BGR2HSV);
    absdiff(frameHSV, lastFrameHSV, dst);
    auto sum_ = sum(dst);
    int frame_pixels = dst.size().width * dst.size().height;
    double deltaH = sum_[0] / frame_pixels;
    double deltaS = sum_[1] / frame_pixels;
    double deltaV = sum_[2] / frame_pixels;
    double deltaHSVAvg = (deltaH + deltaS + deltaV) / (3.0);

    frameHSV.copyTo(lastFrameHSV);
    return deltaHSVAvg > cont_thresh;
}

/*
 * Performs threshold detection for scene fade outs.
 * Note: Once threshold is met, fadeOut is set to true
 * and all subsequent frames in the scene will be marked as fade outs.
 */
bool SceneChangeDetection::DetectChangeThreshold(const cv::Mat &frame, cv::Mat &lastFrame)
{
    bool FUT = frameUnderThreshold(frame, thrs_thresh, numPixels * 3);
    if (!fadeOut && FUT)
    {
        fadeOut = true;
    }
    else if (fadeOut && FUT)
    {
        return true;
    }
    return false;
}

/*
 * Counts number of pixels that fall under threshold value.
 * If total number of dark pixels exceeds minThreshold, return true.
 */
bool SceneChangeDetection::frameUnderThreshold(
        const cv::Mat &image, double threshold, double numPixels) const
{
    int minThreshold = (int)(numPixels * (1.0 - minPercent));
    int frameAmount = 0;
    std::vector<Mat> channels;
    split(image, channels);
    for (int y = 0; y < image.rows; y++)
    {
        cv::Mat dst;
        compare(channels[0].row(y), Scalar(threshold), dst, CMP_GT);
        frameAmount += countNonZero(dst);
        if (frameAmount > minThreshold)
        {
            return false;
        }
    }
    return true;
}

/*
 * Performs up to four different scene change detection protocols.
 */
std::vector<MPFVideoTrack> SceneChangeDetection::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_DEBUG(logger_, "Job feed_forward_is: " << job.has_feed_forward_track);
        if (job.has_feed_forward_track)
        {
            LOG4CXX_DEBUG(logger_, "STARTING FEEDFORWARD GETDETECTIONS" );
        }
        else
        {
            LOG4CXX_DEBUG(logger_, "STARTING NO FF GETDETECTIONS" );
        }

        LOG4CXX_DEBUG(logger_, "Data URI = " << job.data_uri);

        MPFVideoCapture cap(job);

        int frame_count = cap.GetFrameCount();
        LOG4CXX_DEBUG(logger_, "frame count = " << frame_count);
        LOG4CXX_DEBUG(logger_, "begin frame = " << job.start_frame);
        LOG4CXX_DEBUG(logger_, "end frame = " << job.stop_frame);

        Mat lastFrameEdgeFinal, frameGray, lastFrameHSV, lastFrame;
        MatND lastHist;
        int frame_index;
        const std::vector<cv::Mat> &init_frames = cap.GetInitializationFramesIfAvailable(1);
        // Attempt to use the frame before the start of the segment to initialize the foreground.
        // If one is not available, use frame 0 and start processing at frame 1.
        if (init_frames.empty()) {
            frame_index = 1;
            bool success = cap.Read(lastFrame);
            if (!success){
                return { };
            }
        }
        else {
            frame_index = 0;
            lastFrame = init_frames.at(0);
        }


        int lastFrameNum = 0;
        int rows, cols;
        const float* ranges[] = { hranges, sranges };

        // Initialize the first frame.
        edge_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties, "EDGE_THRESHOLD", edge_thresh);
        hist_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties, "HIST_THRESHOLD", hist_thresh);
        cont_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties, "CONT_THRESHOLD", cont_thresh);
        thrs_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties, "THRS_THRESHOLD", thrs_thresh);

        minPercent = DetectionComponentUtils::GetProperty<double>(job.job_properties, "MIN_PERCENT", minPercent);
        minScene = DetectionComponentUtils::GetProperty<int>(job.job_properties, "MIN_SCENECHANGE_LENGTH", minScene);

        do_hist = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "DO_HIST", do_hist);
        do_cont = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "DO_CONT", do_cont);
        do_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "DO_THRS", do_thrs);
        do_edge = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "DO_EDGE", do_edge);

        use_middle_frame = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "USE_MIDDLE_FRAME", use_middle_frame);

        rows = lastFrame.rows;
        cols = lastFrame.cols;
        numPixels = rows * cols;

        cvtColor(lastFrame, frameGray, COLOR_BGR2GRAY);
        cv::Mat frameEdges, frameEdgeFinal;
        blur(frameGray, frameEdges, Size(3, 3));
        Canny(frameEdges, frameEdges, 90, 270, 3);
        frameGray.copyTo(lastFrameEdgeFinal, frameEdges);
        dilate(lastFrameEdgeFinal, lastFrameEdgeFinal, dilateKernel);
        cvtColor(lastFrame, lastFrameHSV, COLOR_BGR2HSV);
        calcHist( &lastFrame, 1, (const int*) &channels[0], Mat(),
             lastHist, 2, histSize, ranges,
             true,
             false );

        cv::Mat frame;
        std::map<int, int> keyframes;
        while (cap.Read(frame)) {

            cvtColor(frame,frameGray,COLOR_BGR2GRAY);
            bool edge_result = do_edge && DetectChangeEdges(frameGray, lastFrameEdgeFinal);
            bool hist_result = do_hist && DetectChangeHistogram(frame, lastHist);
            bool cont_result = do_cont && DetectChangeContent(frame, lastFrameHSV);
            bool thrs_result = do_thrs && DetectChangeThreshold(frame, lastFrame);

            if (edge_result || hist_result || cont_result || thrs_result)
            {
                if (frame_index - lastFrameNum >= minScene)
                {
                    keyframes[frame_index] = lastFrameNum;
                    lastFrameNum = frame_index;
                }
            }

            frame_index++;
        }

        keyframes[frame_index] = lastFrameNum;
        std::vector<MPFVideoTrack> tracks;
        for(auto const& kv : keyframes)
        {
            auto start_frame = kv.second;
            auto end_frame = kv.first;
            MPFVideoTrack track(start_frame, end_frame - 1);
            if (use_middle_frame) {
                track.frame_locations.insert(
                        std::pair<int,MPFImageLocation>(start_frame + (int)((end_frame - start_frame) / 2),
                            MPFImageLocation(0, 0, cols, rows)
                            )
                        );
            } else {
                for(int i = start_frame; i < end_frame; i++)
                {
                    track.frame_locations.insert(
                        std::pair<int,MPFImageLocation>(i,
                            MPFImageLocation(0, 0, cols, rows)
                            )
                        );
                }
            }

            cap.ReverseTransform(track);
            tracks.push_back(track);
        }

        LOG4CXX_INFO(logger_, "Processing complete. Found " + std::to_string(tracks.size()) + " tracks.");
        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }
}

MPF_COMPONENT_CREATOR(SceneChangeDetection);
MPF_COMPONENT_DELETER();
