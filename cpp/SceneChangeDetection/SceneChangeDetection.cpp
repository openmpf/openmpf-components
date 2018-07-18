/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

#include <unistd.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <QDebug>
#include <QtCore>
#include <boost/tokenizer.hpp>



#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>
#include <boost/algorithm/string/predicate.hpp>


#include <Utils.h>
#include <detectionComponentUtils.h>

#include <algorithm>

#include "MPFSimpleConfigLoader.h"

using namespace MPF::COMPONENT;
using namespace cv;

SceneChangeDetection::SceneChangeDetection(){

}

SceneChangeDetection::~SceneChangeDetection() {

}

std::string SceneChangeDetection::GetDetectionType() {
    return "CLASS";
}

bool SceneChangeDetection::Init() {

    // Determine where the executable is running
    std::string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    std::string plugin_path = run_dir + "/SceneChangeDetection";
    std::string config_path = plugin_path + "/config";

    // Configure logger
    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    logger_ = log4cxx::Logger::getLogger("SceneChangeDetection");

    LOG4CXX_DEBUG(logger_, "Plugin path: " << plugin_path);

    // Initialize dilateKernel
    dilateKernel = getStructuringElement(MORPH_RECT,Size(11,11),Point(5,5));


    SetDefaultParameters();
    //Once this is done - parameters will be set and SetReadConfigParameters() can be called again to revert back
    //to the params read at initialization
    std::string config_params_path = config_path + "/mpfSceneChangeDetection.ini";
    int rc = LoadConfig(config_params_path, parameters);
    if (rc) {
        LOG4CXX_ERROR(logger_, "Could not parse config file: " << config_params_path);
        return (false);
    }

    SetReadConfigParameters();


    LOG4CXX_INFO(logger_, "INITIALIZED COMPONENT" );
    return true;
}

/*
 * Called during Init.
 * Initializes default parameter values.
 */
void SceneChangeDetection::SetDefaultParameters() {
    //Threshold for edge detection.
    //Higher values result in less detections (lower sensitivity)
    //Range 0-1
    edge_thresh = 0.75;

    //Threshold for histogram detection.
    //Higher values result in more detections (higher sensitivity)
    //Range 0-1
    hist_thresh = 0.25;

    //Threshold for content detection.
    //Higher values result in less detections (lower sensitivity)
    //Range 0-1
    cont_thresh = 0.30;

    //Threshold for thrs detection.
    //Higher values result in more detections (higher sensitivity)
    //Range 0-1? (most likely 0-255).
    thrs_thresh = 12.0;

    //Second threshold for thrs detection (combines with thrs_thres)
    //Higher values decrease sensitivity.
    //Range 0-1
    minPercent = 0.95;

    //Expected min number of frames between scene changes.
    minScene = 15; 

    //Toggles each type of detection (true = perform detection)
    do_hist = true;
    do_cont = true;
    do_thrs = true;
    do_edge = true;
    
}

/*
 * Called during Init.
 * Sets parameters from .ini file.
 */
void SceneChangeDetection::SetReadConfigParameters() {


    if(parameters.contains("DO_HIST")) {
        do_hist = (parameters["DO_HIST"].toInt() > 0);
    }
    if(parameters.contains("DO_CONT")) {
        do_cont = (parameters["DO_CONT"].toInt() > 0);
    }
    if(parameters.contains("DO_THRS")) {
        do_thrs = (parameters["DO_THRS"].toInt() > 0);
    }
    if(parameters.contains("DO_EDGE")) {
        do_edge = (parameters["DO_EDGE"].toInt() > 0);
    }

    if(parameters.contains("HIST_THRESHOLD")) {
        hist_thresh = parameters["HIST_THRESHOLD"].toDouble();
    }
    if(parameters.contains("CONT_THRESHOLD")) {
        cont_thresh = parameters["CONT_THRESHOLD"].toDouble();
    }
    if(parameters.contains("THRS_THRESHOLD")) {
        thrs_thresh = parameters["THRS_THRESHOLD"].toDouble();
    }
    if(parameters.contains("EDGE_THRESHOLD")) {
        edge_thresh = parameters["EDGE_THRESHOLD"].toDouble();
    }
    if(parameters.contains("MIN_PERCENT")) {
        minPercent = parameters["MIN_PERCENT"].toDouble();
    }

    if(parameters.contains("MIN_SCENE")) {
        minScene = parameters["MIN_SCENE"].toInt();
    }
}


bool SceneChangeDetection::Close() {
    return true;
}

/*
 * Calculates the difference in edge pixels between the last two frames.
 * Returns true when the difference exceeds edge_thresh.
 */
bool SceneChangeDetection::EdgeDetector(cv::Mat frameGray, cv::Mat &lastFrameEdgeFinal)
{
    cv::Mat frameEdges,frameEdgeFinal,edgeDst;
    blur(frameGray,frameEdges,Size(3,3));
    Canny(frameEdges,frameEdges,90,270,3);
    frameGray.copyTo(frameEdgeFinal,frameEdges);
    dilate(frameEdgeFinal,frameEdgeFinal,dilateKernel);
    absdiff(frameEdgeFinal,lastFrameEdgeFinal,edgeDst);
    double sumEdges = sum(edgeDst).val[0];
    int frame_pixels = edgeDst.size().width*edgeDst.size().height;
    double deltaEdges = sumEdges / frame_pixels;//numPixels;

    frameEdges.copyTo(lastFrameEdgeFinal);
    if(deltaEdges > edge_thresh)
    {
        return true;
    }
    else
    {
        return false;
    }
}

/*
 * Performs histogram comparison between the last two frames.
 * Returns true when correlation falls below hist_thresh. 
 */
bool SceneChangeDetection::HistogramDetector(cv::Mat frame, cv::Mat &lastHist)
{
    MatND hist;
    const float* ranges[] = { hranges, sranges };


    calcHist( &frame, 1, (const int*) &channels[0], Mat(), // do not use mask
             hist, 2, histSize, ranges,
             true, // the histogram is uniform
             false );
    double val = compareHist(hist,lastHist,CV_COMP_CORREL);
    hist.copyTo(lastHist);
    if(val <hist_thresh)
    {
        return true;
    }
    else return false;
}

/*
 * Calculates average difference in HSV values between the last two frames.
 * Returns true when total average difference exceeds cont_thresh.
 */
bool SceneChangeDetection::ContentDetector(cv::Mat frame, cv::Mat &lastFrameHSV)
{
    Mat frameHSV, dst;
    cvtColor(frame,frameHSV,COLOR_BGR2HSV);
    absdiff(frameHSV,lastFrameHSV,dst);
    auto sum_ = sum(dst).val;
    int frame_pixels = dst.size().width*dst.size().height;
    double deltaH = sum_[0]/frame_pixels;///numPixels;
    double deltaS = sum_[1]/frame_pixels;
    double deltaV = sum_[2]/frame_pixels;
    double deltaHSVAvg = (deltaH + deltaS + deltaV) / (3.0);

    frameHSV.copyTo(lastFrameHSV);
    if(deltaHSVAvg > cont_thresh)
    {
        return true;
    }
    else return false;

}

/*
 * Performs threshold detection for scene fade outs.
 * Note: Once threshold is met, fadeOut is set to true
 * and all subsequent frames in the scene will be marked as fade outs.
 */
bool SceneChangeDetection::ThresholdDetector(cv::Mat frame, cv::Mat &lastFrame)
{
    bool FUT = frameUnderThreshold(frame,thrs_thresh,numPixels*3);
    if(!fadeOut && FUT)
    {
        fadeOut = true;
    }
    else if(fadeOut && FUT)
    {
        return true;
    }
    else return false;
}

/*
 * Counts number of pixels that fall under threshold value.
 * If total number of dark pixels exceeds minThreshold, return true.
 */
bool SceneChangeDetection::frameUnderThreshold(cv::Mat image, double threshold, double numPixels)
{
    int minThreshold = (int)(numPixels*(1.0-minPercent));
    int frameAmount = 0;
    std::vector<Mat> channels;
    split(image,channels);
    for(int y=0;y<image.rows;y++)
    {
        cv::Mat dst;
        compare(channels[0].row(y),Scalar(threshold),dst,CMP_GT);
        frameAmount += countNonZero(dst);
        if (frameAmount> minThreshold)
        {
            return false;
        }
    }
    return true;
}

/*
 * Performs up to four different scene change detection protocols.
 */
MPFDetectionError SceneChangeDetection::GetDetections(const MPFVideoJob &job, std::vector<MPFVideoTrack> &locations) {
    try {
        LOG4CXX_DEBUG(logger_, "Job feed_forward_is: " << job.has_feed_forward_track);
        if(job.has_feed_forward_track)
        {

            LOG4CXX_INFO(logger_, "STARTING FEEDFORWARD GETDETECTIONS" );
        }
        else
        {
            LOG4CXX_INFO(logger_, "STARTING NO FF GETDETECTIONS" );
        }

       LOG4CXX_DEBUG(logger_, "Data URI = " << job.data_uri);

        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(logger_, "Invalid image file");
            return MPF_INVALID_DATAFILE_URI;
        }


       
        MPFVideoCapture cap(job);
        bool success = false;
        if (!cap.IsOpened()) {
            return MPF_COULD_NOT_OPEN_DATAFILE;
        }

        int frame_count = cap.GetFrameCount();
        LOG4CXX_DEBUG(logger_, "frame count = " << frame_count);
        LOG4CXX_DEBUG(logger_, "begin frame = " << job.start_frame);
        LOG4CXX_DEBUG(logger_, "end frame = " << job.stop_frame);

        std::vector<MPFVideoTrack> tracks;
        int init_frame_index = 0;
        if (job.start_frame > 0) {
            init_frame_index = job.start_frame - 1;
        }
        
        int frame_index = init_frame_index+1;
        
        //Uncomment to enable scene change detection at the first frame.
        //Only applies for non-zero initial frames.
        //Ex. run getDetections on frames 50-100, uncomment allows frame 49 to be extracted and compared against frame 50
        //otherwise, starting comparison will be frame 50 and 51. 
        //if(init_frame_index > 1){
        //    frame_index = frame_index-1;
        //}
        cap.SetFramePosition(frame_index-1); // start from one frame before to get "lasts"
        double maxProgress = 80;
        double progressStep = maxProgress/cap.GetFrameCount();
        int lastFrameNum = frame_index;
        int count = 0;

        Mat lastFrameEdgeFinal, frameGray, lastFrameHSV, lastFrame;
        MatND lastHist;

        int rows, cols;
        const float* ranges[] = { hranges, sranges };
        bool first_it = true;

        while (frame_index < qMin(frame_count, job.stop_frame + 1)) {
            cv::Mat frame;
            bool success = cap.Read(frame);
            rows = frame.rows;
            cols = frame.cols;
            numPixels = rows * cols;
            int currFrameNum = cap.GetFrameCount();
            double msec = cap.GetProperty(CAP_PROP_POS_MSEC);
            cvtColor(frame,frameGray,COLOR_BGR2GRAY);

            if(first_it)
            {
                first_it = false;
                lastFrame = frame;
                cv::Mat frameEdges,frameEdgeFinal;
                blur(frameGray,frameEdges,Size(3,3));
                Canny(frameEdges,frameEdges,90,270,3);
                frameGray.copyTo(lastFrameEdgeFinal,frameEdges);
                dilate(lastFrameEdgeFinal,lastFrameEdgeFinal,dilateKernel);
                cvtColor(frame, lastFrameHSV, COLOR_BGR2HSV);

                calcHist( &frame, 1, (const int*) &channels[0], Mat(), // do not use mask
                     lastHist, 2, histSize, ranges,
                     true, // the histogram is uniform
                     false );
                
                frame_index++;
                continue;


            }
            //Uncomment to enable adjusting property values in MPF Workflow.
            //While disabled, property values are initialized from .ini file instead.
            /*
            edge_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties,"EDGE_THRESHOLD",edge_thresh);
            hist_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties,"HIST_THRESHOLD",hist_thresh);
            cont_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties,"CONT_THRESHOLD",cont_thresh);
            thrs_thresh = DetectionComponentUtils::GetProperty<double>(job.job_properties,"THRS_THRESHOLD",thrs_thresh);

            minPercent = DetectionComponentUtils::GetProperty<double>(job.job_properties,"MIN_PERCENT",minPercent);
            minScene = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MIN_SCENE",minScene);

            do_hist = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"DO_HIST",do_hist);
            do_cont = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"DO_CONT",do_cont);
            do_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"DO_THRS",do_thrs);
            do_edge = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"DO_EDGE",do_edge);*/
            
            bool edge_result = do_edge && EdgeDetector(frameGray, lastFrameEdgeFinal);
            
            bool hist_result = do_hist && HistogramDetector(frame, lastHist);
            
            bool cont_result = do_cont && ContentDetector(frame, lastFrameHSV);
            
            bool thrs_result = do_thrs && ThresholdDetector(frame, lastFrame);

            if(edge_result || hist_result || cont_result || thrs_result)
            {
                if(frame_index-lastFrameNum >= minScene)    
                {
                    keyframes[frame_index]=lastFrameNum;
                    lastFrameNum = frame_index;
                }
            }


            frame_index++;

        }
        keyframes[frame_index]=lastFrameNum;
        bool use_exemplar = true;
        for(auto const& kv : keyframes)
        {
            auto start_frame = kv.second;
            auto end_frame = kv.first;
            MPFVideoTrack track(start_frame,end_frame);
            if(!use_exemplar)
            {
                for(int i = start_frame;i<end_frame;i++)
                {
                    track.frame_locations.insert(
                        std::pair<int,MPFImageLocation>(i-start_frame,
                            MPFImageLocation(0,0,cols, rows)
                            )
                        );
                }
            }
            else
            {
                track.frame_locations.insert(
                        std::pair<int,MPFImageLocation>(start_frame + (int)((end_frame-start_frame)/2),
                            MPFImageLocation(0,0,cols, rows)
                            )
                        );
            }
            cap.ReverseTransform(track);
            locations.push_back(track);
        }

        cap.Release();

        
        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }
}

MPF_COMPONENT_CREATOR(SceneChangeDetection);
MPF_COMPONENT_DELETER();

