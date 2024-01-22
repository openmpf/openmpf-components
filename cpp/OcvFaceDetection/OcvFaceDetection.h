/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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


#ifndef OPENMPF_COMPONENTS_OCVFACEDETECTION_H
#define OPENMPF_COMPONENTS_OCVFACEDETECTION_H

#include <map>
#include <string>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/highgui.hpp>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <MPFDetectionComponent.h>
#include <MPFVideoCapture.h>


#include <log4cxx/logger.h>

#include "OcvDetection.h"



struct Track {
    MPF::COMPONENT::MPFVideoTrack face_track;
    int init_point_count;
    int current_point_count;
    float current_point_percent;
    int last_face_detected_index;
    bool track_lost;
    std::vector <cv::Point2f> previous_points;
    std::vector <cv::Point2f> current_points;

    //TODO: see if these are useful
    cv::Mat first_gray_frame;
    std::vector <cv::KeyPoint> previous_keypoints;
    std::vector <cv::KeyPoint> current_keypoints;
    std::vector <cv::KeyPoint> first_detected_keypoints;

    Track() : init_point_count(0), current_point_count(0), current_point_percent(0.0), last_face_detected_index(-1),
            track_lost(false) { }
};

class OcvFaceDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

private:
    OcvDetection ocv_detection;

    int max_features;
    cv::Ptr <cv::FeatureDetector> feature_detector;

    bool imshow_on;
    int verbosity;

    int min_face_size; //the width and height of min size for detection
    unsigned int min_init_point_count;
    float min_redetect_point_perecent;
    float min_point_percent;
    float max_optical_flow_error;
    float min_initial_confidence;

    //TODO: add to config file
    //when a face is not detected - this is the minimum percentage of features detected compared to initial point count
    //if below this the track should not continue
    float min_good_match_percent;

    std::vector <Track> current_tracks;
    std::vector <Track> saved_tracks;

    log4cxx::LoggerPtr OpenFaceDetectionLogger;

    void SetDefaultParameters();
    void GetPropertySettings(const std::map <std::string, std::string> &algorithm_properties);

    void Display(const std::string title, const cv::Mat &img);

    cv::Rect GetMatch(const cv::Mat &frame_rgb_display, const cv::Mat &frame_gray, const cv::Mat &templ);

    bool IsExistingTrackIntersection(const cv::Rect new_rect, int &intersection_index);

    cv::Rect GetUpscaledFaceRect(const cv::Rect &face_rect);
    cv::Mat GetMask(const cv::Mat &frame, const cv::Rect &face, bool copy_face_rect = false);
    bool IsBadFaceRatio(const cv::Rect &face);

    void CloseAnyOpenTracks(int frame_index);

    void AdjustRectToEdges(cv::Rect &rect, const cv::Mat &src);

    void LogDetection(const MPF::COMPONENT::MPFImageLocation& face);

    void CloseWindows();

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job,
            MPF::COMPONENT::MPFVideoCapture &video_capture);

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetectionsFromImageData(
            const MPF::COMPONENT::MPFImageJob &job, cv::Mat &image_data);

public :

    //removed these two from Init to fit the virtual function
    void SetModes(bool display_window, bool print_debug_info);

    //should continue to define virtual to help identify virtual functions
    bool Init() override;
    bool Close() override;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(const MPF::COMPONENT::MPFImageJob &job) override;

};


#endif //OPENMPF_COMPONENTS_OCVFACEDETECTION_H
