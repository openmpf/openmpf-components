/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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


#ifndef OPENMPF_COMPONENTS_DLIBFACEDETECTION_H
#define OPENMPF_COMPONENTS_DLIBFACEDETECTION_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Qt headers
#include "QHash"

//log4cxx headers
#include <log4cxx/logger.h>

//dlib
// NOTE: Due to a name conflict between a function in the dlib library
// and a macro in the opencv 3.1 library, the dlib headers must be
// included before opencv.
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>
#include <dlib/geometry.h>
#include <dlib/opencv.h>

//opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "Utils.h"
#include "MPFDetectionComponent.h"
#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"
#include "MPFVideoCapture.h"

struct DlibTrack {
    MPF::COMPONENT::MPFVideoTrack mpf_video_track;
    dlib::correlation_tracker correlation_tracker;
    //this keeps track of how many frames (not accounting for frame interval)
    // that have been added without a new detection
    //TODO: currently being logged, but not used
    // to stop tracks
    int frames_since_last_detection;
    //updated represents a new detection matched with the existing track
    bool updated;
    DlibTrack() : frames_since_last_detection(0), updated(false) { }
};

class DlibFaceDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

private:
    log4cxx::LoggerPtr logger_;

    dlib::frontal_face_detector dlib_face_detector;

    int verbosity;
    //part of the config but not the descriptor
    bool imshow_on;

    double min_detection_confidence;
    float max_intersection_overlap_pct;
    int min_track_length;
    float min_track_object_similarity_value;
    double min_update_correlation;

    //not added to the config
    float bb_grow_rate;

    std::vector<DlibTrack> current_tracks;
    std::vector<DlibTrack> saved_tracks;

    QHash<QString, QString> parameters;

    void SetDefaultParameters();
    void SetReadConfigParameters();
    void GetPropertySettings(const std::map <std::string, std::string> &algorithm_properties);

    float GetTrackObjectSimilarity(const DlibTrack &current_track, const dlib::rectangle &new_rect);
    bool IsObjectSimilar(const DlibTrack &current_track, const dlib::rectangle &new_rect);
    bool IsValidNewObject(const DlibTrack &current_track, const dlib::rectangle &new_rect);

    int GetMostSimilarOverlappingObject(const DlibTrack &current_track,
                                        const std::vector<dlib::rect_detection> &next_detected_objects);

    void CloseAnyOpenTracks();

    void GrowRect(dlib::rectangle &rect);
    void AdjustRectToEdgesDlib(dlib::rectangle &rect, const cv::Mat &src);

    void LogDetection(const MPF::COMPONENT::MPFImageLocation& face,
                      const std::string& job_name);

    void CloseWindows();

    void DlibRectToMPFImageLocation(const dlib::rectangle &object_rect,
                                    float object_detection_confidence,
                                    MPF::COMPONENT::MPFImageLocation &mpf_object_detection);

    void UpdateTracks(const dlib::cv_image<dlib::uint8> &next_frame_gray,
                      const cv::Mat &next_frame_gray_mat,
                      std::vector<dlib::rect_detection> &next_detected_objects,
                      int frame_index);

    std::vector<dlib::rect_detection> DetectFacesDlib(const cv::Mat &frame_gray);

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job, MPF::COMPONENT::MPFVideoCapture &video_capture);


    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetectionsFromImageData(
            const MPF::COMPONENT::MPFImageJob &job, cv::Mat &image_data);

public:

    bool Init();

    //removed these two from Init to fit the virtual function
    void SetModes(bool display_window, bool print_debug_info);

    bool Close();

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(const MPF::COMPONENT::MPFImageJob &job) override;


    std::string GetDetectionType();
};




#endif //OPENMPF_COMPONENTS_DLIBFACEDETECTION_H
