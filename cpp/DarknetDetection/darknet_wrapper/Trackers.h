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


#ifndef OPENMPF_COMPONENTS_TRACKERS_H
#define OPENMPF_COMPONENTS_TRACKERS_H

#include <map>
#include <unordered_map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>

#include "../include/DarknetInterface.h"

namespace TrackingHelpers {
    MPF::COMPONENT::MPFImageLocation CreateImageLocation(int num_classes_per_region, DarknetResult &detection);

    void CombineImageLocation(const cv::Rect &rect, float prob,
                              MPF::COMPONENT::MPFImageLocation &image_location);
}


class DefaultTracker {
public:
    DefaultTracker(int num_classes_per_region, double min_overlap);

    void ProcessFrameDetections(std::vector<DarknetResult> &&new_detections, int frame_number);

    // Returns tracks and resets the tracker to its initial state.
    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks();

private:
    const int num_classes_per_region_;

    const double min_overlap_;

    std::multimap<std::pair<int, std::string>, MPF::COMPONENT::MPFVideoTrack> tracks_;

    static double GetOverlap(const cv::Rect &detection_rect, const MPF::COMPONENT::MPFVideoTrack &track);
};




class PreprocessorTracker {
public:

    void ProcessFrameDetections(const std::vector<DarknetResult> &new_detections, int frame_number);

    // Returns tracks and resets the tracker to its initial state.
    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks();

private:
    // standard library does not define a hash function for pairs
    class PairHasher {
    public:
        size_t operator()(const std::pair<int, std::string> &pair) const;

    private:
        std::hash<int> int_hasher_;
        std::hash<std::string> string_hasher_;
    };

    std::unordered_map<std::pair<int, std::string>, // Key is { track.stop_frame, object_type }
                       MPF::COMPONENT::MPFVideoTrack, PairHasher> tracks_;

    void AddNewTrack(const cv::Rect &detection_rect, float prob, const std::string &type, int frame_number);

    void AddNewImageLocationToTrack(const cv::Rect &rect, float prob, const std::string &type,
                                    int frame_number, MPF::COMPONENT::MPFVideoTrack &track);

    static void CombineImageLocation(const cv::Rect &rect, float prob,
                                     int frame_number, MPF::COMPONENT::MPFVideoTrack &track);
};

#endif //OPENMPF_COMPONENTS_TRACKERS_H
