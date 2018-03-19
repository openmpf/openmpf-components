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

#include <unordered_map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>

struct DarknetResult {
    cv::Rect detection_rect;
    std::vector<std::pair<float, std::string>> object_type_probs;
};


class SingleDetectionPerTrackTracker {
public:
    explicit SingleDetectionPerTrackTracker(int num_classes_per_region);

    void ProcessFrameDetections(std::vector<DarknetResult> &&new_detections, int frame_number);

    // Returns tracks and resets the tracker to its initial state.
    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks();

    static MPF::COMPONENT::MPFImageLocation CreateImageLocation(int num_classes_per_region, DarknetResult &detection);

private:
    const int num_classes_per_region_;
    std::vector<MPF::COMPONENT::MPFVideoTrack> tracks_;
};


class PreprocessorTracker {
public:

    void ProcessFrameDetections(std::vector<DarknetResult> &&new_detections, int frame_number);

    // Returns tracks and resets the tracker to its initial state.
    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks();

    static void CombineImageLocation(const cv::Rect &rect, float prob,
                                     MPF::COMPONENT::MPFImageLocation &image_location);

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
