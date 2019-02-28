/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
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

#include <vector>

#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>

#include "include/DarknetInterface.h"



namespace DefaultTracker {
    // Assumes detections is sorted by DarknetResult::frame_number
    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks(int num_classes_per_region, double min_overlap,
                                                         std::vector<DarknetResult> &&detections);

    MPF::COMPONENT::MPFImageLocation CreateImageLocation(int num_classes_per_region, DarknetResult &detection);
}


namespace PreprocessorTracker  {
    // Assumes detections is sorted by DarknetResult::frame_number
    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks(std::vector<DarknetResult> &&detections);

    void CombineImageLocation(const cv::Rect &rect, float prob,
                              MPF::COMPONENT::MPFImageLocation &image_location);
}

#endif //OPENMPF_COMPONENTS_TRACKERS_H
