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

#include <utility>
#include <vector>

#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>
#include <BaseTracker.h>
#include <RegionOverlapTracker.h>

#include "include/DarknetInterface.h"


namespace TrackingHelpers {
    MPF::COMPONENT::MPFImageLocation CreateImageLocation(int num_classes_per_region, DarknetResult &detection);

    void CombineImageLocations(const MPF::COMPONENT::MPFImageLocation &new_img_loc,
                               MPF::COMPONENT::MPFImageLocation &existing_img_loc);
}


class PreprocessorTracker : public MPF::COMPONENT::BaseTracker {
public:
    void ProcessFrameDetections(const std::vector<DarknetResult> &new_detections, int frame_number);


protected:
    bool IsSameTrack(const MPF::COMPONENT::MPFImageLocation &new_loc,
                     int frame_number,
                     const MPF::COMPONENT::MPFVideoTrack &existing_track) override;

    MPF::COMPONENT::MPFVideoTrack CreateTrack(MPF::COMPONENT::MPFImageLocation &&img_loc,
                                              int frame_number) override;

    void AddToTrack(MPF::COMPONENT::MPFImageLocation &&new_img_loc,
                    int frame_number,
                    MPF::COMPONENT::MPFVideoTrack &existing_track) override;
};



class DefaultTracker : public MPF::COMPONENT::RegionOverlapTracker {
public:
    DefaultTracker(int num_classes_per_region, double min_overlap);

    void ProcessFrameDetections(std::vector<DarknetResult> &&new_detections, int frame_number);

protected:
    bool OverlappingDetectionsAreSameTrack(const MPF::COMPONENT::MPFImageLocation &new_loc,
                                           int frame_number,
                                           const MPF::COMPONENT::MPFVideoTrack &existing_track) override;

    MPF::COMPONENT::MPFVideoTrack CreateTrack(MPF::COMPONENT::MPFImageLocation &&img_loc,
                                              int frame_number) override;

    void AddToTrack(MPF::COMPONENT::MPFImageLocation &&new_img_loc,
                    int frame_number,
                    MPF::COMPONENT::MPFVideoTrack &existing_track) override;

private:
    const int num_classes_per_region_;
};



#endif //OPENMPF_COMPONENTS_TRACKERS_H
