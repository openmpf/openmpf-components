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

#ifndef OPENMPF_COMPONENTS_DARKNETSTREAMINGDETECTION_H
#define OPENMPF_COMPONENTS_DARKNETSTREAMINGDETECTION_H

#include <functional>
#include <string>
#include <vector>

#include <log4cxx/logger.h>

#include <MPFDetectionObjects.h>
#include <MPFStreamingDetectionComponent.h>
#include <DlClassLoader.h>

#include "include/DarknetInterface.h"


class DarknetStreamingDetection : public MPF::COMPONENT::MPFStreamingDetectionComponent {

public:
    explicit DarknetStreamingDetection(const MPF::COMPONENT::MPFStreamingVideoJob &job);

    std::string GetDetectionType() override;

    void BeginSegment(const MPF::COMPONENT::VideoSegmentInfo &segment_info) override;

    bool ProcessFrame(const cv::Mat &frame, int frame_number) override;

    std::vector<MPF::COMPONENT::MPFVideoTrack> EndSegment() override;

private:
    log4cxx::LoggerPtr logger_;

    std::string job_name_;

    std::string log_prefix_;

    MPF::COMPONENT::DlClassLoader<DarknetInterface> detector_;

    std::function<
             std::vector<MPF::COMPONENT::MPFVideoTrack> (std::vector<DarknetResult>&&)
    > tracker_;


    std::vector<DarknetResult> current_segment_detections_;

    bool found_track_in_current_segment_ = false;

    DarknetStreamingDetection(const MPF::COMPONENT::MPFStreamingVideoJob &job, log4cxx::LoggerPtr logger);

    [[noreturn]] void LogError(const std::string &message);
};


#endif //OPENMPF_COMPONENTS_DARKNETSTREAMINGDETECTION_H
