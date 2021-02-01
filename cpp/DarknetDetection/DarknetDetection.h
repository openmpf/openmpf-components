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


#ifndef OPENMPF_COMPONENTS_DARKNETDETECTION_H
#define OPENMPF_COMPONENTS_DARKNETDETECTION_H

#include <string>
#include <vector>

#include <log4cxx/logger.h>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <DlClassLoader.h>
#include <ModelsIniParser.h>
#include <MPFDetectionObjects.h>

#include "include/DarknetInterface.h"
#include "Trackers.h"


class DarknetDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:
    bool Init() override;

    bool Close() override;

    std::vector<MPF::COMPONENT::MPFVideoTrack>  GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(
            const MPF::COMPONENT::MPFImageJob &job) override;


    std::string GetDetectionType() override;


private:
    using DarknetDl = MPF::COMPONENT::DlClassLoader<DarknetInterface>;
    using DarknetAsyncDl = MPF::COMPONENT::DlClassLoader<DarknetAsyncInterface>;

    log4cxx::LoggerPtr logger_;
    
    std::string cpu_darknet_lib_path_;

    std::string gpu_darknet_lib_path_;

    MPF::COMPONENT::ModelsIniParser<ModelSettings> models_parser_;


    DarknetAsyncDl GetDarknetImpl(const MPF::COMPONENT::MPFVideoJob &job);

    DarknetDl GetDarknetImpl(const MPF::COMPONENT::MPFImageJob &job);

    template <typename TDarknetDl>
    TDarknetDl GetDarknetImpl(const MPF::COMPONENT::MPFJob &job,
                              const std::string &creator, const std::string &deleter);


    static std::vector<MPF::COMPONENT::MPFImageLocation> ConvertResultsUsingPreprocessor(
            std::vector<DarknetResult> &darknet_results);

    ModelSettings GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetTracks(
            const MPF::COMPONENT::MPFJob &job,
            std::vector<DarknetResult> &&detections);
};







#endif //OPENMPF_COMPONENTS_DARKNETDETECTION_H
