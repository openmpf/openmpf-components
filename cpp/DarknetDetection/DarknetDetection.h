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


#ifndef OPENMPF_COMPONENTS_DARKNETDETECTION_H
#define OPENMPF_COMPONENTS_DARKNETDETECTION_H

#include <memory>

#include <log4cxx/logger.h>
#include <opencv2/core.hpp>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <ModelsIniParser.h>
#include <dlfcn.h>

#include "Trackers.h"

#include "include/DarknetInterface.h"


class DarknetDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:
    bool Init() override;

    bool Close() override;

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks) override;

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFImageJob &job,
            std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;


    std::string GetDetectionType() override;


private:
    class DarknetDl;

    log4cxx::LoggerPtr logger_;
    
    std::string plugin_path_;

    MPF::COMPONENT::ModelsIniParser<ModelSettings> models_parser_;

    DarknetDl GetDarknetImpl(const MPF::COMPONENT::MPFJob &job);

    template <typename Tracker>
    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
            Tracker &tracker);

    static void ConvertResultsUsingPreprocessor(std::vector<DarknetResult> &darknet_results,
                                                std::vector<MPF::COMPONENT::MPFImageLocation> &locations);

    ModelSettings GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const;
    


    class DarknetDl {
    public:
        DarknetDl(const std::string &lib_path, const MPF::COMPONENT::MPFJob &job, const ModelSettings &model_settings);

        std::vector<DarknetResult> Detect(const cv::Mat &cv_image);


    private:
        using darknet_impl_t = std::unique_ptr<DarknetInterface, void(*)(DarknetInterface*)>;

        std::unique_ptr<void, decltype(&dlclose)> lib_handle_;

        darknet_impl_t darknet_impl_;

        static darknet_impl_t LoadDarknetImpl(void* lib_handle, const MPF::COMPONENT::MPFJob &job,
                                              const ModelSettings &model_settings);

        template <typename TFunc>
        static TFunc* LoadFunction(void* lib_handle, const char * symbol_name);
    };
};







#endif //OPENMPF_COMPONENTS_DARKNETDETECTION_H
