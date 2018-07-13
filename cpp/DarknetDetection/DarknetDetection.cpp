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

#include <fstream>

#include <log4cxx/xml/domconfigurator.h>

#include <detectionComponentUtils.h>
#include <Utils.h>

#include "DarknetDetection.h"

using namespace MPF::COMPONENT;


bool DarknetDetection::Init() {
    std::string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }

    plugin_path_ = run_dir + "/DarknetDetection";

    log4cxx::xml::DOMConfigurator::configure(plugin_path_ + "/config/Log4cxxConfig.xml");
    logger_ = log4cxx::Logger::getLogger("DarknetDetection");

    try {
        models_parser_.Init(plugin_path_ + "/models")
                .RegisterField("network_config", &ModelSettings::network_config_file)
                .RegisterField("names", &ModelSettings::names_file)
                .RegisterField("weights", &ModelSettings::weights_file);
    }
    catch (const std::exception &ex) {
        LOG4CXX_ERROR(logger_, "Failed to initialize ModelsIniParser due to: " << ex.what())
        return false;
    }

    LOG4CXX_INFO(logger_, "Initialized DarknetDetection component.");

    return true;
}


MPFDetectionError DarknetDetection::GetDetections(const MPFVideoJob &job, std::vector<MPFVideoTrack> &tracks) {
    LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");

    try {
        DarknetDl detector = GetDarknetImpl(job);

        MPFDetectionError rc = detector->RunDarknetDetection(job, tracks);
        if (rc != MPF_DETECTION_SUCCESS) {
            LOG4CXX_ERROR(logger_, "[" << job.job_name << "] Darknet detection fialed for file: " << job.data_uri);
        }
        
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << tracks.size() << " tracks.");
        return rc;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }

}


MPFDetectionError DarknetDetection::GetDetections(const MPFImageJob &job, std::vector<MPFImageLocation> &locations) {
    try {
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");
        DarknetDl detector = GetDarknetImpl(job);

        MPFDetectionError rc = detector->RunDarknetDetection(job, locations);
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << locations.size() << " detections.");
        return rc;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }

}


DarknetDetection::DarknetDl DarknetDetection::GetDarknetImpl(const MPF::COMPONENT::MPFJob &job) {
    static std::string cpu_darknet_lib = plugin_path_ + "/lib/libdarknet_wrapper.so";
    static std::string gpu_darknet_lib = plugin_path_ + "/lib/libdarknet_wrapper_cuda.so";
    static std::string darknet_impl_creator = "darknet_impl_creator";
    static std::string darknet_impl_deleter = "darknet_impl_deleter";

    const ModelSettings model_settings = GetModelSettings(job.job_properties);

    int cuda_device_id = DetectionComponentUtils::GetProperty(job.job_properties, "CUDA_DEVICE_ID", -1);
    if (cuda_device_id >= 0) {
        try {
            return DarknetDl(gpu_darknet_lib, darknet_impl_creator, darknet_impl_deleter,
                             &job.job_properties, &model_settings);
        }
        catch (const std::exception &ex) {
            if (DetectionComponentUtils::GetProperty(job.job_properties, "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", false)) {
                LOG4CXX_WARN(logger_,
                             "An error occured while trying to load the GPU version of Darknet: " << ex.what());
                LOG4CXX_WARN(logger_, "Falling back to CPU version.");
            }
            else {
                throw;
            }
        }
    }
    return DarknetDl(cpu_darknet_lib, darknet_impl_creator, darknet_impl_deleter,
                     &job.job_properties, &model_settings);
}

ModelSettings DarknetDetection::GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const {
    const std::string &model_name = DetectionComponentUtils::GetProperty<std::string>(
            job_properties, "MODEL_NAME", "tiny yolo");

    const std::string &models_dir_path = DetectionComponentUtils::GetProperty<std::string>(
            job_properties, "MODELS_DIR_PATH", ".");

    return models_parser_.ParseIni(model_name, models_dir_path + "/DarknetDetection");
}




std::string DarknetDetection::GetDetectionType() {
    return "CLASS";
}



bool DarknetDetection::Close() {
    return true;
}


MPF_COMPONENT_CREATOR(DarknetDetection);
MPF_COMPONENT_DELETER();
