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

#include "DarknetDetection.h"

#include <algorithm>
#include <exception>
#include <unordered_map>
#include <utility>

#include <log4cxx/xml/domconfigurator.h>

#include <MPFImageReader.h>
#include <MPFVideoCapture.h>
#include <detectionComponentUtils.h>
#include <Utils.h>

#include "Trackers.h"


using namespace MPF::COMPONENT;


bool DarknetDetection::Init() {
    std::string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }

    std::string plugin_path = run_dir + "/DarknetDetection";
    cpu_darknet_lib_path_ = plugin_path + "/lib/libdarknet_wrapper.so";
    gpu_darknet_lib_path_ = plugin_path + "/lib/libdarknet_wrapper_cuda.so";

    std::cout << "run_dir = " << run_dir << std::endl;
    std::cout << "plugin_path = " << plugin_path << std::endl;

    log4cxx::xml::DOMConfigurator::configure(plugin_path + "/config/Log4cxxConfig.xml");
    std::cout << "Configured log4cxx" << std::endl;
    std::cout << "Getting logger \"DarknetDetection\"" << std::endl;
    logger_ = log4cxx::Logger::getLogger("DarknetDetection");
    
    LOG4CXX_INFO(logger_, "Initializing models parser.");
    models_path_ = plugin_path + "/models";

    try {
        models_parser_.Init(plugin_path + "/models")
                .RegisterPathField("network_config", &ModelSettings::network_config_file)
                .RegisterPathField("names", &ModelSettings::names_file)
                .RegisterPathField("weights", &ModelSettings::weights_file);
    }
    catch (const std::exception &ex) {
        LOG4CXX_ERROR(logger_, "Failed to initialize ModelsIniParser due to: " << ex.what())
        return false;
    }

    LOG4CXX_INFO(logger_, "Initialized DarknetDetection component.");

    return true;
}



std::vector<MPFVideoTrack> DarknetDetection::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");

        if (DetectionComponentUtils::GetProperty(job.job_properties, "FRAME_QUEUE_CAPACITY", 4) <= 0) {
            LOG4CXX_ERROR(logger_, "[" << job.job_name
                    << "] : Detection failed: frame queue capacity property must be greater than 0");
            throw MPFDetectionException(MPF_INVALID_PROPERTY,
                    "Detection failed: frame queue capacity property must be greater than 0.");
        }

        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Attempting to open video from \"" << job.data_uri << "\"...")
        MPFVideoCapture video_cap(job);
        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Successfully opened video file.")

        DarknetAsyncDl detector = GetDarknetImpl(job);

        cv::Mat frame;
        int frame_number = -1;
        while (video_cap.Read(frame)) {
            frame_number++;
            detector->Submit(frame_number, frame);
        }
        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Read " << frame_number << " frames from video.")

        std::vector<MPFVideoTrack> tracks = GetTracks(job, detector->GetResults());

        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Successfully combined detections in to "
                << tracks.size() << " tracks.");

        // If we have tracks, see if we need to run a color characterizer function.
        if (tracks.size() > 0 &&
            DetectionComponentUtils::GetProperty(job.job_properties, "RUN_COLOR_CHARACTERIZER", false)) {
            const std::string &libname = DetectionComponentUtils::GetProperty<std::string>(
            job.job_properties, "COLOR_CHARACTERIZER_LIBRARY_NAME", "");
            if (libname.empty()) {
                throw std::runtime_error("Could not run a color characterizer because no color characterizer library name was provided. Please set the COLOR_CHARACTERIZER_LIBRARY_NAME property.");
            }
            std::unique_ptr<void, decltype(&dlclose)> handle(dlopen(libname.c_str(), RTLD_NOW), dlclose);
            if (NULL == handle) {
                std::string err(dlerror());
                throw std::runtime_error("Could not load library named " + libname + "because of " + err);
            }
            using color_func_t = void(*)(const MPF::COMPONENT::MPFVideoJob &job,
                                         const std::string &path,
                                         MPF::COMPONENT::MPFVideoCapture &video_cap,
                                         std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);
            const char *func_name = "runColorCharacterizer";
            color_func_t lib_func_ptr = reinterpret_cast<color_func_t>(dlsym(handle.get(), func_name));
            LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Running color characterizer");
            lib_func_ptr(job, models_path_, video_cap, tracks);
        }

        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Attempting to apply reverse transform to tracks...")
        for (MPFVideoTrack &track : tracks) {
            video_cap.ReverseTransform(track);
        }
        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Successfully applied reverse transform to tracks.")

        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << tracks.size() << " tracks.");

        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }
}


std::vector<MPFVideoTrack> DarknetDetection::GetTracks(const MPFJob &job, std::vector<DarknetResult> &&detections) {

    if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
        LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Attempting to generate tracks from " << detections.size()
                << " detections using PreprocessorTracker...")
        return PreprocessorTracker::GetTracks(std::move(detections));
    }

    int number_of_classifications = DetectionComponentUtils::GetProperty(
            job.job_properties, "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5);
    double rect_min_overlap = DetectionComponentUtils::GetProperty(
            job.job_properties, "MIN_OVERLAP", 0.5);
    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Attempting to generate tracks from " << detections.size()
                               << " detections using DefaultTracker...")
    return DefaultTracker::GetTracks(number_of_classifications, rect_min_overlap, std::move(detections));
}



std::vector<MPFImageLocation> DarknetDetection::GetDetections(const MPFImageJob &job) {
    try {
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");
        MPFImageReader image_reader(job);


        DarknetDl detector = GetDarknetImpl(job);
        std::vector<DarknetResult> results = detector->Detect(0, image_reader.GetImage());

        std::vector<MPFImageLocation> locations;
        if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
            locations = ConvertResultsUsingPreprocessor(results);
        }
        else {
            int number_of_classifications
                    = std::max(1, DetectionComponentUtils::GetProperty(job.job_properties,
                                                                       "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5));
            for (auto& result : results) {
                locations.push_back(
                        DefaultTracker::CreateImageLocation(number_of_classifications, result));
            }
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << locations.size() << " detections.");

        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }
}


DarknetDetection::DarknetAsyncDl DarknetDetection::GetDarknetImpl(const MPFVideoJob &job) {
    return GetDarknetImpl<DarknetAsyncDl>(job, "darknet_async_impl_creator", "darknet_async_impl_deleter");
}

DarknetDetection::DarknetDl DarknetDetection::GetDarknetImpl(const MPFImageJob &job) {
    return GetDarknetImpl<DarknetDl>(job, "darknet_impl_creator", "darknet_impl_deleter");
}


template <typename TDarknetDl>
TDarknetDl DarknetDetection::GetDarknetImpl(const MPFJob &job,
                                            const std::string &creator, const std::string &deleter) {

    const ModelSettings model_settings = GetModelSettings(job.job_properties);

    int cuda_device_id = DetectionComponentUtils::GetProperty(job.job_properties, "CUDA_DEVICE_ID", -1);
    if (cuda_device_id >= 0) {
        try {
            LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Attempting to load the GPU version of Darknet...")
            auto darknetDl = TDarknetDl(gpu_darknet_lib_path_, creator, deleter,
                                        &job.job_name, &job.job_properties, &model_settings, &logger_);
            LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Successfully loaded the GPU version of Darknet.")
            return darknetDl;
        }
        catch (const std::exception &ex) {
            if (DetectionComponentUtils::GetProperty(job.job_properties, "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", false)) {
                LOG4CXX_WARN(logger_, "[" << job.job_name
                        << "] An error occurred while trying to load the GPU version of Darknet: " << ex.what())
                LOG4CXX_WARN(logger_, "Falling back to CPU version.");
            }
            else {
                throw;
            }
        }
    }
    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Attempting to load the CPU version of Darknet...")
    auto darknetDl = TDarknetDl(cpu_darknet_lib_path_, creator, deleter,
                                &job.job_name, &job.job_properties, &model_settings, &logger_);
    LOG4CXX_DEBUG(logger_, "[" << job.job_name << "] Successfully loaded the CPU version of Darknet.")
    return darknetDl;
}



ModelSettings DarknetDetection::GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const {
    const std::string &model_name = DetectionComponentUtils::GetProperty<std::string>(
            job_properties, "MODEL_NAME", "tiny yolo");

    const std::string &models_dir_path = DetectionComponentUtils::GetProperty<std::string>(
            job_properties, "MODELS_DIR_PATH", ".");

    return models_parser_.ParseIni(model_name, models_dir_path + "/DarknetDetection");
}


std::vector<MPFImageLocation> DarknetDetection::ConvertResultsUsingPreprocessor(
        std::vector<DarknetResult> &darknet_results) {

    std::unordered_map<std::string, MPFImageLocation> type_to_image_loc;

    for (DarknetResult &darknet_result : darknet_results) {
        const cv::Rect &rect = darknet_result.detection_rect;

        for (std::pair<float, std::string> &class_prob : darknet_result.object_type_probs) {
            auto it = type_to_image_loc.find(class_prob.second);
            if (it == type_to_image_loc.cend()) {
                type_to_image_loc.emplace(
                        class_prob.second,
                        MPFImageLocation(rect.x, rect.y, rect.width, rect.height, class_prob.first,
                                         Properties{ {"CLASSIFICATION", class_prob.second} }));
            }
            else {
                PreprocessorTracker::CombineImageLocation(rect, class_prob.first, it->second);
            }
        }
    }
    std::vector<MPFImageLocation> locations;
    for (auto &image_loc_pair : type_to_image_loc) {
        locations.push_back(std::move(image_loc_pair.second));
    }
    return locations;
}


std::string DarknetDetection::GetDetectionType() {
    return "CLASS";
}



bool DarknetDetection::Close() {
    return true;
}


MPF_COMPONENT_CREATOR(DarknetDetection);
MPF_COMPONENT_DELETER();
