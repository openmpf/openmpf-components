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

#include "DarknetStreamingDetection.h"

#include <exception>
#include <sstream>
#include <utility>

#include <ModelsIniParser.h>
#include <detectionComponentUtils.h>

#include "include/DarknetInterface.h"
#include "Trackers.h"


using namespace MPF::COMPONENT;


namespace {

    ModelSettings GetModelSettings(const MPFStreamingVideoJob &job) {
        const std::string &model_name = DetectionComponentUtils::GetProperty<std::string>(
                job.job_properties, "MODEL_NAME", "tiny yolo");

        const std::string &models_dir_path = DetectionComponentUtils::GetProperty<std::string>(
                job.job_properties, "MODELS_DIR_PATH", ".");

        return ModelsIniParser<ModelSettings>()
                .Init(job.run_directory + "/DarknetDetection/models")
                .RegisterPathField("network_config", &ModelSettings::network_config_file)
                .RegisterPathField("names", &ModelSettings::names_file)
                .RegisterPathField("weights", &ModelSettings::weights_file)
                .ParseIni(model_name, models_dir_path + "/DarknetDetection");
    }


    DlClassLoader<DarknetInterface> GetDarknetImpl(const MPFStreamingVideoJob &job, log4cxx::LoggerPtr &logger) {
        static const std::string creator_fn_name = "darknet_impl_creator";
        static const std::string deleter_fn_name = "darknet_impl_deleter";

        const ModelSettings model_settings = GetModelSettings(job);

        int cuda_device_id = DetectionComponentUtils::GetProperty(job.job_properties, "CUDA_DEVICE_ID", -1);
        if (cuda_device_id >= 0) {
            try {
                std::string gpu_darknet_lib_path
                        = job.run_directory + "/DarknetDetection/lib/libdarknet_wrapper_cuda.so";
                return { gpu_darknet_lib_path, creator_fn_name, deleter_fn_name,
                            &job.job_name, &job.job_properties, &model_settings, &logger };
            }
            catch (const std::exception &ex) {
                if (DetectionComponentUtils::GetProperty(job.job_properties, "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", false)) {
                    LOG4CXX_WARN(logger,
                                 "An error occurred while trying to load the GPU version of Darknet: " << ex.what());
                    LOG4CXX_WARN(logger, "Falling back to CPU version.");
                }
                else {
                    throw;
                }
            }
        }
        std::string cpu_darknet_lib_path
                = job.run_directory + "/DarknetDetection/lib/libdarknet_wrapper.so";
        return { cpu_darknet_lib_path, creator_fn_name, deleter_fn_name,
                    &job.job_name, &job.job_properties, &model_settings, &logger };
    }

    std::function< std::vector<MPFVideoTrack> (std::vector<DarknetResult>&&) >
    GetTracker(const MPFStreamingVideoJob &job) {
        if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
            return PreprocessorTracker::GetTracks;
        }

        int number_of_classifications = DetectionComponentUtils::GetProperty(
                job.job_properties, "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5);
        double rect_min_overlap = DetectionComponentUtils::GetProperty(
                job.job_properties, "MIN_OVERLAP", 0.5);

        return [=](std::vector<DarknetResult>&& detections) {
            return DefaultTracker::GetTracks(number_of_classifications, rect_min_overlap, std::move(detections));
        };
    }


    [[noreturn]] void LogError(const std::string &message, log4cxx::LoggerPtr &logger) {
        try {
            throw;
        }
        catch (const std::exception &ex) {
            LOG4CXX_ERROR(logger, message << ": " << ex.what());
            throw;
        }
        catch (...) {
            LOG4CXX_ERROR(logger, message << ".");
            throw;
        }
    }
} // end anonymous namespace



DarknetStreamingDetection::DarknetStreamingDetection(const MPFStreamingVideoJob &job)
    : DarknetStreamingDetection(job, log4cxx::Logger::getLogger("DarknetStreamingDetection"))
{
}


DarknetStreamingDetection::DarknetStreamingDetection(const MPFStreamingVideoJob &job, log4cxx::LoggerPtr logger)
try
        : MPFStreamingDetectionComponent(job)
        , logger_(logger)
        , job_name_(job.job_name)
        , log_prefix_("[" + job.job_name + "] ")
        , detector_(GetDarknetImpl(job, logger_))
        , tracker_(GetTracker(job))
{
}
catch (...) {
    ::LogError("An error occurred while initializing job \"" + job.job_name + "\"", logger);
}


std::string DarknetStreamingDetection::GetDetectionType() {
    return "CLASS";
}


void DarknetStreamingDetection::BeginSegment(const VideoSegmentInfo &segment_info) {
    std::ostringstream ss;
    ss << "[" << job_name_ << ": Segment #" << segment_info.segment_number
            << " (" << segment_info.start_frame << " - " << segment_info.end_frame << ")] ";
    log_prefix_ = ss.str();
}


bool DarknetStreamingDetection::ProcessFrame(const cv::Mat &frame, int frame_number) {
    try {
        detector_->Detect(frame_number, frame, current_segment_detections_);
        if (current_segment_detections_.empty()) {
            return false;
        }
        if (found_track_in_current_segment_) {
            return false;
        }
        LOG4CXX_INFO(logger_, log_prefix_ << "Found first detection in segment in frame number: " << frame_number);
        found_track_in_current_segment_ = true;
        return true;
    }
    catch (...) {
        LogError("An error occurred while processing frame " + std::to_string(frame_number));
    }
}


std::vector<MPF::COMPONENT::MPFVideoTrack> DarknetStreamingDetection::EndSegment() {
    try {
        auto detection_count = current_segment_detections_.size();

        auto tracks = tracker_(std::move(current_segment_detections_));
        LOG4CXX_INFO(logger_, log_prefix_ << "End segment. " << tracks.size() << " tracks reported.")

        current_segment_detections_ = {};
        // Call reserve to prevent the vector from doing extra resizing.
        // Assumes that the next segment will have a similar number of detections.
        current_segment_detections_.reserve(detection_count);

        found_track_in_current_segment_ = false;
        return tracks;
    }
    catch (...) {
        LogError("An error occurred while generating tracks for segment");
    }
}

void DarknetStreamingDetection::LogError(const std::string &message) {
    ::LogError(log_prefix_ + message, logger_);
}


EXPORT_MPF_STREAMING_COMPONENT(DarknetStreamingDetection);
