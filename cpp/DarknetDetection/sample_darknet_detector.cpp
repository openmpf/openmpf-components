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

#include <iostream>
#include <QCoreApplication>
#include <iomanip>
#include <chrono>
#include <opencv2/core/cuda.hpp>


#include "DarknetDetection.h"


using namespace MPF::COMPONENT;

void print_tracks(const std::vector<MPFVideoTrack> &tracks);

int main(int argc, char* argv[]) {
    if (argc == 2) {
        std::string arg1 = argv[1];
        if (arg1 == "gpu-info") {
            int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
            std::cout << "Cuda device count: " << cuda_device_count << std::endl;
            if (cuda_device_count > 0) {
                for (int i = 0; i < cuda_device_count; i++) {
                    std::cout << "==== Device #" << i << " ====" << std::endl;
                    cv::cuda::printCudaDeviceInfo(i);
                    std::cout << "=================================" << std::endl;
                }
            }
            return EXIT_SUCCESS;
        }
    }

    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <uri> <model_name> [gpu_index]" << std::endl;
        std::cout << "Usage: " << argv[0] << " <uri> <model_name> <start_frame> <end_frame> [gpu_index] [queue_capacity]" << std::endl;
        std::cout << "Usage: " << argv[0] << " gpu-info" << std::endl;
        return 1;
    }

    std::string uri = argv[1];
    std::string model_name = argv[2];
    std::string gpu_index = "-1";
    std::string queue_capacity = "4";
    int start_frame = 0;
    int end_frame = -1;

    if (argc == 4) {
        gpu_index = argv[3];
    }
    if (argc >= 5) {
        start_frame = std::stoi(argv[3]);
        end_frame = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        gpu_index = argv[5];
    }
    if (argc >= 7) {
        queue_capacity = argv[6];
    }

    std::cout << "URI: " << uri << std::endl;
    std::cout << "model name: " << model_name << std::endl;
    std::cout << "GPU Index: " << gpu_index << std::endl;
    std::cout << "queue capacity: " << queue_capacity << std::endl;
    std::cout << "start frame: " << start_frame << std::endl;
    std::cout << "end frame: " << end_frame << std::endl;

    std::string app_dir = QCoreApplication(argc, argv).applicationDirPath().toStdString();

    DarknetDetection detector;
    detector.SetRunDirectory(app_dir + "/plugin");

    if (!detector.Init()) {
        std::cout << "Init failed" << std::endl;
        return 1;
    }

    MPFVideoJob job("Test", uri, start_frame, end_frame,
                    { {"CUDA_DEVICE_ID", gpu_index},
                      {"MODEL_NAME", model_name},
                      {"FRAME_QUEUE_CAPACITY", queue_capacity} },
                    {});


    try {
        auto job_start_time = std::chrono::system_clock::now();
        std::vector<MPFVideoTrack> tracks = detector.GetDetections(job);
        auto job_stop_time = std::chrono::system_clock::now();

        std::chrono::duration<float, std::chrono::seconds::period> job_duration = job_stop_time - job_start_time;
        std::cout << "Found " << tracks.size() << " tracks in " << job_duration.count() << " seconds." << std::endl;

        print_tracks(tracks);

        return 0;
    }
    catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


void print_tracks(const std::vector<MPFVideoTrack> &tracks) {
    if (tracks.empty()) {
        return;
    }

    std::map<std::string, std::pair<int, float>> grouped_results;

    for (const auto &track : tracks) {
        const auto &classification = track.detection_properties.at("CLASSIFICATION");

        auto &pair = grouped_results[classification];
        pair.first++;
        pair.second = std::max(pair.second, track.confidence);
    }


    std::cout << '\n';
    std::cout << std::left << std::setw(12) << "Class";
    std::cout << std::left << std::setw(12) << "Confidence";
    std::cout << std::right << std::setw(6) << "Count" << std::endl;
    std::cout << "----------------------------------" << std::endl;
    for (const auto &group : grouped_results) {
        std::cout << std::left << std::setw(12) << group.first;
        std::cout << std::left << std::setw(12) << group.second.second;
        std::cout << std::right << std::setw(6) << group.second.first << std::endl;
    }
}
