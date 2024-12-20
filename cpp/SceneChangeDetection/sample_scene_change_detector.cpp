/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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
#include <sstream>
#include <algorithm>
#include <iterator>
#include <dlfcn.h>
#include <string>
#include <vector>
#include "SceneChangeDetection.h"

using namespace MPF::COMPONENT;
using namespace std;

int main(int argc, char* argv[]) {

    try {
        if ((argc < 2) || (argc > 4)) {
            std::cout << "argc = " << argc << std::endl;
            std::cout << "Usage: " << argv[0] << " <video URI> <start frame> <end frame>" << std::endl;
            return EXIT_SUCCESS;
        }

        SceneChangeDetection scene_change_component;
        scene_change_component.SetRunDirectory("plugin");

        if (!scene_change_component.Init()) {
            std::cout << "Component initialization failed, exiting." << std::endl;
            return EXIT_FAILURE;
        }

        Properties algorithm_properties;
        std::string uri(argv[1]);
        int start_frame = 0, stop_frame = 200;

        if (argc == 3) {
            std::cout << "Stop frame not provided. Setting stop frame to 200.\n";
            start_frame = std::stoi(argv[2]);
        }
        else if (argc == 4) {
            start_frame = std::stoi(argv[2]);
            stop_frame = std::stoi(argv[3]);
        }
        else {
            std::cout << "Start and stop frames not provided. Setting frame range to 0-200.\n";
        }

        Properties media_properties;
        std::string job_name("Testing Scene Change");
        std::cout << "testing scene change" << std::endl;
        MPFVideoJob job(job_name, uri, start_frame, stop_frame, algorithm_properties, media_properties);
        std::vector<MPFVideoTrack> detections = scene_change_component.GetDetections(job);
        std::cout << "number of final scenes: " << detections.size() << std::endl;
        for (int i = 0; i < detections.size(); i++) {
            std::cout << "scene number "
                      << i
                      << ": start frame is "
                      << detections[i].start_frame
                      << "; stop frame is "
                      << detections[i].stop_frame
                      << std::endl;
        }

        scene_change_component.Close();
        return EXIT_SUCCESS;
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
