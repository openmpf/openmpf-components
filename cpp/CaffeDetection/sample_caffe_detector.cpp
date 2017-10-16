/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2017 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2017 The MITRE Corporation                                       *
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
#include <dlfcn.h>
#include <string>
#include <vector>
#include "CaffeDetection.h"

using namespace MPF::COMPONENT;

int main(int argc, char* argv[]) {

    if ((argc < 2) || (argc > 5)) {
        std::cout << "argc = " << argc << std::endl;
        std::cout << "Usage: " << argv[0] << " <image URI> <num classifications> <confidence threshold> <ROTATE | CROP | FLIP>" << std::endl;
        return EXIT_SUCCESS;
    }

    CaffeDetection caffe_component;

    caffe_component.SetRunDirectory("plugin");

    if (!caffe_component.Init()) {
        std::cout << "Component initialization failed, exiting." << std::endl;
        return EXIT_FAILURE;
    }

    Properties algorithm_properties;

    // DEBUG
    algorithm_properties["MODEL_NAME"] = "googlenet";
    algorithm_properties["NUMBER_OF_CLASSIFICATIONS"] = "10";
    algorithm_properties["RESIZE_HEIGHT"] = "224";
    algorithm_properties["RESIZE_WIDTH"] = "224";
    algorithm_properties["SUBTRACT_BLUE_VALUE"] = "104.0";
    algorithm_properties["SUBTRACT_GREEN_VALUE"] = "117.0";
    algorithm_properties["SUBTRACT_RED_VALUE"] = "123.0";

    /*
    // DEBUG
    algorithm_properties["MODEL_NAME"] = "yahoo_nsfw";
    algorithm_properties["NUMBER_OF_CLASSIFICATIONS"] = "2";
    algorithm_properties["RESIZE_HEIGHT"] = "256";
    algorithm_properties["RESIZE_WIDTH"] = "256";
    algorithm_properties["TOP_AND_BOTTOM_CROP"] = "16";
    algorithm_properties["LEFT_AND_RIGHT_CROP"] = "16";
    algorithm_properties["SUBTRACT_BLUE_VALUE"] = "104.0";
    algorithm_properties["SUBTRACT_GREEN_VALUE"] = "117.0";
    algorithm_properties["SUBTRACT_RED_VALUE"] = "123.0";
    */

    MPFDetectionDataType media_type = IMAGE;
    std::string uri(argv[1]);
    std::cout << "uri is " << uri << std::endl;

    if (argc > 2) {
        // read the number of classifications returned
        std::string num_classes(argv[2]);
        algorithm_properties["NUMBER_OF_CLASSIFICATIONS"] = num_classes;
        std::cout << "Number of classifications = " << num_classes << std::endl;
    }

    if (argc > 3) {
        // read the confidence threshold
        std::string threshold(argv[3]);
        algorithm_properties["CONFIDENCE_THRESHOLD"] = threshold;
        std::cout << "Confidence threshold = " << threshold << std::endl;
    }

    if (argc > 4) {
        std::string transformation(argv[4]);

        if (transformation == "ROTATE") {
            // The input image will be rotated 270 degrees
            algorithm_properties["ROTATION"] = std::to_string(270);
            std::cout << "Rotating the image by " << algorithm_properties["ROTATION"] << " degrees" << std::endl;
        }
        else if (transformation == "CROP") {
            // The input image will be cropped to a 100x100 pixel rectangle
            // with the upper left corner at 100,100
            algorithm_properties["SEARCH_REGION_TOP_LEFT_X_DETECTION"] = std::to_string(100);
            algorithm_properties["SEARCH_REGION_TOP_LEFT_Y_DETECTION"] = std::to_string(100);
            algorithm_properties["SEARCH_REGION_BOTTOM_RIGHT_X_DETECTION"] = std::to_string(200);
            algorithm_properties["SEARCH_REGION_BOTTOM_RIGHT_Y_DETECTION"] = std::to_string(200);
            algorithm_properties["SEARCH_REGION_ENABLE_DETECTION"] = "true";
            std::cout << "Cropping the image" << std::endl;
        }
        else if (transformation == "FLIP") {
            // The input image will be flipped horizontally, i.e., left to right
            algorithm_properties["HORIZONTAL_FLIP"] = "true";
            std::cout << "Flipping the image" << std::endl;
        }
    }

    Properties media_properties;
    std::string job_name("Testing Caffe");

    std::vector<MPFImageLocation> detections;
    MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
    caffe_component.GetDetections(job, detections);
    for (int i = 0; i < detections.size(); i++) {
        std::cout << "detection number "
                  << i
                  << " classification is "
                  << detections[i].detection_properties["CLASSIFICATION"]
                  << " and confidence is "
                  << detections[i].confidence << std::endl
                  << " classifications list: "
                  << detections[i].detection_properties["CLASSIFICATION LIST"]
                  << std::endl
                  << " classifications confidence list: "
                  << detections[i].detection_properties["CLASSIFICATION CONFIDENCE LIST"]
                  << std::endl;
    }
    caffe_component.Close();
    return EXIT_SUCCESS;
}

