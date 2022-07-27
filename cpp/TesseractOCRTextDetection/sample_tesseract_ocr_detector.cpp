/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
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

#include <string>
#include <vector>
#include <iostream>
#include "TesseractOCRTextDetection.h"

/**
 * NOTE: This main is only intended to serve as a test harness for compiling a
 * stand-alone binary to debug the component logic independently of MPF.
 * MPF requires that the component logic be compiled into a shared object
 * library that is then dynamically loaded into a common detection component
 * executable.
 */

using namespace MPF;
using namespace COMPONENT;

using std::to_string;

void print_usage(char *argv[]) {

    std::cout << "Usage: " << argv[0] <<
                 " <-i | -v | -g> [--osd] [--oem TESSERACT_OEM] <IMAGE_URI | VIDEO_URI <START_FRAME> <END_FRAME> | GENERIC_URI>  [TESSERACT_LANGUAGE]" <<
                 std::endl << std::endl;
    std::cout << "Notes: " << std::endl << std::endl;
    std::cout << " <-i | -v | -g>  : Specifies whether to process an image (-i <IMAGE_URI>), video (-v <VIDEO_URI>  <START_FRAME> <END_FRAME>), or generic document (-g <GENERIC_URI>)." <<
                 std::endl << std::endl;
    std::cout << " --osd   : When provided, runs the job with automatic orientation and script detection (OSD). " <<
                 std::endl;
    std::cout << "           Input tesseract languages are generally ignored" <<
                 " whenever OSD returns successful predictions and can be left out." << std::endl << std::endl;

    std::cout << " --oem TESSERACT_OEM : When provided runs the job with the specified TESSERACT_OEM engine mode" <<
                 std::endl <<
                 "                       Tesseract currently supports legacy (0) lstm (1), lstm + legacy (2)," <<
                 " and default (3)." << std::endl <<
                 "                       Default (OEM = 3) setting uses whichever language engine is currently available." <<
                 std::endl << std::endl;

    std::cout << "  TESSERACT_LANGUAGE : When provided, sets the default TESSERACT_LANGUAGE to the given value." <<
                 std::endl << std::endl;

}

void print_detection_properties(Properties properties, float confidence) {
    if (properties.count("OSD_PRIMARY_SCRIPT") > 0) {
        std::cout << "OSD result:" << std::endl
                  << "    OSD fallback occurred: " << properties.at("OSD_FALLBACK_OCCURRED") << std::endl
                  << "    Detected script: " << properties.at("OSD_PRIMARY_SCRIPT") << std::endl
                  << "    Script confidence: " << properties.at("OSD_PRIMARY_SCRIPT_CONFIDENCE") << std::endl
                  << "    Script score: " << properties.at("OSD_PRIMARY_SCRIPT_SCORE") << std::endl
                  << "    Detected orientation: " << properties.at("ROTATION") << std::endl
                  << "    Orientation confidence: " << properties.at("OSD_TEXT_ORIENTATION_CONFIDENCE") << std::endl;
        if (properties.count("ROTATE_AND_DETECT_PASS") > 0) {
            std::cout << "    Orientation pass: " << properties.at("ROTATE_AND_DETECT_PASS") << std::endl;
        }
        if (properties.count("OSD_SECONDARY_SCRIPTS") > 0) {
            std::cout << "    Secondary scripts: " << properties.at("OSD_SECONDARY_SCRIPTS") << std::endl
                      << "    Secondary script scores: " << properties.at("OSD_SECONDARY_SCRIPT_SCORES") << std::endl;
        }
    }
    if (properties.count("MISSING_LANGUAGE_MODELS") > 0){
        std::cout << "Missing language models: " << properties.at("MISSING_LANGUAGE_MODELS") << std::endl;
    }
    if (properties.count("TEXT") > 0) {
        std::cout << "OCR result:" << std::endl
                  << "    Text: \"" << properties.at("TEXT") << "\"" << std::endl
                  << "    OCR language: " << properties.at("TEXT_LANGUAGE") << std::endl
                  << "    Confidence: " << confidence << std::endl;
    }
    std::cout << std::endl;
}

bool check_options(const std::string &next_option,  const int &argc, char *argv[],
                   Properties &algorithm_properties, int &uri_index){

    if (next_option == "--osd") {
        algorithm_properties["ENABLE_OSD_AUTOMATION"] = "true";
        uri_index++;
    } else if (next_option == "--oem" && argc - uri_index > 2) {
        std::cout << "Updating OEM MODE " << argv[uri_index + 1] << std::endl;
        algorithm_properties["TESSERACT_OEM"] = argv[uri_index + 1];
        uri_index += 2;
    } else {
        return false;
    }
    return true;

}

int main(int argc, char *argv[]) {
    try {
        if ((argc < 3)) {
            print_usage(argv);
            return 0;
        }

        std::string media_option(argv[1]);
        std::string uri = "";

        Properties algorithm_properties;
        Properties media_properties;
        std::string job_name("OCR_test");
        algorithm_properties["THRS_FILTER"] = "false";
        algorithm_properties["HIST_FILTER"] = "false";
        algorithm_properties["SHARPEN"] = "1.0";
        algorithm_properties["ENABLE_OSD_AUTOMATION"] = "false";

        int uri_index = 2, video_params = 0, start_frame = 0, end_frame = 1;

        std::string next_option = std::string(argv[uri_index]);
        if (check_options(next_option, argc, argv, algorithm_properties, uri_index)) {
            next_option = std::string(argv[uri_index]);
            check_options(next_option, argc, argv, algorithm_properties, uri_index);
        }

        if (media_option == "-v") {
            video_params = 2;
            if (argc - uri_index < 3) {
                print_usage(argv);
                return 0;

            }
            start_frame = std::stoi(argv[uri_index+1]);
            end_frame = std::stoi(argv[uri_index+2]);
        }

        if (argc - uri_index - video_params == 1) {
            uri = argv[uri_index];
        } else if (argc - uri_index - video_params == 2) {
            uri = argv[uri_index];
            algorithm_properties["TESSERACT_LANGUAGE"] = argv[uri_index + video_params + 1];
        } else {
             print_usage(argv);
             return 0;
        }

        // Instantiate the component.
        TesseractOCRTextDetection im;
        im.SetRunDirectory("./plugin");
        im.Init();

        if (media_option == "-g") {
            // Run uri as a generic data file.
            std::cout << "Running job on generic data uri: " << uri << std::endl;
            MPFGenericJob job(job_name, uri, algorithm_properties, media_properties);

            std::vector<MPFGenericTrack> tracks = im.GetDetections(job);
            std::cout << "Number of tracks: " << tracks.size() << std::endl << std::endl;
            for (int i = 0; i < tracks.size(); i++) {
                std::cout << "Page number: " << tracks[i].detection_properties.at("PAGE_NUM") << std::endl;
                print_detection_properties(tracks[i].detection_properties, tracks[i].confidence);
            }
        }
        else if (media_option == "-i") {
            // Run uri as an image data file.
            std::cout << "Running job on image data uri: " << uri << std::endl;
            MPFImageJob job(job_name, uri, algorithm_properties, media_properties);

            std::vector<MPFImageLocation> locations = im.GetDetections(job);
            std::cout << "Number of image locations: " << locations.size() << std::endl << std::endl;
            for (int i = 0; i < locations.size(); i++) {
                print_detection_properties(locations[i].detection_properties, locations[i].confidence);
            }
        }
        else if (media_option == "-v") {
            // Run uri as an image data file.
            std::cout << "Running job on video data uri: " << uri << std::endl;
            MPFVideoJob job(job_name, uri, start_frame, end_frame, algorithm_properties, media_properties);
            int count = 0;
            for (auto track: im.GetDetections(job)) {
                std::cout << "Track number: " << count << std::endl;
                std::map<int, MPFImageLocation> locations = track.frame_locations;
                std::cout << "Number of image locations: " << locations.size() << std::endl << std::endl;
                for (const auto &location: locations) {
                    std::cout << "Frame number: " << location.first << std::endl;
                    print_detection_properties(location.second.detection_properties, location.second.confidence);
                }
                count ++;
            }
        }
        else {
            print_usage(argv);
        }

        im.Close();
        return 0;
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}
