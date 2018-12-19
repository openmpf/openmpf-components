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

#include <string>
#include <vector>
#include <iostream>
#include "TesseractOCRTextDetection.h"
#include "MPFComponentInterface.h"
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

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

int main(int argc, char* argv[]) {

  if ((argc < 3)) {
        std::cout << "Usage: " << argv[0] << " -i <IMAGE_DATA_URI> [TESSERACT_LANGUAGES]" << std::endl;
        std::cout << "Usage: " << argv[0] << " -g <GENERIC_DATA_URI> [TESSERACT_LANGUAGES]" << std::endl;
        return 0;
    }
  else{
	std::string option(argv[1]);
	std::string uri(argv[2]);

	Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("OCR_test");
    algorithm_properties["TAGGING_FILE"] = "text-tags.json";
    algorithm_properties["THRS_FILTER"] = "false";
    algorithm_properties["HIST_FILTER"] = "false";
    algorithm_properties["SHARPEN"] = "1.0";
    if (argc == 4) {
        algorithm_properties["TESSERACT_LANGUAGE"] = argv[3];
    }

    // Instantiate the component.
    TesseractOCRTextDetection im;
    im.SetRunDirectory("./plugin");
    im.Init();

    if (option == "-g") {
        // Run uri as a generic data file.
        std::cout << "Running job on generic data uri: " << uri << std::endl;
        MPFGenericJob job(job_name, uri, algorithm_properties, media_properties);
        // Declare the vector of image tracks to be filled in by the
        // component.
        std::vector<MPFGenericTrack> tracks;
        // Pass the job to the image detection component.
        MPFDetectionError rc = MPF_DETECTION_SUCCESS;
        rc = im.GetDetections(job, tracks);
        if (rc == MPF_DETECTION_SUCCESS) {
            std::cout << "Number of image tracks = "
                      << tracks.size() << std::endl;

            for (int i = 0; i < tracks.size(); i++) {
                std::cout << "OCR result: " << i << "\n"
                          << "   metadata = \"" << tracks[i].detection_properties.at("TEXT") << "\"" << std::endl;
                std::cout << "OCR text tags: " << i << "\n"
                          << "   detected string tags = \"" << tracks[i].detection_properties.at("TAGS") << "\"" << std::endl;
                std::cout << "OCR page_num: " << tracks[i].detection_properties.at("PAGE_NUM") << "\"" << std::endl;

            }
        } else {
            std::cout << "GetDetections failed" << std::endl;
        }
    } else if (option == "-i") {
        // Run uri as an image data file.
        std::cout << "Running job on image data uri: " << uri << std::endl;
        MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
        // Declare the vector of image locations to be filled in by the
        // component.
        std::vector<MPFImageLocation> locations;
        // Pass the job to the image detection component.
        MPFDetectionError rc = MPF_DETECTION_SUCCESS;
        rc = im.GetDetections(job, locations);
        if (rc == MPF_DETECTION_SUCCESS) {
            std::cout << "Number of image locations = "
                      << locations.size() << std::endl;

            for (int i = 0; i < locations.size(); i++) {
                std::cout << "OCR result: " << i << "\n"
                          << "   metadata = \"" << locations[i].detection_properties.at("TEXT") << "\"" << std::endl;
                std::cout << "OCR text tags: " << i << "\n"
                          << "   detected string tags = \"" << locations[i].detection_properties.at("TAGS") << "\"" << std::endl;
            }
        } else {
                std::cout << "GetDetections failed" << std::endl;
        }
     } else {
        std::cout << "Usage: " << argv[0] << " -i <IMAGE_DATA_URI> [TESSERACT_LANGUAGES]" << std::endl;
        std::cout << "Usage: " << argv[0] << " -g <GENERIC_DATA_URI> [TESSERACT_LANGUAGES]" << std::endl;
        return 0;
     }


  }

    return 0;
}
