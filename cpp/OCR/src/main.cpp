/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2016 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2016 The MITRE Corporation                                       *
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
#include "ImageTransformer.h"
#include "MPFComponentInterface.h"

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

  if ((argc < 2 || argc > 3)) {
        std::cout << "Usage: " << argv[0] << " IMAGE_FILE_URI <ROTATE | CROP | FLIP>" << std::endl;
        return 0;
    }
  else{
 
	std::string uri(argv[1]);
	Properties algorithm_properties;

        
    Properties media_properties;
    std::string job_name("OCR_test");
    algorithm_properties["TAGGING_FILE"] = "text-tags.json";
    algorithm_properties["SHARPEN"] = "1.0";
    MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
    //SetRunDirectory(".");
    // Instantiate the component
    ImageTransformerComponent im;
    im.SetRunDirectory("./plugin");
    im.Init();
    // Declare the vector of image locations to be filled in by the
    // component.
    std::vector<MPFImageLocation> locations;
    // Pass the job to the image detection component
    MPFDetectionError rc = MPF_DETECTION_SUCCESS;
    rc = im.GetDetections(job, locations);
    if (rc == MPF_DETECTION_SUCCESS) {
        std::cout << "Number of image locations = "
                  << locations.size() << std::endl;

        for (int i = 0; i < locations.size(); i++) {
            std::cout << "OCR result: " << i << "\n"
                      << "   metadata = \"" << locations[i].detection_properties.at("TEXT") << "\"" << std::endl;
            std::cout << "OCR tags: " << i << "\n"
                      << "   string tags = \"" << locations[i].detection_properties.at("TAGS_STRING") << "\"" << std::endl;          
                      std::cout << "OCR tags: " << i << "\n"
                      << "   regex tags = \"" << locations[i].detection_properties.at("TAGS_REGEX") << "\"" << std::endl;          
        }
    }
    else {
        std::cout << "GetDetections failed" << std::endl;
    }
  }

    return 0;
}


