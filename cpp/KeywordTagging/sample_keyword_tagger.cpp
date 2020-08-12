/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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
#include "KeywordTaggingComponent.h"

using namespace MPF::COMPONENT;


int main(int argc, char *argv[]) {
    try {
        if ((argc != 2)) {
            std::cout << "Usage: " << argv[0] << " DATA_URI" << std::endl;
        }

        std::string uri(argv[1]);
        std::string job_name("tagger_test");

        Properties algorithm_properties;
        Properties media_properties;
        MPFDetectionError rc = MPF_DETECTION_SUCCESS;

        algorithm_properties["TAGGING_FILE"] = "text-tags.json";
        MPFGenericTrack text_tags;
        text_tags.detection_properties["TEXT"] = "Passenger Passport";
        text_tags.detection_properties["EXTRA"] = "extra property";


        KeywordTagger tagger;

        tagger.SetRunDirectory("./plugin");
        tagger.Init();

        MPFGenericJob job(job_name, uri, text_tags, algorithm_properties, media_properties);

        std::vector<MPFGenericTrack> tracks;

        tracks = tagger.GetDetections(job);
        if (rc == MPF_DETECTION_SUCCESS) {
            std::cout << "Number of generic tracks = " << tracks.size() << std::endl;

            for(int i = 0; i < tracks.size(); i++) {
                std::cout << "tags: " << tracks[i].detection_properties.at("TAGS") << std::endl;
                std::cout << "extra property: " << tracks[i].detection_properties.at("EXTRA") << std::endl;
                std::cout << "text: " << tracks[i].detection_properties.at("TEXT") << std::endl;
            }
        }

        tagger.Close();
        return 0;
    }
    catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }
}