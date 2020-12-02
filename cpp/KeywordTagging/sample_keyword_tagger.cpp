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
#include "KeywordTagging.h"

using namespace MPF::COMPONENT;


int main(int argc, char *argv[]) {
    try {
        if ((argc != 2)) {
            std::cout << "Usage: " << argv[0] << " DATA_URI" << std::endl;
            return 1;
        }

        std::string uri(argv[1]);
        std::string job_name("tagger_test");

        Properties algorithm_properties;
        algorithm_properties["TAGGING_FILE"] = "text-tags.json";

        KeywordTagging tagger;
        tagger.SetRunDirectory("./plugin");
        tagger.Init();

        MPFGenericJob job(job_name, uri, algorithm_properties, {});

        std::vector<MPFGenericTrack> tracks;
        tracks = tagger.GetDetections(job);

        if (tracks.size() != 1) {
            std::cerr << "Unexpected number of tracks: " << tracks.size() << std::endl;
            return 1;
        }

        Properties props = tracks.at(0).detection_properties;
        std::string text = props["TEXT"];

        if (text.empty()) {
            std::cout << "Empty text file." << std::endl;
        } else {
            std::cout << "TEXT: " << std::endl;
            std::cout << text << std::endl;
            std::cout << std::endl;

            if (props.find("TAGS") != props.end()) {
                std::cout << "TAGS: " << props["TAGS"] << std::endl;
            }
            if (props.find("TRIGGER_WORDS") != props.end()) {
                std::cout << "TRIGGER_WORDS: " << props["TRIGGER_WORDS"] << std::endl;
            }
            if (props.find("TRIGGER_WORDS_OFFSET") != props.end()) {
                std::cout << "TRIGGER_WORDS_OFFSET: " << props["TRIGGER_WORDS_OFFSET"] << std::endl;
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