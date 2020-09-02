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
#include <string>
#include <vector>
#include "TrtisDetection.h"

using namespace MPF::COMPONENT;
using namespace std;

int main(int argc, char *argv[]) {

    if ((argc < 2) || (argc > 4)) {
        cout << "argc = " << argc << endl;
        cout << "Usage: " << argv[0] << " <image URI> <trtis_server> <confidence threshold>"
             << endl;
        return EXIT_FAILURE;
    }

    TrtisDetection trtis_component;
    trtis_component.SetRunDirectory("plugin");

    if (!trtis_component.Init()) {
        cout << "Component initialization failed, exiting." << endl;
        return EXIT_FAILURE;
    }

    Properties algorithm_properties;

    // DEBUG
    algorithm_properties["TRTIS_SERVER"] = "localhost:8001";
    algorithm_properties["MODEL_NAME"] = "ip_irv2_coco";
    algorithm_properties["CONFIDENCE_THRESHOLD"] = "0.2";
    algorithm_properties["RESIZE"] = "true";

    string uri(argv[1]);
    cout << "Media = " << uri << endl;

    if (argc > 2) { algorithm_properties["TRTIS_SERVER"] = string(argv[2]); }
    cout << "TRTIS server name and port = "
         << algorithm_properties["TRTIS_SERVER"] << endl;

    if (argc > 3) { algorithm_properties["CONFIDENCE_THRESHOLD"] = string(argv[3]); }
    cout << "Confidence threshold = " << algorithm_properties["CONFIDENCE_THRESHOLD"] << endl;

    Properties media_properties;
    string job_name("Testing TRTIS");

    MPFImageLocationVec detections;
    MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
    detections = trtis_component.GetDetections(job);

    cout << "Found " << detections.size() << " detections." << endl;

    for (int i = 0; i < detections.size(); i++) {
        string feature_type = detections[i].detection_properties["FEATURE TYPE"];
        cout << "Detection number " << i << ":\n"
             << "\tFEATURE TYPE = " << feature_type << endl;
        if (feature_type == "CLASS") {
            cout << "\tCLASSIFICATION = " << detections[i].detection_properties["CLASSIFICATION"] << endl;
        }
        cout << "\tConfidence = " << detections[i].confidence << endl;
    }

    trtis_component.Close();
    return EXIT_SUCCESS;
}
