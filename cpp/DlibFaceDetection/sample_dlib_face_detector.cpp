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

#include "DlibFaceDetection.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <QCoreApplication>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>

using namespace std;

using namespace MPF;
using namespace COMPONENT;

// Main program to run the Dlib face detection in standalone mode.

int processImage(MPFDetectionComponent *detection_engine, int argc, char* argv[]);
int processVideo(MPFDetectionComponent *detection_engine, int argc, char* argv[]);

int main(int argc, char* argv[]) {

    if (argc < 2 || argc > 5) {
        printf("Usage (IMAGE): %s <uri>\n", argv[0]);
        printf("Usage (VIDEO): %s <uri> <start_index> <end_index> <detection_interval (optional)>\n", argv[0]);
        return -1;
    }

    QCoreApplication *this_app = new QCoreApplication(argc, argv);
    string app_dir = (this_app->applicationDirPath()).toStdString();
    delete this_app;

    DlibFaceDetection *dlib_detection = new DlibFaceDetection();
    MPFDetectionComponent *detection_engine = dlib_detection;
    detection_engine->SetRunDirectory(app_dir + "/plugin");

    if (!detection_engine->Init()) {
        printf("Failed to initialize");
        return -1;
    }

    int rc;
    if (argc == 2) {
        rc = processImage(detection_engine, argc, argv);
    } else {
        rc = processVideo(detection_engine, argc, argv);
    }

    if (!detection_engine->Close()) {
        printf("Failed to close");
    }

    delete detection_engine;

    return rc;
}

int processImage(MPFDetectionComponent *detection_engine, int argc, char* argv[]) {

    MPFImageJob job("Testing", argv[1], { }, { });
    vector<MPFImageLocation> locations;

    MPFDetectionError rc = detection_engine->GetDetections(job, locations);

    if (rc != MPF_DETECTION_SUCCESS) {
        printf("Failed to get detections: rc = %i\n", rc);
    }

    printf("Number of detections: %i\n", locations.size());

    return rc;
}

int processVideo(MPFDetectionComponent *detection_engine, int argc, char* argv[]) {

    // get detection interval if argument is present
    int detection_interval = 1;
    if (argc > 4) {
        detection_interval = stoi(argv[4]);
    }
    printf("Using detection interval: %i\n", detection_interval);

    map<string, string> algorithm_properties;
    algorithm_properties.insert(pair<string, string>("FRAME_INTERVAL", to_string(detection_interval)));

    MPFVideoJob job("Testing", argv[1], stoi(argv[2]), stoi(argv[3]), algorithm_properties, { });
    vector<MPFVideoTrack> tracks;

    MPFDetectionError rc = detection_engine->GetDetections(job, tracks);

    if (rc != MPFDetectionError::MPF_DETECTION_SUCCESS) {
        printf("Failed to get detections: rc = %i\n", rc);
    }

    cout << "Number of video tracks = " << tracks.size() << endl;
    for (int i = 0; i < tracks.size(); i++) {
        cout << "\nVideo track " << i << "\n"
                  << "   start frame = " << tracks[i].start_frame << "\n"
                  << "   stop frame = " << tracks[i].stop_frame << "\n"
                  << "   number of locations = " << tracks[i].frame_locations.size() << "\n"
                  << "   confidence = " << tracks[i].confidence << endl;

        for (auto it : tracks[i].frame_locations) {
            cout << "   Image location frame = " << it.first << "\n"
                      << "      x left upper = " << it.second.x_left_upper << "\n"
                      << "      y left upper = " << it.second.y_left_upper << "\n"
                      << "      width = " << it.second.width << "\n"
                      << "      height = " << it.second.height << "\n"
                      << "      confidence = " << it.second.confidence << endl;
        }
    }

    return rc;
}