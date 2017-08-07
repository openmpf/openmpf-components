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

#include "OcvFaceDetection.h"

#include <cstdlib>
#include <stdio.h>
#include <iostream>

#include <QCoreApplication>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>

using namespace std;

using namespace MPF;
using namespace COMPONENT;

///*
//  This file contains the main program to run the OpenCV face detection
// in standalone mode.
// */

int main(int argc, char* argv[])
{
    QCoreApplication *this_app = new QCoreApplication(argc, argv);
    string app_dir = (this_app->applicationDirPath()).toStdString();

    delete this_app;

    bool print_tracks_from_main = false;

    map<string, string> algorithm_properties;

    if(argc == 2)
    {
        char *image_file;

        image_file = argv[1];

        printf ("file = %s\n", image_file);

        /****************************/
        /* Configure and Initialize */
        /****************************/

        OCVFaceDetection *open_detection = new OCVFaceDetection();
        MPFDetectionComponent *detection_engine = open_detection;
        detection_engine->SetRunDirectory(app_dir + "/plugin");

        if ( !detection_engine->Init() ) {
            std::cout << "OpenCV Detector initialization failed, exiting." << std::endl;
            return -1;
        }

        // sdk_profile_timers_init();
        // sdk_profile_timers_name_timer(0, "total");
        // sdk_profile_timers_start_timer(0);

        /**************************/
        /* Read and Submit Images */
        /**************************/

        vector<MPFImageLocation> locations;

        MPFDetectionError rc = MPF_DETECTION_SUCCESS;
        MPFImageJob job("Testing", image_file, algorithm_properties, { });
        rc = detection_engine->GetDetections(job, locations);
        if (rc != MPF_DETECTION_SUCCESS) {
            std::cout << "OpenCV Face detector failed to get detections: rc = " << rc << std::endl;
        }
        std::cout << "Number of detections: " << locations.size() << std::endl;

        // sdk_profile_timers_stop_timer(0);
        // sdk_profile_timers_print_timers();

        if ( !detection_engine->Close() ) {
            std::cout << "OpenCV Face detection component failed in closing." << std::endl;
        }

        delete detection_engine;

        exit(0);
    }
    else if(argc > 3)
    {
        OCVFaceDetection *open_detection = new OCVFaceDetection();

        //transfer the pointer
        MPFDetectionComponent *detection_engine = open_detection;

        detection_engine->SetRunDirectory(app_dir);

        if(!detection_engine->Init())
        {
            printf("\nError: Open Face Detection failed to initialize \n\n");
            return -1;
        }

        //get detection interval if argument is present
        int detection_interval = 1;
        if(argc > 4)
        {
            detection_interval = atoi(argv[4]);
        }
        ostringstream stringStream;
        stringStream << detection_interval;
        algorithm_properties.insert(pair<string, string>("FRAME_INTERVAL", stringStream.str()));
        vector<MPFVideoTrack> tracks;
        MPFVideoJob job("Testing", argv[1], std::stoi(argv[2]), std::stoi(argv[3]), algorithm_properties, { });
        if (detection_engine->GetDetections(job, tracks) != MPFDetectionError::MPF_DETECTION_SUCCESS)
        {
            printf("\nError: Failed to get tracks \n\n");
            return -1;
        }

        if(print_tracks_from_main)
        {
            printf("\nchecking for tracks... \n\n");
            printf("detection interval: %i \n", detection_interval);

            //now print tracks if available
            if(!tracks.empty())
            {
                printf("\n----Tracks---- \n\n");
                for(unsigned int i=0; i<tracks.size(); i++)
                {
                    printf("\ntrack index: %i \n", i);
                    printf("track start index: %i \n", tracks[i].start_frame);
                    printf("track end index: %i \n", tracks[i].stop_frame);

                    //temp
                    printf("faces size for testing: %i \n", static_cast<int>(tracks[i].frame_locations.size()));;

                    for (std::map<int, MPFImageLocation>::const_iterator it = tracks[i].frame_locations.begin(); it != tracks[i].frame_locations.end(); ++it)
                    {
                        printf("frame index, track bounding rect (x,y,w,h), and confidence: %i, %i, %i, %i, %i, %2.3f \n",
                               it->first, it->second.x_left_upper, it->second.y_left_upper, it->second.width, it->second.height, it->second.confidence);
                    }
                }
            }
            else
            {
                printf("\n--No tracks found--\n");
            }
        }

        if ( !detection_engine->Close() ) {
            std::cout << "OpenCV Face detection component failed in closing." << std::endl;
        }

        //pointer was transferred earlier
        delete detection_engine;

        return 0;
    }
    else
    {
        printf("Usage (IMAGE): sample_ocv_face_detector <uri> \n");
        printf("Usage (VIDEO): sample_ocv_face_detector <uri> <start_index> <end_index> <detection_interval (optional)> \n");
        printf("If end index is set to 0, the remaining video will be used. \n");
    }
}
