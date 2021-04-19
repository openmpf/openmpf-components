/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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

#include <stdio.h>
#include <iostream>

#include <gtest/gtest.h>
#include <log4cxx/basicconfigurator.h>

#include <Utils.h>
#include <DetectionComparison.h>
#include <ReadDetectionsFromFile.h>
#include <WriteDetectionsToFile.h>
#include <VideoGeneration.h>
#include <ImageGeneration.h>
#include <MPFSimpleConfigLoader.h>

#include "DlibFaceDetection.h"

#include <map>
#include <string>
#include <vector>

//  Qt headers
#include <QDir>
#include <QHash>
#include <QString>

using std::string;
using std::vector;
using std::map;

using namespace std;
using namespace MPF;
using namespace COMPONENT;


// global variable to hold the file name parameters
QHash<QString, QString> parameters;
bool parameters_loaded = false;

string GetCurrentWorkingDirectory() {
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        std::cout << "Current working dir: " << cwd << std::endl;
        return string(cwd);
    }
    else {
        std::cout << "getcwd() error";
        return "";
    }
}

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();

TEST(Detection, Init) {

    string current_working_dir = GetCurrentWorkingDirectory();

    DlibFaceDetection *dlib_face_detection = new DlibFaceDetection();
    ASSERT_TRUE(NULL != dlib_face_detection);

    string dir_input(current_working_dir + "/../plugin");
    dlib_face_detection->SetRunDirectory(dir_input);
    string rundir = dlib_face_detection->GetRunDirectory();
    EXPECT_EQ(dir_input, rundir);

    ASSERT_TRUE(dlib_face_detection->Init());

    MPFComponentType comp_type = dlib_face_detection->GetComponentType();
    EXPECT_EQ(MPF_DETECTION_COMPONENT, comp_type);

    EXPECT_TRUE(dlib_face_detection->Close());
    delete dlib_face_detection;
}

TEST(VideoGeneration, TestOnKnownVideo) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string test_output_dir = current_working_dir + "/test/test_output/";

    int start = 0;
    int stop = 0;
    int rate = 0;
    float comparison_score_threshold = 0.0;
    string inTrackFile;
    string inVideoFile;
    string outTrackFile;
    string outVideoFile;
    float threshold;

    if (!parameters_loaded) {
        QString current_path = QDir::currentPath();
        std::string config_path(current_path.toStdString() + "/config/test_dlib_face_config.ini");
        int rc = LoadConfig(config_path, parameters);
        std::cout << "Test TestOnKnownVideo: config file loaded response code: " << rc << std::endl;
    } else {
    	std::cout << "Test TestOnKnownVideo: config file NOT LOADED" << std::endl;
    }

    std::cout << "Reading parameters for video test." << std::endl;

    start = parameters["DLIB_FACE_START_FRAME"].toInt();
    stop = parameters["DLIB_FACE_STOP_FRAME"].toInt();
    rate = parameters["DLIB_FACE_FRAME_RATE"].toInt();
    inTrackFile = parameters["DLIB_FACE_KNOWN_TRACKS"].toStdString();
    inVideoFile = parameters["DLIB_FACE_VIDEO_FILE"].toStdString();
    outTrackFile = parameters["DLIB_FACE_FOUND_TRACKS"].toStdString();
    outVideoFile = parameters["DLIB_FACE_VIDEO_OUTPUT_FILE"].toStdString();
    comparison_score_threshold = parameters["DLIB_FACE_COMPARISON_SCORE_VIDEO"].toFloat();

    // 	Create a dlib face detection object.
    DlibFaceDetection *dlib_face_detection = new DlibFaceDetection();
    dlib_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    EXPECT_TRUE(dlib_face_detection->Init());

    cout << "Start:\t" << start << endl;
    cout << "Stop:\t" << stop << endl;
    cout << "Rate:\t" << rate << endl;
    cout << "inTrack:\t" << inTrackFile << endl;
    cout << "outTrack:\t" << outTrackFile << endl;
    cout << "inVideo:\t" << inVideoFile << endl;
    cout << "outVideo:\t" << outVideoFile << endl;
    cout << "comparison threshold:\t" << comparison_score_threshold << endl;

    // 	Load the known tracks into memory.
    cout << "\tLoading the known tracks into memory: " << inTrackFile << endl;
    std::vector<MPFVideoTrack> known_tracks;
    EXPECT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    // 	Evaluate the known video file to generate the test tracks.
    cout << "\tRunning the tracker on the video: " << inVideoFile << endl;
    MPFVideoJob job("Testing", inVideoFile, start, stop, { }, { });
    std::vector<MPFVideoTrack> found_tracks = dlib_face_detection->GetDetections(job);
    EXPECT_FALSE(found_tracks.empty());

    // 	Compare the known and test track output.
    cout << "\tComparing the known and test tracks." << endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    cout << "Tracker comparison score: " << comparison_score << endl;
    EXPECT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    cout << "\tWriting detected video and test tracks to files." << endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    // don't forget
    cout << "\tClosing down detection." << endl;
    EXPECT_TRUE(dlib_face_detection->Close());
    delete dlib_face_detection;
}

TEST(ImageGeneration, TestOnKnownImage) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string test_output_dir = current_working_dir + "/test/test_output/";

    std::string known_image_file;
    std::string known_detections_file;
    std::string output_image_file;
    std::string output_detections_file;
    float comparison_score_threshold = 0.2;

    if (!parameters_loaded) {
        QString current_path = QDir::currentPath();
        std::string config_path(current_path.toStdString() + "/config/test_dlib_face_config.ini");
        int rc = LoadConfig(config_path, parameters);
    	std::cout << "Test TestOnKnownVideo: config file loaded response code: " << rc << std::endl;
    } else {
    	std::cout << "Test TestOnKnownVideo: config file NOT LOADED" << std::endl;
    }

    cout << "Setting read parameters for DLIB_FACE_DETECTION." << endl;

    known_image_file = parameters["DLIB_FACE_IMAGE_FILE"].toStdString();
    known_detections_file = parameters["DLIB_FACE_KNOWN_DETECTIONS"].toStdString();
    output_image_file = parameters["DLIB_FACE_IMAGE_OUTPUT_FILE"].toStdString();
    output_detections_file = parameters["DLIB_FACE_FOUND_DETECTIONS"].toStdString();

    comparison_score_threshold = parameters["DLIB_FACE_COMPARISON_SCORE_IMAGE"].toFloat();

    // 	Create a dlib face detection object.
    DlibFaceDetection *dlib_face_detection = new DlibFaceDetection();
    dlib_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    EXPECT_TRUE(dlib_face_detection->Init());

    cout << "Input Known Detections:\t" << known_detections_file << endl;
    cout << "Output Found Detections:\t" << output_detections_file << endl;
    cout << "Input Image:\t" << known_image_file << endl;
    cout << "Output Image:\t" << output_image_file << endl;
    cout << "comparison threshold:\t" << comparison_score_threshold << endl;

    // 	Load the known detections into memory.
    std::vector<MPFImageLocation> known_detections;
    EXPECT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file, known_detections));

    MPFImageJob job("Testing", known_image_file, { }, { });
    std::vector<MPFImageLocation> found_detections = dlib_face_detection->GetDetections(job);
    EXPECT_FALSE(found_detections.empty());

    float comparison_score = DetectionComparison::CompareDetectionOutput(found_detections, known_detections);

    cout << "Detection comparison score: " << comparison_score << endl;

    EXPECT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                                found_detections,
                                                test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                              found_detections);

    EXPECT_TRUE(dlib_face_detection->Close());
    delete dlib_face_detection;
}
