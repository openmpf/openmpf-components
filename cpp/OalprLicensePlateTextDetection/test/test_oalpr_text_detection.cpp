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
#include <map>
#include <string>
#include <vector>

#include <QDir>

#include <gtest/gtest.h>

#include <Utils.h>
#include <DetectionComparison.h>
#include <ReadDetectionsFromFile.h>
#include <WriteDetectionsToFile.h>
#include <VideoGeneration.h>
#include <ImageGeneration.h>
#include <MPFSimpleConfigLoader.h>

#include "LicensePlateTextDetection.h"

using std::string;
using std::vector;
using std::map;
using std::pair;
using namespace MPF;
using namespace COMPONENT;


// global variable to hold the file name parameters
QHash<QString, QString> parameters;
bool parameters_loaded = false;

static string GetCurrentWorkingDirectory() {
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

TEST(Detection, Init) {
    string current_working_dir = GetCurrentWorkingDirectory();

    LicensePlateTextDetection *text_detection = new LicensePlateTextDetection();
    ASSERT_TRUE(NULL != text_detection);

    string dir_input(current_working_dir + "/../plugin");
    text_detection->SetRunDirectory(dir_input);
    string rundir = text_detection->GetRunDirectory();
    EXPECT_EQ(dir_input, rundir);

    ASSERT_TRUE(text_detection->Init());

    MPFComponentType comp_type = text_detection->GetComponentType();
    EXPECT_EQ(MPF_DETECTION_COMPONENT, comp_type);

    EXPECT_TRUE(text_detection->Close());
    delete text_detection;
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
      string config_path(current_path.toStdString() + "/config/test_oalpr_text_config.ini");
      std::cout << "config path: " << config_path << std::endl;
      int rc = LoadConfig(config_path, parameters);
      ASSERT_EQ(0, rc);
      std::cout << "Test TestOnKnownVideo: config file loaded" << std::endl;
      parameters_loaded = true;
    }

    std::cout << "Reading parameters for video test." << std::endl;

    start = parameters["OALPR_TEXT_START_FRAME"].toInt();
    stop = parameters["OALPR_TEXT_STOP_FRAME"].toInt();
    rate = parameters["OALPR_TEXT_FRAME_RATE"].toInt();
    inTrackFile = parameters["OALPR_TEXT_KNOWN_TRACKS"].toStdString();
    inVideoFile = parameters["OALPR_TEXT_VIDEO_FILE"].toStdString();
    outTrackFile = parameters["OALPR_TEXT_FOUND_TRACKS"].toStdString();
    outVideoFile = parameters["OALPR_TEXT_VIDEO_OUTPUT_FILE"].toStdString();
    comparison_score_threshold = parameters["OALPR_TEXT_COMPARISON_SCORE_VIDEO"].toFloat();

    // 	Create an OpenALPR text detection object.
    std::cout << "\tCreating OpenALPR text detection" << std::endl;
    LicensePlateTextDetection *text_detection = new LicensePlateTextDetection();
    ASSERT_TRUE(NULL != text_detection);
    text_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(text_detection->Init());

    std::cout << "Start:\t" << start << std::endl;
    std::cout << "Stop:\t" << stop << std::endl;
    std::cout << "Rate:\t" << rate << std::endl;
    std::cout << "inTrack:\t" << inTrackFile << std::endl;
    std::cout << "outTrack:\t" << outTrackFile << std::endl;
    std::cout << "inVideo:\t" << inVideoFile << std::endl;
    std::cout << "outVideo:\t" << outVideoFile << std::endl;
    std::cout << "comparison threshold:\t" << comparison_score_threshold << std::endl;

    // 	Load the known tracks into memory.
    std::cout << "\tLoading the known tracks into memory: " << inTrackFile << std::endl;
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    // 	Evaluate the known video file to generate the test tracks.
    std::cout << "\tRunning the tracker on the video: " << inVideoFile << std::endl;
    const MPFVideoJob video_job("Testing", inVideoFile, start, stop, { }, { });
    vector<MPFVideoTrack> found_tracks = text_detection->GetDetections(video_job);
    EXPECT_FALSE(found_tracks.empty());

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    // don't forget
    std::cout << "\tClosing down detection." << std::endl;
    EXPECT_TRUE(text_detection->Close());
    delete text_detection;
}

TEST(ImageGeneration, TestOnKnownImage) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string test_output_dir = current_working_dir + "/test/test_output/";

    string known_image_file;
    string known_detections_file;
    string output_image_file;
    string output_detections_file;
    float comparison_score_threshold = 0.0;

    if (!parameters_loaded) {
      QString current_path = QDir::currentPath();
      string config_path(current_path.toStdString() + "/config/test_oalpr_text_config.ini");
      std::cout << "config path: " << config_path << std::endl;
      int rc = LoadConfig(config_path, parameters);
      ASSERT_EQ(0, rc);
      std::cout << "Test TestOnKnownImage: config file loaded" << std::endl;
      parameters_loaded = true;
    }

    std::cout << "Reading parameters for image test." << std::endl;

    known_image_file = parameters["OALPR_TEXT_IMAGE_FILE"].toStdString();
    known_detections_file = parameters["OALPR_TEXT_KNOWN_DETECTIONS"].toStdString();
    output_image_file = parameters["OALPR_TEXT_IMAGE_OUTPUT_FILE"].toStdString();
    output_detections_file = parameters["OALPR_TEXT_FOUND_DETECTIONS"].toStdString();

    comparison_score_threshold = parameters["OALPR_TEXT_COMPARISON_SCORE_IMAGE"].toFloat();
    // 	Create a text detection object.
    LicensePlateTextDetection *text_detection = new LicensePlateTextDetection();
    ASSERT_TRUE(NULL != text_detection);

    text_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(text_detection->Init());

    std::cout << "Input Known Detections:\t" << known_detections_file << std::endl;
    std::cout << "Output Found Detections:\t" << output_detections_file << std::endl;
    std::cout << "Input Image:\t" << known_image_file << std::endl;
    std::cout << "Output Image:\t" << output_image_file << std::endl;
    std::cout << "comparison threshold:\t" << comparison_score_threshold << std::endl;

    // 	Load the known detections into memory.
    vector<MPFImageLocation> known_detections;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file,
					       known_detections));

    const MPFImageJob job("Testing", known_image_file, { }, { });
    vector<MPFImageLocation> found_detections = text_detection->GetDetections(job);
    EXPECT_FALSE(found_detections.empty());

    float comparison_score = DetectionComparison::CompareDetectionOutput(found_detections, known_detections);

    std::cout << "Detection comparison score: " << comparison_score << std::endl;

    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                                found_detections,
                                                test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                              found_detections);

    EXPECT_TRUE(text_detection->Close());
    delete text_detection;
}
