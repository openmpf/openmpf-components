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

#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <log4cxx/basicconfigurator.h>

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

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();


TEST(Detection, Init) {
    LicensePlateTextDetection text_detection;
    string dir_input("../plugin");
    text_detection.SetRunDirectory(dir_input);
    string rundir = text_detection.GetRunDirectory();
    EXPECT_EQ(dir_input, rundir);

    ASSERT_TRUE(text_detection.Init());

    MPFComponentType comp_type = text_detection.GetComponentType();
    EXPECT_EQ(MPF_DETECTION_COMPONENT, comp_type);

    EXPECT_TRUE(text_detection.Close());
}

TEST(VideoGeneration, TestOnKnownVideo) {
    int start = 0;
    int stop = 49;
    int rate = 1;
    float comparison_score_threshold = 0.3;
    string inTrackFile = "test/test_vids/oalpr_text_known_tracks.txt";
    string inVideoFile = "test/test_vids/oalpr_text_video.avi";
    string outTrackFile = "alpr_text_found_tracks.txt";
    string outVideoFile = "oalpr_text_found_tracks.avi";

    // 	Create an OpenALPR text detection object.
    std::cout << "\tCreating OpenALPR text detection" << std::endl;
    LicensePlateTextDetection text_detection;
    text_detection.SetRunDirectory("../plugin");
    ASSERT_TRUE(text_detection.Init());

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
    vector<MPFVideoTrack> found_tracks = text_detection.GetDetections(video_job);
    EXPECT_FALSE(found_tracks.empty());

    // 	Compare the known and test track output.
    std::cout << "\tComparing the known and test tracks." << std::endl;
    float comparison_score = DetectionComparison::CompareDetectionOutput(found_tracks, known_tracks);
    std::cout << "Tracker comparison score: " << comparison_score << std::endl;
    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    std::cout << "\tWriting detected video and test tracks to files." << std::endl;
    VideoGeneration video_generation;
    string test_output_dir = "test/test_output/";
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    // don't forget
    std::cout << "\tClosing down detection." << std::endl;
    EXPECT_TRUE(text_detection.Close());
}

TEST(ImageGeneration, TestOnKnownImage) {

    string known_image_file = "test/test_imgs/oalpr_text_image.png";
    string known_detections_file = "test/test_imgs/oalpr_text_known_detections.txt";
    string output_image_file = "oalpr_text_found_detections.png";
    string output_detections_file = "oalpr_text_found_detections.txt";
    float comparison_score_threshold = 0.6;


    // 	Create a text detection object.
    LicensePlateTextDetection text_detection;

    text_detection.SetRunDirectory("../plugin");
    ASSERT_TRUE(text_detection.Init());

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
    vector<MPFImageLocation> found_detections = text_detection.GetDetections(job);
    EXPECT_FALSE(found_detections.empty());

    float comparison_score = DetectionComparison::CompareDetectionOutput(found_detections, known_detections);

    std::cout << "Detection comparison score: " << comparison_score << std::endl;

    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    string test_output_dir = "test/test_output/";
    image_generation.WriteDetectionOutputImage(known_image_file,
                                                found_detections,
                                                test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                              found_detections);

    EXPECT_TRUE(text_detection.Close());
}
