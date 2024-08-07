/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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
#include <utility>
#include <vector>

#include <log4cxx/basicconfigurator.h>
#include <gtest/gtest.h>
#include <MPFDetectionComponent.h>
#include <MPFSimpleConfigLoader.h>
#include <Utils.h>
#include <ReadDetectionsFromFile.h>
#include <DetectionComparison.h>
#include <VideoGeneration.h>
#include <WriteDetectionsToFile.h>
#include <ImageGeneration.h>

#include "OcvFaceDetection.h"

using std::pair;
using std::string;
using std::vector;

using namespace MPF;
using namespace COMPONENT;


bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();


TEST(Detection, Init) {
    OcvFaceDetection ocv_face_detection;

    string dir_input("../plugin");
    ocv_face_detection.SetRunDirectory(dir_input);
    string rundir = ocv_face_detection.GetRunDirectory();
    EXPECT_EQ(dir_input, rundir);

    ASSERT_TRUE(ocv_face_detection.Init());

    MPFComponentType comp_type = ocv_face_detection.GetComponentType();
    ASSERT_TRUE(MPF_DETECTION_COMPONENT == comp_type);

    EXPECT_TRUE(ocv_face_detection.Close());
}


// This test checks the confidence of faces detected by the OpenCV low-level detection facility,
// which is used by the OpenCV face detection component.
TEST(OcvFaceDetection, VerifyQuality) {
    string plugins_dir = "../plugin/OcvFaceDetection";

    // 	Create an OCV  detection object.
    std::cout << "\tCreating OCV Detection" << std::endl;
    OcvDetection ocv_detection;
    ASSERT_TRUE(ocv_detection.Init(plugins_dir));

    string test_image_path = "test/test_imgs/S001-01-t10_01.jpg";

    cv::Mat image = cv::imread(test_image_path, cv::IMREAD_IGNORE_ORIENTATION + cv::IMREAD_COLOR);
    ASSERT_TRUE(!image.empty());

    cv::Mat image_gray = Utils::ConvertToGray(image);
    vector<pair<cv::Rect,int>> face_rects = ocv_detection.DetectFaces(image_gray, 10);
    ASSERT_TRUE(face_rects.size() == 1);

    float detection_confidence = static_cast<float>(face_rects[0].second);
    std::cout << "Ocv Detection Confidence Score: " << detection_confidence << std::endl;
    ASSERT_TRUE(detection_confidence > 30);
}


TEST(VideoGeneration, TestOnKnownVideo) {
    int start = 0;
    int stop = 99;
    int rate = 1;
    float comparison_score_threshold = 0.6;
    string inTrackFile = "test/test_vids/ocv_face_known_tracks.txt";
    string inVideoFile = "test/test_vids/new_face_video.avi";
    string outTrackFile = "ocv_face_found_tracks.txt";
    string outVideoFile = "ocv_face_found_tracks.avi";

    // 	Create an OCV face detection object.
    std::cout << "\tCreating OCV Face Detection" << std::endl;
    OcvFaceDetection ocv_face_detection;
    ocv_face_detection.SetRunDirectory("../plugin");
    ASSERT_TRUE(ocv_face_detection.Init());

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
    MPFVideoJob videoJob("Testing", inVideoFile, start, stop, { }, { });
    vector<MPFVideoTrack> found_tracks = ocv_face_detection.GetDetections(videoJob);
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
    EXPECT_TRUE(ocv_face_detection.Close());
}

TEST(ImageGeneration, TestOnKnownImage) {
    string known_image_file = "test/test_imgs/meds_faces_image.png";
    string known_detections_file = "test/test_imgs/ocv_face_known_detections.txt";
    string output_image_file = "ocv_face_found_detections.png";
    string output_detections_file = "ocv_face_found_detections.txt";
    float comparison_score_threshold = 0.2;

    // 	Create an OCV face detection object.
    OcvFaceDetection ocv_face_detection;

    ocv_face_detection.SetRunDirectory("../plugin");
    ASSERT_TRUE(ocv_face_detection.Init());

    std::cout << "Input Known Detections:\t" << known_detections_file << std::endl;
    std::cout << "Output Found Detections:\t" << output_detections_file << std::endl;
    std::cout << "Input Image:\t" << known_image_file << std::endl;
    std::cout << "Output Image:\t" << output_image_file << std::endl;
    std::cout << "comparison threshold:\t" << comparison_score_threshold << std::endl;

    // 	Load the known detections into memory.
    vector<MPFImageLocation> known_detections;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file,
                                               known_detections));

    MPFImageJob image_job("Testing", known_image_file, { }, { });
    vector<MPFImageLocation> found_detections = ocv_face_detection.GetDetections(image_job);
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

    EXPECT_TRUE(ocv_face_detection.Close());
}
