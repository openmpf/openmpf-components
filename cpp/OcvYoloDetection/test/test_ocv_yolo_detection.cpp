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


#include <string>
#include <utility>
#include <vector>
#include <chrono>

#include <gtest/gtest.h>
#include <MPFDetectionComponent.h>
#include <QDir>
#include <MPFSimpleConfigLoader.h>
#include <Utils.h>
#include <ReadDetectionsFromFile.h>
//#include <DetectionComparison.h>
#include <VideoGeneration.h>
#include <WriteDetectionsToFile.h>
#include <ImageGeneration.h>

#define private public
#include "OcvYoloDetection.h"
#include "DetectionComparisonA.h"
using namespace std;
using namespace MPF::COMPONENT;

/** ***************************************************************************
*   macros for "pretty" gtest messages
**************************************************************************** */
#define ANSI_TXT_GRN "\033[0;32m"
#define ANSI_TXT_MGT "\033[0;35m" //Magenta
#define ANSI_TXT_DFT "\033[0;0m" //Console default
#define GTEST_BOX "[          ] "
#define GOUT(MSG){                                                            \
  std::cout << GTEST_BOX << MSG << std::endl;                                 \
}
#define GOUT_MGT(MSG){                                                        \
  std::cout << ANSI_TXT_MGT << GTEST_BOX << MSG << ANSI_TXT_DFT << std::endl; \
}
#define GOUT_GRN(MSG){                                                        \
  std::cout << ANSI_TXT_GRN << GTEST_BOX << MSG << ANSI_TXT_DFT << std::endl; \
}

/** ***************************************************************************
*   global variable to hold the file name parameters
**************************************************************************** */

 QHash<QString, QString> GetTestParameters(){

  QString current_path = QDir::currentPath();
  QHash<QString, QString> parameters;
  string config_path(current_path.toStdString() + "/config/test_ocv_yolo_config.ini");
  int rc = LoadConfig(config_path, parameters);
  if(rc == 0){
    parameters["CONFIG_FILE"] = QString::fromStdString(config_path);
  }else{
    parameters.clear();
    GOUT("config file failed to load with error:" << rc << " for '" << config_path << "'");
  }
  return parameters;
}
/** ***************************************************************************
*   get current working directory with minimal error checking
**************************************************************************** */

static string GetCurrentWorkingDirectory() {
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
      return string(cwd);
  }else{
      return "";
  }
}

/** ***************************************************************************
*   Test component opencv version
**************************************************************************** */
TEST(OcvYoloDetection, OpenCVVersion) {
  #if(CV_MAJOR_VERSION > 3)
    GOUT("OpenCV Version: 4.x");
  #elif(CV_MAJOR_VERSION > 2)
    GOUT("OpenCV Version: 3.x");
  #endif
}

/** ***************************************************************************
*   Test initializing component
**************************************************************************** */
TEST(OcvYoloDetection, Init) {

  string current_working_dir = GetCurrentWorkingDirectory();
  GOUT("current working dir: " << current_working_dir);
  ASSERT_TRUE(current_working_dir != "");

  QHash<QString, QString> parameters = GetTestParameters();
  GOUT("config file:" << parameters["CONFIG_FILE"].toStdString());
  ASSERT_TRUE(parameters.count() > 1);

  OcvYoloDetection *ocv_yolo_detection = new OcvYoloDetection();
  ASSERT_TRUE(NULL != ocv_yolo_detection);

  string dir_input(current_working_dir + "/../plugin");
  ocv_yolo_detection->SetRunDirectory(dir_input);
  string rundir = ocv_yolo_detection->GetRunDirectory();
  EXPECT_EQ(dir_input, rundir);

  ASSERT_TRUE(ocv_yolo_detection->Init());

  MPFComponentType comp_type = ocv_yolo_detection->GetComponentType();
  ASSERT_TRUE(MPF_DETECTION_COMPONENT == comp_type);

  EXPECT_TRUE(ocv_yolo_detection->Close());
  delete ocv_yolo_detection;
}

/** ***************************************************************************
*   Test phase correlator and similarity score images
**************************************************************************** */
TEST(OcvYoloDetection, TestCorrelator) {
    string current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";
    string image_file                = parameters["OCV_CORRELATOR_IMAGE_FILE" ].toStdString();
    string output_image_file         = parameters["OCV_CORRELATOR_OUTPUT_FILE"].toStdString();

    // 	Create an OCV face detection object.
    OcvYoloDetection *detection = new OcvYoloDetection();
    ASSERT_TRUE(NULL != detection);

    detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(detection->Init());

    GOUT("Correlator Output:\t" << output_image_file);
    GOUT("Input Image:\t"       << image_file);

    auto cfgPtr = make_shared<Config>(MPFImageJob("Testing", image_file, { }, { }));
    EXPECT_TRUE(NULL != cfgPtr->bgrFrame.data) << "Could not load:" << image_file;

    //if(cfg.dftHannWindow){ // since we used image job create hanning window manually
    // cv::createHanningWindow(cfg.defaultHanningWindow,cv::Size(cfg.dftSize,cfg.dftSize),CV_32F);
   // }

    DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfgPtr);
    EXPECT_FALSE(detections.empty());

    DetectionLocationPtr dogPtr;
    cv::Point2d center;
    for(auto& detPtr:detections){
      if(detPtr->detection_properties["CLASSIFICATION"]=="dog"){
        dogPtr = move(detPtr);
        center = cv::Point2d(dogPtr->getRect().tl() + dogPtr->getRect().br()) / 2;
        GOUT("Found:\t" << dogPtr->detection_properties["CLASSIFICATION"]
                        << "  " << dogPtr->getRect()
                        << " centered at " << center
                        << " with conf:" << dogPtr->confidence);
      }
    }
    EXPECT_TRUE(dogPtr != NULL) << "Could not find dog in image.";
    cv::Point2d offset = cv::Point2d(15.5,22.5);
    cv::Size2d size(dogPtr->width*0.95,dogPtr->height*0.95);
    cv::Mat dog;
    cv::getRectSubPix(dogPtr->_bgrFrame, size, center+offset, dog);
    cv::imwrite("correlationPatch.png", dog);

    cv::Mat frame = cv::Mat::zeros(dogPtr->_bgrFrame.size(),dogPtr->_bgrFrame.type());
    cv::Rect2i pasteRoi(cv::Point2i(frame.size() - dog.size())/2, dog.size());
    dog.copyTo(frame(pasteRoi));
    cv::imwrite("correlationFrame.png",frame);

    DetectionLocationPtr dogPtr2;
    dogPtr2 = DetectionLocationPtr(new DetectionLocation(
      cfgPtr,pasteRoi,0.97,cv::Point2f(0.5,0.5)));
    dogPtr2->_bgrFrame = frame;
    GOUT("Shift image " << dogPtr2->getRect() << " centered at " << (dogPtr2->getRect().tl()+dogPtr2->getRect().br())/2);

    Track t(cfgPtr, move(dogPtr2));

    cv::Point2d ph_offset =  dogPtr->_phaseCorrelate(t);
    cv::Point2d diff = (offset + ph_offset);
    double dist = sqrt(diff.dot(diff));
    GOUT("phase correlation found offset:" << ph_offset << " at a distance of " << dist << " pixels");
    EXPECT_LE(dist,2.0);

    float feature_dist = dogPtr->featureDist(t);
    GOUT("feature distance: "<< feature_dist );
    EXPECT_LE(feature_dist,1E-3);

    EXPECT_TRUE(detection->Close());
    delete detection;
}

/** ***************************************************************************
*   Test face detection in images
**************************************************************************** */
TEST(OcvYoloDetection, TestOnKnownImage) {

    string current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";
    string known_image_file          = parameters["OCV_IMAGE_FILE"       ].toStdString();
    string known_detections_file     = parameters["OCV_IMAGE_KNOWN_DETECTIONS" ].toStdString();
    string output_image_file         = parameters["OCV_IMAGE_OUTPUT_FILE"].toStdString();
    string output_detections_file    = parameters["OCV_IMAGE_FOUND_DETECTIONS" ].toStdString();
    float comparison_score_threshold = parameters["OCV_IMAGE_COMPARISON_SCORE"].toFloat();

    // 	Create an OCV face detection object.
    OcvYoloDetection *detection = new OcvYoloDetection();
    ASSERT_TRUE(NULL != detection);

    detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(detection->Init());

    GOUT("Input Known Detections:\t"  << known_detections_file);
    GOUT("Output Found Detections:\t" << output_detections_file);
    GOUT("Input Image:\t"             << known_image_file);
    GOUT("Output Image:\t"            << output_image_file);
    GOUT("comparison threshold:\t"    << comparison_score_threshold);

    // 	Load the known detections into memory.
    vector<MPFImageLocation> known_detections;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file, known_detections));


    MPFImageJob image_job("Testing", known_image_file, { }, { });
    vector<MPFImageLocation> found_detections = detection->GetDetections(image_job);
    EXPECT_FALSE(found_detections.empty());

    //float comparison_score = DetectionComparisonA::CompareDetectionOutput(found_detections, known_detections);
    //GOUT("Detection comparison score: " << comparison_score);
    //ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                               found_detections,
                                               test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                            found_detections);

    EXPECT_TRUE(detection->Close());
    delete detection;
}


/** ***************************************************************************
*   Test face detection and tracking in videos
**************************************************************************** */
TEST(OcvYoloDetection, DISABLED_TestOnKnownVideo) {

    string         current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";

    GOUT("Reading parameters for video test.");

    int start = parameters["OCV_FACE_START_FRAME"].toInt();
    int  stop = parameters["OCV_FACE_STOP_FRAME" ].toInt();
    int  rate = parameters["OCV_FACE_FRAME_RATE" ].toInt();
    string inTrackFile  = parameters["OCV_FACE_KNOWN_TRACKS"     ].toStdString();
    string inVideoFile  = parameters["OCV_FACE_VIDEO_FILE"       ].toStdString();
    string outTrackFile = parameters["OCV_FACE_FOUND_TRACKS"     ].toStdString();
    string outVideoFile = parameters["OCV_FACE_VIDEO_OUTPUT_FILE"].toStdString();
    float comparison_score_threshold = parameters["OCV_FACE_COMPARISON_SCORE_VIDEO"].toFloat();

    GOUT("Start:\t"    << start);
    GOUT("Stop:\t"     << stop);
    GOUT("Rate:\t"     << rate);
    GOUT("inTrack:\t"  << inTrackFile);
    GOUT("outTrack:\t" << outTrackFile);
    GOUT("inVideo:\t"  << inVideoFile);
    GOUT("outVideo:\t" << outVideoFile);
    GOUT("comparison threshold:\t" << comparison_score_threshold);

    // 	Create an OCV face detection object.
    GOUT("\tCreating OCV Face Detection");
    OcvYoloDetection *ocv_yolo_detection = new OcvYoloDetection();
    ASSERT_TRUE(NULL != ocv_yolo_detection);
    ocv_yolo_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_yolo_detection->Init());

    // 	Load the known tracks into memory.
    GOUT("\tLoading the known tracks into memory: " << inTrackFile);
    vector<MPFVideoTrack> known_tracks;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadVideoTracks(inTrackFile, known_tracks));

    // create output known video to view ground truth
    GOUT("\tWriting ground truth video and test tracks to files.");
    VideoGeneration video_generation_gt;
    video_generation_gt.WriteTrackOutputVideo(inVideoFile, known_tracks, (test_output_dir + "/ground_truth.avi"));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/ground_truth.txt"), known_tracks);

    // 	Evaluate the known video file to generate the test tracks.
    GOUT("\tRunning the tracker on the video: " << inVideoFile);
    MPFVideoJob videoJob("Testing", inVideoFile, start, stop, { }, { });
    auto start_time = chrono::high_resolution_clock::now();
    vector<MPFVideoTrack> found_tracks = ocv_yolo_detection->GetDetections(videoJob);
    auto end_time = chrono::high_resolution_clock::now();
    EXPECT_FALSE(found_tracks.empty());
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    time_taken = time_taken * 1e-9;
    GOUT("\tVideoJob processing time: " << fixed << setprecision(5) << time_taken << "[sec]");

    // create output video to view performance
    GOUT("\tWriting detected video and test tracks to files.");
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    // 	Compare the known and test track output.
    GOUT("\tComparing the known and test tracks.");
    float comparison_score = DetectionComparisonA::CompareDetectionOutput(found_tracks, known_tracks);
    GOUT("Tracker comparison score: " << comparison_score);
    ASSERT_TRUE(comparison_score > comparison_score_threshold);



    // don't forget
    GOUT("\tClosing down detection.");
    EXPECT_TRUE(ocv_yolo_detection->Close());
    delete ocv_yolo_detection;
}

/** **************************************************************************
*
**************************************************************************** */
int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
