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
#include "OcvSsdFaceDetection.h"
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
  std::cout << GTEST_BOX << MSG << std::endl;                 \
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
  string config_path(current_path.toStdString() + "/config/test_ocv_ssd_face_config.ini");
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
TEST(OcvSsdFaceDetection, OpenCVVersion) {
  #if(CV_MAJOR_VERSION > 3)
    GOUT("OpenCV Version: 4.x");
  #elif(CV_MAJOR_VERSION > 2)
    GOUT("OpenCV Version: 3.x");
  #endif
}

/** ***************************************************************************
*   Test initializing component
**************************************************************************** */
TEST(OcvSsdFaceDetection, Init) {

  string current_working_dir = GetCurrentWorkingDirectory();
  GOUT("current working dir: " << current_working_dir);
  ASSERT_TRUE(current_working_dir != "");


  QHash<QString, QString> parameters = GetTestParameters();
  GOUT("config file:" << parameters["CONFIG_FILE"].toStdString());
  ASSERT_TRUE(parameters.count() > 1);

  OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
  ASSERT_TRUE(NULL != ocv_ssd_face_detection);

  string dir_input(current_working_dir + "/../plugin");
  ocv_ssd_face_detection->SetRunDirectory(dir_input);
  string rundir = ocv_ssd_face_detection->GetRunDirectory();
  EXPECT_EQ(dir_input, rundir);

  ASSERT_TRUE(ocv_ssd_face_detection->Init());

  MPFComponentType comp_type = ocv_ssd_face_detection->GetComponentType();
  ASSERT_TRUE(MPF_DETECTION_COMPONENT == comp_type);

  EXPECT_TRUE(ocv_ssd_face_detection->Close());
  delete ocv_ssd_face_detection;
}

/** ***************************************************************************
*   Test frame preprocessing
**************************************************************************** */
TEST(OcvSsdFaceDetection, DISABLED_PreProcess) {
  string         current_working_dir = GetCurrentWorkingDirectory();
  QHash<QString, QString> parameters = GetTestParameters();

  string test_output_dir = current_working_dir + "/test/test_output/";

  GOUT("Reading parameters for preprocessor video test.");

  int start = parameters["OCV_FACE_START_FRAME"].toInt();
  int  stop = parameters["OCV_FACE_STOP_FRAME" ].toInt();
  int  rate = parameters["OCV_FACE_FRAME_RATE" ].toInt();
  string inVideoFile  = parameters["OCV_PREPROCESS_VIDEO_FILE"       ].toStdString();
  string outVideoFile = parameters["OCV_PREPROCESS_VIDEO_OUTPUT_FILE"].toStdString();

  GOUT("Start:\t"    << start);
  GOUT("Stop:\t"     << stop);
  GOUT("Rate:\t"     << rate);
  GOUT("inVideo:\t"  << inVideoFile);
  GOUT("outVideo:\t" << outVideoFile);

    // 	Create an OCV face detection object.
  GOUT("\tRunning Preprocessor only");
  OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
  ASSERT_TRUE(NULL != ocv_ssd_face_detection);
  ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
  ASSERT_TRUE(ocv_ssd_face_detection->Init());

  MPFVideoJob videoJob("Testing", inVideoFile, start, stop, { }, { });
  JobConfig cfg(videoJob);
  cv::VideoWriter video(outVideoFile,cv::VideoWriter::fourcc('X','2','6','4'),cfg._videocapPtr->GetFrameRate(),cfg._videocapPtr->GetFrameSize());

  while(cfg.nextFrame()) {
    ocv_ssd_face_detection->_normalizeFrame(cfg);
    video.write(cfg.bgrFrame);
  }
  video.release();
  GOUT("\tClosing down detection.");
  EXPECT_TRUE(ocv_ssd_face_detection->Close());
  delete ocv_ssd_face_detection;

}


/** ***************************************************************************
* This test checks the confidence of faces detected by the OpenCV low-level
* detection facility, which is used by the OpenCV face detection component.
**************************************************************************** */
TEST(OcvSsdFaceDetection, VerifyQuality) {

    string current_working_dir = GetCurrentWorkingDirectory();
    string plugins_dir         = current_working_dir + "/../plugin/OcvSsdFaceDetection";

    QHash<QString, QString> parameters = GetTestParameters();
    ASSERT_TRUE(parameters.count() > 1);


    // 	Create an OCV face detection object.
    OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ocv_ssd_face_detection);
    ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_ssd_face_detection->Init());

    //  Load test image
    string test_image_path = parameters["OCV_FACE_1_FILE"].toStdString();
    if(test_image_path.find_first_of('.') == 0) {
        test_image_path = current_working_dir + "/" + test_image_path;
    }
    // Detect detections and check conf levels
    MPFImageJob job1("Testing1", test_image_path, { }, { });
    vector<MPFImageLocation> detections = ocv_ssd_face_detection->GetDetections(job1);
    ASSERT_TRUE(detections.size() == 1);
    GOUT("Detection: " << detections[0]);
    ASSERT_TRUE(detections[0].confidence > .9);
    detections.clear();

    // Detect detections and check conf level
    MPFImageJob job2("Testing2", test_image_path, {{"MIN_DETECTION_SIZE","500"}}, { });
    detections = ocv_ssd_face_detection->GetDetections(job2);
    ASSERT_TRUE(detections.size() == 0);
    detections.clear();

    // Detect detections and check conf levels
    MPFImageJob job3("Testing2", test_image_path, {{"MIN_DETECTION_SIZE","48"},{"CONFIDENCE_THRESHOLD","1.1"}}, { });
    detections = ocv_ssd_face_detection->GetDetections(job3);
    ASSERT_TRUE(detections.size() == 0);
    detections.clear();

    delete ocv_ssd_face_detection;


}

/** ***************************************************************************
*   Test face detection in images
**************************************************************************** */
TEST(OcvSsdFaceDetection, TestOnKnownImage) {

    string current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";
    string known_image_file          = parameters["OCV_FACE_IMAGE_FILE"       ].toStdString();
    string known_detections_file     = parameters["OCV_FACE_KNOWN_DETECTIONS" ].toStdString();
    string output_image_file         = parameters["OCV_FACE_IMAGE_OUTPUT_FILE"].toStdString();
    string output_detections_file    = parameters["OCV_FACE_FOUND_DETECTIONS" ].toStdString();
    float comparison_score_threshold = parameters["OCV_FACE_COMPARISON_SCORE_IMAGE"].toFloat();

    // 	Create an OCV face detection object.
    OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ocv_ssd_face_detection);

    ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_ssd_face_detection->Init());

    GOUT("Input Known Detections:\t"  << known_detections_file);
    GOUT("Output Found Detections:\t" << output_detections_file);
    GOUT("Input Image:\t"             << known_image_file);
    GOUT("Output Image:\t"            << output_image_file);
    GOUT("comparison threshold:\t"    << comparison_score_threshold);

    // 	Load the known detections into memory.
    vector<MPFImageLocation> known_detections;
    ASSERT_TRUE(ReadDetectionsFromFile::ReadImageLocations(known_detections_file, known_detections));


    MPFImageJob image_job("Testing", known_image_file, { }, { });
    vector<MPFImageLocation> found_detections = ocv_ssd_face_detection->GetDetections(image_job);
    EXPECT_FALSE(found_detections.empty());

    float comparison_score = DetectionComparisonA::CompareDetectionOutput(found_detections, known_detections);
    GOUT("Detection comparison score: " << comparison_score);
    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                               found_detections,
                                               test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                            found_detections);

    EXPECT_TRUE(ocv_ssd_face_detection->Close());
    delete ocv_ssd_face_detection;
}


/** ***************************************************************************
*   Test face-recognition with thumbnail images
**************************************************************************** */

TEST(OcvSsdFaceDetection, Thumbnails) {
    string current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";

    // Get test image filenames into vector
    string test_file_dir = parameters["OCV_FACE_THUMBNAIL_TEST_FILE_DIR"].toStdString();
    GOUT("Input Image Dir: " << test_file_dir);
    vector<string> img_file_names;
    size_t idx = 0;
    while(1){
      stringstream ss;
      ss << "OCV_FACE_THUMBNAIL_TEST_FILE_" << setfill('0') << setw(2) << idx;
      QString key = QString::fromStdString(ss.str());
      auto img_file_it = parameters.find(key);
      if(img_file_it == parameters.end()) break;
      img_file_names.push_back(img_file_it.value().toStdString());
      idx++;
    }
    GOUT("Found " << img_file_names.size() << " test images");

    // 	Create an OCV face detection object.
    OcvSsdFaceDetection *ssd = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ssd);

    ssd->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ssd->Init());

    TrackPtrList tracks;
    for(auto img_file_name:img_file_names){
      string img_file = test_file_dir + img_file_name;
      MPFImageLocationVec found_detections;
      MPFImageJob job("Testing", img_file, { }, { });

      JobConfig cfg(job);
      //cfg.bgrFrame = cv::imread(img_file);
      EXPECT_TRUE(NULL != cfg.bgrFrame.data) << "Could not load:" << img_file;

      // find detections
      DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfg);
      EXPECT_FALSE(detections.empty());

      // get landmarks
      for(auto &det:detections){
        EXPECT_FALSE(det->getLandmarks().empty());
      }

      // draw landmarks
      cv::Mat frame = cfg.bgrFrame.clone();
      for(auto &det:detections){
        det->drawLandmarks(frame,cv::Scalar(255,255,255));
      }
      cv::imwrite(test_output_dir + "lm_" + img_file_name,frame);

      // calculate thumbnails and feature vectors
      for(auto &det:detections){
        EXPECT_FALSE(det->getFeature().empty());
      }

      // calculate some feature distances
      GOUT("feature-magnitude1:" << cv::norm(detections.front()->getFeature(),cv::NORM_L2));
      GOUT("feature-magnitude2:" << cv::norm(detections.back()->getFeature(),cv::NORM_L2));
      //GOUT("self feature dist: " << detections.back()->featureDist(tr));
      //ASSERT_TRUE(detections.front()->featureDist(*detections.front()) < 1e-6);
      //GOUT("cross feature dist: " << detections.front()->featureDist(*detections.back()));

      if(tracks.size() == 0){
        for(auto &det:detections){
          tracks.push_back(unique_ptr<Track>(new Track(move(det),cfg)));
        }
      }else{
        //ssd->_assignDetections2Tracks(tracks,detections, ass);
        vector<long> av = ssd->_calcAssignmentVector<&DetectionLocation::iouDist>(tracks,detections,cfg.maxIOUDist);
        ssd->_assignDetections2Tracks(tracks,detections, av);
      }
      float self_fd = tracks.back()->back()->featureDist(*tracks.back());
      GOUT("self feature dist: " << self_fd);
      ASSERT_TRUE(self_fd < 1e-6);
      float cross_fd = tracks.front()->front()->featureDist(*tracks.back());
      GOUT("cross feature dist: " << cross_fd);
      ASSERT_TRUE(cross_fd > 1e-6);
    }

    // write out thumbnail image tracks
    int t=0;
    for(auto &track:tracks){
      for(size_t i=0; i < track->size(); i++){
        EXPECT_FALSE((*track)[i]->getThumbnail().empty());
        stringstream ss;
        ss << test_output_dir << "t" << t << "_i"<< i << ".png";
        string out_file = ss.str();
        GOUT("Writing thumbnail: " << out_file);
        cv::imwrite(out_file,(*track)[i]->getThumbnail());
      }
      t++;
    }

    EXPECT_TRUE(ssd->Close());
    delete ssd;
}

/** ***************************************************************************
*   Test face detection and tracking in videos
**************************************************************************** */
TEST(OcvSsdFaceDetection, TestOnKnownVideo) {

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
    OcvSsdFaceDetection *ocv_ssd_face_detection = new OcvSsdFaceDetection();
    ASSERT_TRUE(NULL != ocv_ssd_face_detection);
    ocv_ssd_face_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_ssd_face_detection->Init());

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
    vector<MPFVideoTrack> found_tracks = ocv_ssd_face_detection->GetDetections(videoJob);
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
    EXPECT_TRUE(ocv_ssd_face_detection->Close());
    delete ocv_ssd_face_detection;
}

/** **************************************************************************
*
**************************************************************************** */
int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
