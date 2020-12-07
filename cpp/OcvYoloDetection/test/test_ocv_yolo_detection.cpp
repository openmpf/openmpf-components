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
#include "DetectionLocation.h"
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
  GOUT("OpenCV Version:" << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION);

  if(CV_MAJOR_VERSION > 4) return;
  if(CV_MAJOR_VERSION == 4 && CV_MINOR_VERSION >=5) return;

  FAIL() << "YoloV4 requires OpenCV Version 4.5 or higher";

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

  map<int,vector<string>> classGroups;
  for(int c=0; c < DetectionLocation::_classGroupIdx.size(); c++){
    classGroups[DetectionLocation::_classGroupIdx[c]].push_back(DetectionLocation::_netClasses[c]);
  }

  for(auto &g : classGroups){
    stringstream ss;
    for(string s : g.second) ss << s << ",";
    GOUT("Group " << g.first << ":" << ss.str());
  }

  EXPECT_TRUE(ocv_yolo_detection->Close());
  delete ocv_yolo_detection;
}

/** ***************************************************************************
*   Test yolo on GPU
**************************************************************************** */
TEST(OcvYoloDetection, DISABLED_YoloInference) {

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

  string image_file = parameters["OCV_IMAGE_FILE" ].toStdString();

  ASSERT_TRUE(ocv_yolo_detection->Init());

  cv::Mat img = cv::imread(image_file);
  ASSERT_FALSE(img.empty());

  cv::Mat blob = cv::dnn::blobFromImage(img,
                                         1/255.0,
                                         cv::Size(416,416),
                                         cv::Scalar(0,0,0),
                                         true,false);
  GOUT("input blob " << blob.size);

  vector<cv::Mat> outs;
  int inference_count = 30;
  GOUT("Default device");
  cv::cuda::DeviceInfo di;
  GOUT("CUDA Device:" << di.name());
  auto start_time = chrono::high_resolution_clock::now();
  for(int i=0;i<inference_count;i++){
    DetectionLocation::_net.setInput(blob,"data");
    DetectionLocation::_net.forward(outs, DetectionLocation::_netOutputNames);
  }
  auto end_time = chrono::high_resolution_clock::now();
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() * 1e-9;
  GOUT("\tCUDA inference time: " << fixed << setprecision(5) << time_taken/inference_count << "[sec]");
  for(int i=0;i<outs.size();i++){
    GOUT("output blob[" << i << "]" << outs[i].size);
  }

  GOUT("CPU only");
  DetectionLocation::loadNetToCudaDevice(-1);
  start_time = chrono::high_resolution_clock::now();
  for(int i=0;i<inference_count;i++){
    DetectionLocation::_net.setInput(blob,"data");
    DetectionLocation::_net.forward(outs, DetectionLocation::_netOutputNames);
  }
  end_time = chrono::high_resolution_clock::now();
  time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() * 1e-9;
  GOUT("\tCPU inference time: " << fixed << setprecision(5) << time_taken/inference_count << "[sec]");
  for(int i=0;i<outs.size();i++){
    GOUT("output blob[" << i << "]" << outs[i].size);
  }


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

    // 	Create an OCV detection object.
    OcvYoloDetection *detection = new OcvYoloDetection();
    ASSERT_TRUE(NULL != detection);

    detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(detection->Init());

    GOUT("Correlator Output:\t" << output_image_file);
    GOUT("Input Image:\t"       << image_file);

    Config cfg(MPFImageJob("Testing", image_file, { }, { }));
    FrameVec framesVec = {cfg.getImageFrame()};
    EXPECT_FALSE(framesVec[0].bgr.empty()) << "Could not load:" << image_file;


    DetectionLocationListVec detections = DetectionLocation::createDetections(cfg, framesVec);
    EXPECT_FALSE(detections[0].empty());

    DetectionLocationList::iterator dogItr=detections[0].begin();
    while(dogItr != detections[0].end()){
      if(dogItr->detection_properties["CLASSIFICATION"]=="dog"){
        GOUT("Found:\t" << dogItr->detection_properties["CLASSIFICATION"]
                        << "  " << dogItr->getRect()
                        << " centered at " << (dogItr->getRect().tl() + dogItr->getRect().br()) / 2.0
                        << " with conf:" << dogItr->confidence);
        break;
      }
      dogItr++;
    }
    EXPECT_TRUE(dogItr != detections[0].end()) << "Could not find dog in image.";
    cv::Point2d offset = cv::Point2d(15.5,22.5);
    cv::Size2d size(dogItr->width*0.95,dogItr->height*0.95);
    cv::Point2d center(cv::Point2d(dogItr->getRect().tl() + dogItr->getRect().br()) / 2.0);
    cv::Mat dog;
    cv::getRectSubPix(dogItr->frame.bgr, size, center + offset, dog);
    cv::imwrite("correlationPatch.png", dog);

    Frame frame;
    frame.bgr = cv::Mat::zeros(dogItr->frame.bgr.size(),dogItr->frame.bgr.type());
    cv::Rect2i pasteRoi(cv::Point2i(frame.bgr.size() - dog.size()) / 2, dog.size());
    dog.copyTo(frame.bgr(pasteRoi));
    cv::imwrite("correlationFrame.png",frame.bgr);

    Track t;
    t.locations.push_back(
      DetectionLocation(cfg,
                        frame,
                        pasteRoi,
                        0.97,
                        dogItr->getClassFeature(),
                        cv::Mat()));
    GOUT("Shift image " << t.locations.back().getRect() << " centered at " << (t.locations.back().getRect().tl()+t.locations.back().getRect().br())/2.0);


    cv::Point2d ph_offset =  dogItr->_phaseCorrelate(t);
    cv::Point2d diff = (offset + ph_offset);
    double dist = sqrt(diff.dot(diff));
    GOUT("phase correlation found offset:" << ph_offset << " at a distance of " << dist << " pixels");
    EXPECT_LE(dist,2.0);

    float feature_dist = dogItr->featureDist(t);
    GOUT("feature distance: "<< feature_dist );
    EXPECT_LE(feature_dist,1E-3);

    EXPECT_TRUE(detection->Close());
    delete detection;
}

/** ***************************************************************************
*   Test detection in images
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

    // 	Create an OCV detection object.
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


    MPFImageJob image_job("Testing", known_image_file, {{"DETECTION_CONFIDENCE_THRESHOLD","0.3"},
                                                        {"DETECTION_NMS_THRESHOLD","0.5"},
                                                        {"DETECTION_IMAGE_SIZE","320"}},
                                                       {});
    vector<MPFImageLocation> found_detections = detection->GetDetections(image_job);
    EXPECT_FALSE(found_detections.empty());

    // create output video to view performance
    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(known_image_file,
                                               found_detections,
                                               test_output_dir + "/" + output_image_file);

    WriteDetectionsToFile::WriteVideoTracks(test_output_dir + "/" + output_detections_file,
                                            found_detections);

    float comparison_score = DetectionComparisonA::CompareDetectionOutput(found_detections, known_detections);
    GOUT("Detection comparison score: " << comparison_score);
    ASSERT_TRUE(comparison_score > comparison_score_threshold);

    EXPECT_TRUE(detection->Close());
    delete detection;
}


/** ***************************************************************************
*   Test detection and tracking in videos
**************************************************************************** */
TEST(OcvYoloDetection, TestOnKnownVideo) {

    string         current_working_dir = GetCurrentWorkingDirectory();
    QHash<QString, QString> parameters = GetTestParameters();

    string test_output_dir = current_working_dir + "/test/test_output/";

    GOUT("Reading parameters for video test.");

    int start = parameters["OCV_VIDEO_START_FRAME"].toInt();
    int  stop = parameters["OCV_VIDEO_STOP_FRAME" ].toInt();
    int  rate = parameters["OCV_VIDEO_FRAME_RATE" ].toInt();
    string inTrackFile  = parameters["OCV_VIDEO_KNOWN_TRACKS"].toStdString();
    string inVideoFile  = parameters["OCV_VIDEO_FILE"].toStdString();
    string outTrackFile = parameters["OCV_VIDEO_FOUND_TRACKS"].toStdString();
    string outVideoFile = parameters["OCV_VIDEO_OUTPUT_FILE"].toStdString();
    float comparison_score_threshold = parameters["OCV_VIDEO_COMPARISON_SCORE_VIDEO"].toFloat();

    GOUT("Start:\t"    << start);
    GOUT("Stop:\t"     << stop);
    GOUT("Rate:\t"     << rate);
    GOUT("inTrack:\t"  << inTrackFile);
    GOUT("outTrack:\t" << outTrackFile);
    GOUT("inVideo:\t"  << inVideoFile);
    GOUT("outVideo:\t" << outVideoFile);
    GOUT("comparison threshold:\t" << comparison_score_threshold);

    GOUT("\tCreating OCV Yolo Detection");
    OcvYoloDetection *ocv_yolo_detection = new OcvYoloDetection();
    ASSERT_TRUE(NULL != ocv_yolo_detection);
    ocv_yolo_detection->SetRunDirectory(current_working_dir + "/../plugin");
    ASSERT_TRUE(ocv_yolo_detection->Init());

    GOUT("\tRunning the tracker on the video: " << inVideoFile);
    MPFVideoJob videoJob("Testing", inVideoFile, start, stop, { }, { });
    auto start_time = chrono::high_resolution_clock::now();
    vector<MPFVideoTrack> found_tracks = ocv_yolo_detection->GetDetections(videoJob);
    auto end_time = chrono::high_resolution_clock::now();
    EXPECT_FALSE(found_tracks.empty());
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    time_taken = time_taken * 1e-9;
    GOUT("\tVideoJob processing time: " << fixed << setprecision(5) << time_taken << "[sec]");

    GOUT("\tWriting detected video and test tracks to files.");
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
    WriteDetectionsToFile::WriteVideoTracks((test_output_dir + "/" + outTrackFile), found_tracks);

    GOUT("\tClosing down detection.");
    EXPECT_TRUE(ocv_yolo_detection->Close());
    delete ocv_yolo_detection;
}


/** ***************************************************************************
*   Test dlib max cost detection
**************************************************************************** */
#include <dlib/optimization.h>
TEST(OcvYoloDetection, DISABLED_MaxCostAssignment) {
  dlib::matrix<long> costs = dlib::zeros_matrix<long>(8,8);
  //dlib::matrix<int> costs = dlib::zeros_matrix<int>(8,8);  // this gives wrong result. Overflow ?!
  costs = 1,2,2080142655,2086630807,3,4,5,6,
          2096932173,2068833426,7,8,2098569871,9,10,11,
          2144440038,2115634073,12,13,2134986491,14,15,16,
          2129194210,2147086229,17,18,2132959446,19,20,21,
          2141311336,2123985951,22,23,2144927865,24,25,26,
          27,28,2087059297,2147231454,29,30,31,32,
          33,34,2147248774,2082771391,35,36,37,38,
          2126672429,2138828231,39,40,2116060790,41,42,43;

  vector<long> av; //assignment vector to return
  av = dlib::max_cost_assignment(costs);
  GOUT( "solved assignment vec["<< av.size() << "] = "  << av);

}


/** **************************************************************************
*
**************************************************************************** */
int main(int argc, char **argv){
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
