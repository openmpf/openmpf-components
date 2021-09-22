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

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <log4cxx/propertyconfigurator.h>
#include <MPFDetectionObjects.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>
#include <VideoGeneration.h>
#include <WriteDetectionsToFile.h>
#include <ImageGeneration.h>

#include "Config.h"
#include "Frame.h"
#include "DetectionLocation.h"
#include "Track.h"
#include "YoloNetwork.h"
#include "OcvYoloDetection.h"

using namespace MPF::COMPONENT;

/** ***************************************************************************
*   macros for "pretty" gtest messages
**************************************************************************** */
#define GTEST_BOX "[          ] "
#define GOUT(MSG){                                                            \
  std::cout << GTEST_BOX << MSG << std::endl;                                 \
}

bool init_logging() {
    log4cxx::PropertyConfigurator::configure("data/log4cxx.properties");
    return true;
}
bool logging_initialized = init_logging();


OcvYoloDetection initComponent() {
    OcvYoloDetection component;
    component.SetRunDirectory(".");
    component.Init();
    return component;
}


template <typename T>
T findDetectionWithClass(const std::string& classification,
                         const std::vector<T> &detections) {
    for (const auto& detection : detections) {
        if (classification == detection.detection_properties.at("CLASSIFICATION")) {
            return detection;
        }
    }

    throw std::runtime_error("No detection with class: " + classification);
}


float iou(const cv::Rect& r1, const cv::Rect& r2) {
    int intersectionArea = (r1 & r2).area();
    int unionArea = (r1 | r2).area();
    return static_cast<float>(intersectionArea) / static_cast<float>(unionArea);
}


float iou(const DetectionLocation& l1, const DetectionLocation& l2) {
  return iou(l1.getRect(),l2.getRect());
}


float iou(const MPFImageLocation& l1, const MPFImageLocation& l2){
  return iou(cv::Rect2i(l1.x_left_upper, l1.y_left_upper, l1.width, l1.height),
             cv::Rect2i(l2.x_left_upper, l2.y_left_upper, l2.width, l2.height));
}


Properties getTinyYoloConfig(float confidenceThreshold = 0.5) {
    return {
            { "MODEL_NAME", "tiny yolo" },
            { "NET_INPUT_IMAGE_SIZE", "416"},
            { "CONFIDENCE_THRESHOLD", std::to_string(confidenceThreshold) }
    };
}


Properties getYoloConfig(float confidenceThreshold = 0.5) {
    return {
            { "MODEL_NAME", "yolo" },
            { "NET_INPUT_IMAGE_SIZE", "416"},
            { "CONFIDENCE_THRESHOLD", std::to_string(confidenceThreshold) }
    };
}


bool same(MPFImageLocation& l1, MPFImageLocation& l2,
          float confidenceTolerance, float iouTolerance,
          float& confidenceDiff,
          float& iouValue){

  confidenceDiff = fabs(l1.confidence - l2.confidence);
  iouValue = iou(l1, l2);
  return   (confidenceDiff <= confidenceTolerance)
        && (l1.detection_properties.at("CLASSIFICATION") == l1.detection_properties.at("CLASSIFICATION"))
        && (1.0f - iouValue <= iouTolerance);
}


bool same(MPFImageLocation& l1, MPFImageLocation& l2,
          float confidenceTolerance = 0.01, float iouTolerance = 0.1 ){
  float tmp1,tmp2;
  return same(l1, l2, confidenceTolerance, iouTolerance, tmp1, tmp2);
}


bool same(MPFVideoTrack& t1, MPFVideoTrack& t2,
          float confidenceTolerance, float iouTolerance,
          float& confidenceDiff,
          float& aveIou){

  confidenceDiff = fabs(t1.confidence - t2.confidence);
  if(   t1.detection_properties.at("CLASSIFICATION") != t2.detection_properties.at("CLASSIFICATION")
     || confidenceDiff > confidenceTolerance) {
       return false;
  }

  int start_frame = min(t1.start_frame, t2.start_frame);
  int stop_frame = max(t1.stop_frame, t2.stop_frame);
  aveIou = 0.0;
  int numCommonFrames = 0;
  for(int f = start_frame; f <= stop_frame; ++f) {
    auto l1Itr = t1.frame_locations.find(f);
    auto l2Itr = t2.frame_locations.find(f);
    if(   l1Itr != t1.frame_locations.end()
       && l2Itr != t2.frame_locations.end()) {
         aveIou += iou(l1Itr->second, l2Itr->second);
         ++numCommonFrames;
    }else if (   l1Itr != t1.frame_locations.end()
              || l2Itr != t2.frame_locations.end()) {
      ++numCommonFrames;
    }
  }
  aveIou /= numCommonFrames;
  return (1.0 - aveIou <= iouTolerance);
}


bool same(MPFVideoTrack& t1, MPFVideoTrack& t2, float confidenceTolerance = 0.01, float iouTolerance = 0.1 ){
  float tmp1, tmp2;
  return same(t1, t2, confidenceTolerance, iouTolerance, tmp1, tmp2);
}


void write_track_output(vector<MPFVideoTrack>& tracks, string outTrackFileName, MPFVideoJob& videoJob){
  std::ofstream outTrackFile(outTrackFileName);
  if(outTrackFile.is_open()){
    for(int i = 0; i < tracks.size(); ++i){
      outTrackFile << "#" << i << " ";
      outTrackFile << tracks.at(i) << std::endl;
    }
    outTrackFile.close();
  }else{
    std::cerr << "Could not open '" << outTrackFileName << "'" << std::endl;
    throw exception();
  }
}


vector<MPFVideoTrack> read_track_output(string inTrackFileName){
  std::ifstream inTrackFile(inTrackFileName);
  vector<MPFVideoTrack> tracks;

  if(inTrackFile.is_open()){
    int idx;
    inTrackFile.ignore(1000,'#');
    while(!inTrackFile.eof()){
      inTrackFile >> idx;
      MPFVideoTrack track;
      inTrackFile >> track;
      tracks.insert(tracks.begin() + idx , track);
      inTrackFile.ignore(1000,'#');
    }
  }else{
    std::cerr << "Could not open '" << inTrackFileName << "'" << std::endl;
    throw exception();
  }
  return tracks;
}


void write_track_output_video(string inVideoFileName, vector<MPFVideoTrack>& tracks, string outVideoFileName, MPFVideoJob& videoJob){

   MPFVideoCapture cap(inVideoFileName);
   cv::VideoWriter writer(outVideoFileName,
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                cap.GetFrameRate(), cap.GetFrameSize());
  //Sort tracks into frames
   map<int,vector<MPFVideoTrack*>> frameTracks;
   int trackIdx = 0;
   for(auto& track:tracks){
     track.detection_properties.emplace("idx",std::to_string(trackIdx++));
     for(auto& det:track.frame_locations){
       if(det.first < track.start_frame || det.first > track.stop_frame){
         GOUT("\tdetection index " + to_string(det.first) + " outside of track frame range [" + to_string(track.start_frame) + "," + to_string(track.stop_frame) + "]");
       }
     }
    for(int fr = track.start_frame; fr <= track.stop_frame; fr++){
      frameTracks[fr].push_back(&track);
    }
   }

   //Render tracks
   cv::Mat frame;
   int frameIdx = cap.GetCurrentFramePosition();
   int calFrameIdx = round(cap.GetCurrentTimeInMillis() * cap.GetFrameRate() / 1000.0);
   int numColors = 16;
   std::vector<cv::Scalar> randomPalette;
   for(int i = 0; i < numColors; ++i){
     randomPalette.emplace_back(rand() % 255,rand() % 255,rand() % 255);
   }
   while(cap.Read(frame)){
     if(frameIdx > videoJob.stop_frame) break;
     if(frameIdx >= videoJob.start_frame){
      map<int,vector<MPFVideoTrack*>>::iterator tracksItr = frameTracks.find(frameIdx);
      if(tracksItr != frameTracks.end()){
        for(MPFVideoTrack* trackPtr:tracksItr->second){
          map<int,MPFImageLocation>::iterator detItr = trackPtr->frame_locations.find(frameIdx);
          if(detItr != trackPtr->frame_locations.end()){
            cv::Rect detection_rect(detItr->second.x_left_upper, detItr->second.y_left_upper, detItr->second.width, detItr->second.height);


            cv::rectangle(frame, detection_rect, randomPalette.at(atoi(trackPtr->detection_properties["idx"].c_str()) % numColors), 2);
            std::stringstream ss;
            ss << trackPtr->detection_properties["idx"] << ":" <<  detItr->second.detection_properties["CLASSIFICATION"] << ":" << std::setprecision(3) << detItr->second.confidence;
            cv::putText(frame, ss.str(), detection_rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 200), 1);
          }
        }
      }
      string disp = "# " + to_string(frameIdx) + ":" + to_string(calFrameIdx);
      putText(frame, disp, cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 200, 200), 4);
      writer.write(frame);
     }
     frameIdx = cap.GetCurrentFramePosition();
     calFrameIdx = round(cap.GetCurrentTimeInMillis() * cap.GetFrameRate() / 1000.0);
   }

}


bool objectFound(const std::string &expectedObjectName, const Properties &detectionProperties) {
    return expectedObjectName == detectionProperties.at("CLASSIFICATION");
}


bool objectFound(const std::string &expectedObjectName, int frameNumber,
                 const std::vector<MPFVideoTrack> &tracks) {

    return std::any_of(tracks.begin(), tracks.end(), [&](const MPFVideoTrack &track) {
        return frameNumber >= track.start_frame
               && frameNumber <= track.stop_frame
               && objectFound(expectedObjectName, track.detection_properties)
               && objectFound(expectedObjectName,
                              track.frame_locations.at(frameNumber).detection_properties);

    });
}


bool objectFound(const std::string &expected_object_name,
                 const std::vector<MPFImageLocation> &detections) {
    return std::any_of(detections.begin(), detections.end(), [&](const MPFImageLocation &loc) {
        return objectFound(expected_object_name, loc.detection_properties);
    });
}


/** ***************************************************************************
*   Test phase correlator and similarity score images
**************************************************************************** */
// TODO: Determine if this is worth saving. If it is, then clean it up.
TEST(OcvYoloDetection, TestCorrelator) {
    string image_file = "data/dog.jpg";
    string output_image_file = "correlator.png";

    GOUT("Correlator Output:\t" << output_image_file);
    GOUT("Input Image:\t"       << image_file);
    MPFImageJob job("Testing", image_file, { }, { });
    MPFImageReader imageReader(job);
    Config cfg(job.job_properties);
    Frame frame1(imageReader.GetImage());
    EXPECT_FALSE(frame1.data.empty()) << "Could not load:" << image_file;

    ModelSettings modelSettings;
    modelSettings.networkConfigFile = "OcvYoloDetection/models/yolov4-tiny.cfg";
    modelSettings.weightsFile = "OcvYoloDetection/models/yolov4-tiny.weights";
    modelSettings.namesFile = "OcvYoloDetection/models/coco.names";

    std::vector<Frame> frameBatch = {frame1};
    std::vector<std::vector<DetectionLocation>> detections;
    YoloNetwork(modelSettings, cfg).GetDetections(
      frameBatch,
      [&detections](std::vector<std::vector<DetectionLocation>> detectionsVec,
        std::vector<Frame>::const_iterator,
        std::vector<Frame>::const_iterator){
        detections = detectionsVec;
      },
      cfg);

    EXPECT_FALSE(detections[0].empty());

    std::vector<DetectionLocation>::iterator dogItr=detections[0].begin();
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
    cv::getRectSubPix(dogItr->frame.data, size, center + offset, dog);
    cv::imwrite("correlationPatch.png", dog);

    Frame frame2(cv::Mat::zeros(dogItr->frame.data.size(), dogItr->frame.data.type()));
    cv::Rect2i pasteRoi(cv::Point2i(frame2.data.size() - dog.size()) / 2, dog.size());
    dog.copyTo(frame2.data(pasteRoi));
    cv::imwrite("correlationFrame.png",frame2.data);

    Track t;
    t.add(
      DetectionLocation(cfg,
                        frame2,
                        pasteRoi,
                        0.97,
                        dogItr->getClassFeature(),
                        cv::Mat()));
    GOUT("Shift image " << t.back().getRect() << " centered at " << (t.back().getRect().tl()+t.back().getRect().br())/2.0);


    cv::Point2d ph_offset =  dogItr->phaseCorrelate(t);
    cv::Point2d diff = (offset + ph_offset);
    double dist = sqrt(diff.dot(diff));
    GOUT("phase correlation found offset:" << ph_offset << " at a distance of " << dist << " pixels");
    EXPECT_LE(dist,2.0);

    float feature_dist = dogItr->featureDist(t);
    GOUT("feature distance: "<< feature_dist );
    EXPECT_LE(feature_dist,1E-3);
}


TEST(OcvYoloDetection, TestImage) {
    MPFImageJob job("Test", "data/dog.jpg", getYoloConfig(), {});

    auto detections = initComponent().GetDetections(job);
    ASSERT_EQ(3, detections.size());

    {
        const auto& dogDetection = findDetectionWithClass("dog", detections);
        ASSERT_NEAR(132, dogDetection.x_left_upper,2);
        ASSERT_NEAR(229, dogDetection.y_left_upper,2);
        ASSERT_NEAR(178, dogDetection.width,2);
        ASSERT_NEAR(312, dogDetection.height,2);
        ASSERT_NEAR(0.987, dogDetection.confidence, 0.01);
        ASSERT_EQ("dog", dogDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto& bikeDetection = findDetectionWithClass("bicycle", detections);
        ASSERT_NEAR(124, bikeDetection.x_left_upper,2);
        ASSERT_NEAR(135, bikeDetection.y_left_upper,2);
        ASSERT_NEAR(451, bikeDetection.width,2);
        ASSERT_NEAR(274, bikeDetection.height,2);
        ASSERT_NEAR(0.990, bikeDetection.confidence, 0.01);
        ASSERT_EQ("bicycle", bikeDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto& carDetection = findDetectionWithClass("truck", detections);
        ASSERT_NEAR(462, carDetection.x_left_upper,2);
        ASSERT_NEAR(78, carDetection.y_left_upper,2);
        ASSERT_NEAR(230, carDetection.width,2);
        ASSERT_NEAR(92, carDetection.height,2);
        ASSERT_NEAR(0.910, carDetection.confidence, 0.01);
        ASSERT_EQ("truck", carDetection.detection_properties.at("CLASSIFICATION"));
    }
}


TEST(OcvYoloDetection, TestInvalidModel) {
    ModelSettings modelSettings;
    modelSettings.networkConfigFile = "fake config";
    modelSettings.namesFile = "fake names";
    modelSettings.weightsFile = "fake weights";
    Config config({});

    try {
        YoloNetwork network(modelSettings, config);
        FAIL() << "Expected exception not thrown.";
    }
    catch (const MPFDetectionException &e) {
        ASSERT_EQ(MPF_COULD_NOT_READ_DATAFILE, e.error_code);
    }
}


TEST(OcvYoloDetection, TestWhitelist) {
    Properties jobProps = getTinyYoloConfig();
    auto component = initComponent();

    {
        jobProps["CLASS_WHITELIST_FILE"] = "data/test-whitelist.txt";
        MPFImageJob job("Test", "data/dog.jpg", jobProps, {});

        std::vector<MPFImageLocation> results = component.GetDetections(job);
        ASSERT_TRUE(objectFound("dog", results));
        ASSERT_TRUE(objectFound("bicycle", results));
        ASSERT_FALSE(objectFound("car", results));
    }
    {
        int end_frame = 2;
        setenv("TEST_ENV_VAR", "data", true);
        setenv("TEST_ENV_VAR2", "whitelist", true);
        jobProps["CLASS_WHITELIST_FILE"] = "$TEST_ENV_VAR/test-${TEST_ENV_VAR2}.txt";

        MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, jobProps, {});

        std::vector<MPFVideoTrack> results = component.GetDetections(job);
        for (int i = 0; i <= end_frame; i++) {
            ASSERT_TRUE(objectFound("person", i, results));
            ASSERT_FALSE(objectFound("car", i, results));
        }
    }
}
