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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <log4cxx/propertyconfigurator.h>
#include <grpc_client.h>
#include <MPFDetectionObjects.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>

#include <VideoGeneration.h>
#include <WriteDetectionsToFile.h>
#include <ImageGeneration.h>

#include "Config.h"
#include "TritonTensorMeta.h"
#include "TritonClient.h"
#include "TritonInferencer.h"
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
    std::vector<std::vector<DetectionLocation>> detections
            = YoloNetwork(modelSettings, cfg).GetDetections({frame1}, cfg);

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


Properties getTinyYoloConfig(float confidenceThreshold = 0.5) {
    return {
            { "MODEL_NAME", "tiny yolo" },
            {"NET_INPUT_IMAGE_SIZE", "416"},
            { "CONFIDENCE_THRESHOLD", std::to_string(confidenceThreshold) }
    };
}


TEST(OcvYoloDetection, TestImage) {
    MPFImageJob job("Test", "data/dog.jpg", getTinyYoloConfig(), {});

    auto detections = initComponent().GetDetections(job);
    ASSERT_EQ(3, detections.size());

    {
        const auto& dogDetection = findDetectionWithClass("dog", detections);
        ASSERT_EQ(127, dogDetection.x_left_upper);
        ASSERT_EQ(210, dogDetection.y_left_upper);
        ASSERT_EQ(201, dogDetection.width);
        ASSERT_EQ(319, dogDetection.height);
        ASSERT_NEAR(0.727862, dogDetection.confidence, 0.001);
        ASSERT_EQ("dog", dogDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto& bikeDetection = findDetectionWithClass("bicycle", detections);
        ASSERT_EQ(185, bikeDetection.x_left_upper);
        ASSERT_EQ(134, bikeDetection.y_left_upper);
        ASSERT_EQ(392, bikeDetection.width);
        ASSERT_EQ(296, bikeDetection.height);
        ASSERT_NEAR(0.74281, bikeDetection.confidence, 0.001);
        ASSERT_EQ("bicycle", bikeDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto& carDetection = findDetectionWithClass("car", detections);
        ASSERT_EQ(467, carDetection.x_left_upper);
        ASSERT_EQ(78, carDetection.y_left_upper);
        ASSERT_EQ(227, carDetection.width);
        ASSERT_EQ(89, carDetection.height);
        ASSERT_NEAR(0.656565, carDetection.confidence, 0.001);
        ASSERT_EQ("car", carDetection.detection_properties.at("CLASSIFICATION"));
    }
}


void show_track(const std::string& videoPath, const MPFVideoTrack &track) {
    MPFVideoCapture cap(videoPath);
    std::cout << "Track class: " << track.detection_properties.at("CLASSIFICATION") << std::endl;

    for (const auto& pair : track.frame_locations) {
        std::cout << "Frame: " << pair.first << "   Class: " << pair.second.detection_properties.at("CLASSIFICATION") << std::endl;
        cap.SetFramePosition(pair.first);
        cv::Mat frame;
        cap.Read(frame);
        cv::Rect detectionRect(pair.second.x_left_upper, pair.second.y_left_upper,
                               pair.second.width, pair.second.height);

        cv::rectangle(frame, detectionRect, {255, 0, 0});

        cv::imshow("test", frame);
        cv::waitKey();
    }
}


TEST(OcvYoloDetection, TestVideo) {
    auto jobProps = getTinyYoloConfig(0.92);
    jobProps.emplace("TRACKING_DISABLE_MOSSE_TRACKER", "true");
    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 2, 10,
                    jobProps, {});

    auto tracks = initComponent().GetDetections(job);
    ASSERT_EQ(3, tracks.size());
    {
        auto personTrack = findDetectionWithClass("person", tracks);
        ASSERT_EQ(2, personTrack.start_frame);
        ASSERT_EQ(5, personTrack.stop_frame);
        ASSERT_EQ(2, personTrack.frame_locations.size());
        ASSERT_NEAR(0.927688, personTrack.confidence, 0.001);

        const auto& personDetection = personTrack.frame_locations.at(2);
        ASSERT_EQ(532, personDetection.x_left_upper);
        ASSERT_EQ(0, personDetection.y_left_upper);
        ASSERT_EQ(70, personDetection.width);
        ASSERT_EQ(147, personDetection.height);
        ASSERT_EQ(personTrack.confidence, personDetection.confidence);
    }

    int carTrack1Idx = -1;
    int carTrack2Idx = -1;
    for (int i = 0; i < tracks.size(); ++i) {
        const auto& track = tracks[i];
        if (track.detection_properties.at("CLASSIFICATION") != "car") {
            continue;
        }
        if (track.frame_locations.at(2).x_left_upper == 223) {
            carTrack1Idx = i;
        }
        else {
            carTrack2Idx = i;
        }
    }
    ASSERT_TRUE(carTrack1Idx >= 0);
    ASSERT_TRUE(carTrack2Idx >= 0);

    {
        const auto& carTrack = tracks.at(carTrack1Idx);
        ASSERT_EQ(2, carTrack.start_frame);
        ASSERT_EQ(10, carTrack.stop_frame);
        ASSERT_EQ(9, carTrack.frame_locations.size());
        ASSERT_NEAR(0.961101, carTrack.confidence, 0.001);

        const auto& carDetection = carTrack.frame_locations.at(2);
        ASSERT_EQ(223, carDetection.x_left_upper);
        ASSERT_EQ(20, carDetection.y_left_upper);
        ASSERT_EQ(318, carDetection.width);
        ASSERT_EQ(86, carDetection.height);
        ASSERT_NEAR(0.952526, carDetection.confidence, 0.001);
    }

    {
        const auto& carTrack = tracks.at(carTrack2Idx);
        ASSERT_EQ(2, carTrack.start_frame);
        ASSERT_EQ(10, carTrack.stop_frame);
        ASSERT_EQ(7, carTrack.frame_locations.size());
        ASSERT_NEAR(0.9496, carTrack.confidence, 0.001);

        const auto& carDetection = carTrack.frame_locations.at(3);
        ASSERT_EQ(591, carDetection.x_left_upper);
        ASSERT_EQ(37, carDetection.y_left_upper);
        ASSERT_EQ(434, carDetection.width);
        ASSERT_EQ(131, carDetection.height);
        ASSERT_EQ(carTrack.confidence, carDetection.confidence);
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

////////////////////////////////////////////////////////////////////////////////
std::vector<MPFImageLocation> mpfImageLocations(const std::vector<DetectionLocation> dets){
  std::vector<MPFImageLocation> locs;
  for(auto& det : dets){
    locs.emplace_back(
          det.x_left_upper,
          det.y_left_upper,
          det.width,
          det.height,
          det.confidence,
          det.detection_properties);
  }
  return locs;
}

////////////////////////////////////////////////////////////////////////////////
float iou(const DetectionLocation& l1, const DetectionLocation& l2) {
    int intersectionArea = (l1.getRect() & l2.getRect()).area();
    int unionArea = (l1.getRect() | l2.getRect()).area();
    return static_cast<float>(intersectionArea) / static_cast<float>(unionArea);
}

////////////////////////////////////////////////////////////////////////////////
void maxIOU(const DetectionLocation& l, const std::vector<DetectionLocation>& candidates,DetectionLocation& best, float& max_iou){
  max_iou = 0;
  for(auto& c : candidates){
    float s = iou(l, c);
    if(s >= max_iou){
      max_iou = s;
      best = c;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
bool comp_xy(const DetectionLocation& l1, const DetectionLocation& l2){
  if(l1.y_left_upper < l2.y_left_upper){
    return true;
  }else if(l1.y_left_upper == l2.y_left_upper){
    return l1.x_left_upper < l2.x_left_upper;
  }else{
    return false;
  }
}

////////////////////////////////////////////////////////////////////////////////
TEST(OcvYoloDetection, TestTritonClient) {
  auto component = initComponent();

  MPFImageJob job("Testing", "", {{"TRTIS_SERVER","triton:8001"},
                                          {"MODEL_NAME", "yolo"},
                                          {"MAX_INFER_CONCURRENCY", "2"},
                                          {"DETECTION_FRAME_BATCH_SIZE", "3"},
                                          {"NET_INPUT_IMAGE_SIZE", "416"},
                                          {"TRTIS_USE_SHM","false"}
                                         }, { });
  ImageGeneration iGen;

  Config cfg(job.job_properties);
  GOUT("Configuration:" << cfg);
  int samples = 1;

  std::vector<std::string> testFiles = {"data/dog.jpg","data/dog.jpg","data/bird.jpg","data/dog.jpg",
                                        "data/dog.jpg","data/dog.jpg","data/bird.jpg","data/dog.jpg",
                                        "data/dog.jpg","data/dog.jpg","data/bird.jpg","data/dog.jpg",
                                        "data/dog.jpg","data/dog.jpg","data/bird.jpg","data/dog.jpg"};
/*
  std::vector<std::string> testFiles = {"data/dog.jpg","data/dog.jpg","data/dog.jpg","data/dog.jpg",
                                        "data/dog.jpg","data/dog.jpg","data/dog.jpg","data/dog.jpg",
                                        "data/dog.jpg","data/dog.jpg","data/dog.jpg","data/dog.jpg",
                                        "data/dog.jpg","data/dog.jpg","data/dog.jpg","data/dog.jpg"};
*/
  std:vector<Frame> frames;
  for(auto& file : testFiles){
    frames.emplace_back(cv::imread(file));
    EXPECT_FALSE(frames.back().data.empty()) << "Could not load '" << file << "' test images";
  }

  ModelSettings modelSettings;
  modelSettings.networkConfigFile = "OcvYoloDetection/models/yolov4.cfg";
  modelSettings.weightsFile = "OcvYoloDetection/models/yolov4.weights";
  modelSettings.namesFile = "OcvYoloDetection/models/coco.names";
  YoloNetwork yolo(modelSettings, cfg);

  GOUT("cv::dnn detections:\t");

  std::vector<std::vector<DetectionLocation>> detections;
  auto start_time = chrono::high_resolution_clock::now();
  for(int i=0; i<samples; i++){
    detections = yolo.GetDetections(frames, cfg);
  }
  auto end_time = chrono::high_resolution_clock::now();
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() / (float)samples * 1e-9;

  for(int f = 0; f < detections.size(); f++){
    GOUT("\tframe["<< f << "]");
    std::sort(detections[f].begin(),detections[f].end(),comp_xy);
    iGen.WriteDetectionOutputImage(testFiles[f], mpfImageLocations(detections[f]),
       std::string("frame"+std::to_string(f)+"_OCV.jpg"));
    for(int d = 0; d < detections[f].size(); d++){
      GOUT("\t\tdet[" << d <<"]:" << detections[f][d]);
    }
  }


  cfg.trtisEnabled = true;
  GOUT("triton detections:\t");
  component.Init();

  std::vector<std::vector<DetectionLocation>> detectionsTritonGRPC;
  start_time = chrono::high_resolution_clock::now();
  for(int i=0; i<samples; i++){
    detectionsTritonGRPC = yolo.GetDetections(frames, cfg);
  }
  end_time = chrono::high_resolution_clock::now();
  double grpc_time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() / (float)samples * 1e-9;

  for(int f = 0; f < detectionsTritonGRPC.size(); f++){
    GOUT("\tframe["<< f << "]");
    std::sort(detectionsTritonGRPC[f].begin(),detectionsTritonGRPC[f].end(),comp_xy);
    iGen.WriteDetectionOutputImage(testFiles[f], mpfImageLocations(detectionsTritonGRPC[f]),
       std::string("frame"+std::to_string(f)+"_GRPC.jpg"));
    for(int d = 0; d < detectionsTritonGRPC[f].size(); d++){
      GOUT("\t\tdet[" << d <<"]:" << detectionsTritonGRPC[f][d]);
    }
  }


  cfg.trtisUseShm = true;
  GOUT("triton shm detections:\t");
  component.Init();

  std::vector<std::vector<DetectionLocation>> detectionsTritonSHM;
  start_time = chrono::high_resolution_clock::now();
  for(int i=0; i<samples; i++){
    detectionsTritonSHM = yolo.GetDetections(frames, cfg);
  }
  end_time = chrono::high_resolution_clock::now();
  double shm_time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() / (float)samples * 1e-9;

  for(int f = 0; f < detectionsTritonSHM.size(); f++){
     GOUT("\tframe["<< f << "]");
    std::sort(detectionsTritonSHM[f].begin(),detectionsTritonSHM[f].end(),comp_xy);
    iGen.WriteDetectionOutputImage(testFiles[f], mpfImageLocations(detectionsTritonSHM[f]),
       std::string("frame"+std::to_string(f)+"_SHM.jpg"));
    for(int d = 0; d < detectionsTritonSHM[f].size(); d++){
      GOUT("\t\tdet[" << d <<"]:" << detectionsTritonSHM[f][d]);
    }
  }



  GOUT("OCVDNN inferencing time time: " << fixed << setprecision(5) << time_taken/frames.size() << "[sec] or " << frames.size()/time_taken << "[FPS]");
  GOUT("Triton inferencing time time: " << fixed << setprecision(5) << grpc_time_taken/frames.size() << "[sec] or " << frames.size()/grpc_time_taken << "[FPS]");
  GOUT("Triton Shm inferencing time time: " << fixed << setprecision(5) << shm_time_taken/frames.size() << "[sec] or " << frames.size()/shm_time_taken << "[FPS]");

  // make sure corresponding detections match
  for(int f = 0; f < frames.size(); f++){
    ASSERT_EQ(detections[f].size(), detectionsTritonGRPC[f].size()) << "GRPC detection count for frame[" << f <<"] doesn't match ocvdnn";
    ASSERT_EQ(detections[f].size(), detectionsTritonSHM[f].size()) << "SHM detection count for frame[" << f <<"] doesn't match ocvdnn";
    const float minIOU = 0.85;
    std::vector<DetectionLocation> poorMatches;
    for(int d = 0; d < detections[f].size(); d++){
      float max_iou;
      DetectionLocation& best = detections[f][0];
      maxIOU(detectionsTritonGRPC[f][d],detections[f], best, max_iou);
      if(max_iou < minIOU){
        GOUT("iou = " << max_iou << " GRPC detection {" << detectionsTritonGRPC[f][d] << "} in frame[" << f <<"] doesn't match ocvdnn {" << best << "}");
        poorMatches.push_back(detectionsTritonGRPC[f][d]);
        poorMatches.push_back(best);
      }
      maxIOU(detectionsTritonSHM[f][d],detections[f], best, max_iou);
      if(max_iou < minIOU){
        GOUT("iou = " << max_iou << " SHM detection {" << detectionsTritonSHM[f][d] << "} in frame[" << f <<"] doesn't match ocvdnn {" << best << "}");
        poorMatches.push_back(detectionsTritonSHM[f][d]);
        poorMatches.push_back(best);
      }
    }
    if(!poorMatches.empty()){
      iGen.WriteDetectionOutputImage(testFiles[f], mpfImageLocations(poorMatches),
        std::string("frame"+std::to_string(f)+"_bad.jpg"));
    }
  }

}


////////////////////////////////////////////////////////////////////////////////

void write_track_output_video(string inVideoFileName, vector<MPFVideoTrack>& tracks, string outVideoFileName, MPFVideoJob& videoJob){

   MPFVideoCapture cap(inVideoFileName);
   //MPFVideoJob temp("Testing", videoJob.data_uri, 0, videoJob.stop_frame, {}, {});

   //MPFVideoCapture cap(temp,false,false);

   //cv::VideoCapture cap(videoJob.data_uri);
   cv::VideoWriter writer(outVideoFileName,
                //cv::VideoWriter::fourcc('X', '2', '6', '4'),
                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                //cv::VideoWriter::fourcc('A', 'V', 'C', '1'),
                //cv::VideoWriter::fourcc('M', 'P', '4', 'V'),
                cap.GetFrameRate(), cap.GetFrameSize());
  GOUT("\tSorting tracks into frames");
   map<int,vector<MPFVideoTrack*>> frameTracks;
   for(auto& track:tracks){
     for(auto& det:track.frame_locations){
       if(det.first < track.start_frame || det.first > track.stop_frame){
         GOUT("\tdetection index" + to_string(det.first) + " outside of track frame range [" + to_string(track.start_frame) + "," + to_string(track.stop_frame) + "]");
       }
     }
    for(int fr = track.start_frame; fr <= track.stop_frame; fr++){
      frameTracks[fr].push_back(&track);
    }
   }

   GOUT("\tRendering tracks");
   cv::Mat frame;
   int frameIdx = cap.GetCurrentFramePosition();
   int calFrameIdx = round(cap.GetCurrentTimeInMillis() * cap.GetFrameRate() / 1000.0);

   while(cap.Read(frame)){
     if(frameIdx > videoJob.stop_frame) break;
     if(frameIdx >= videoJob.start_frame){
      map<int,vector<MPFVideoTrack*>>::iterator tracksItr = frameTracks.find(frameIdx);
      if(tracksItr != frameTracks.end()){
        for(MPFVideoTrack* trackPtr:tracksItr->second){
          map<int,MPFImageLocation>::iterator detItr = trackPtr->frame_locations.find(frameIdx);
          if(detItr != trackPtr->frame_locations.end()){
            cv::Rect detection_rect(detItr->second.x_left_upper, detItr->second.y_left_upper, detItr->second.width, detItr->second.height);
            cv::rectangle(frame, detection_rect, {255, 0, 0}, 2);
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

////////////////////////////////////////////////////////////////////////////////
TEST(OcvYoloDetection, TestTritonClientVideo) {


  int start = 0;
  int  stop = 336;
  int  rate = 1;
  string inVideoFile  = "data/Stockholm_Marathon_9_km.webm";
  string outTrackFile = "ocv_yolo_found_tracks.txt";
  string outVideoFile = "ocv_yolo_found_tracks.avi";
  float comparison_score_threshold = 0.6;

  GOUT("Start:\t"    << start);
  GOUT("Stop:\t"     << stop);
  GOUT("Rate:\t"     << rate);
  GOUT("outTrack:\t" << outTrackFile);
  GOUT("inVideo:\t"  << inVideoFile);
  GOUT("outVideo:\t" << outVideoFile);
  GOUT("comparison threshold:\t" << comparison_score_threshold);

  auto component = initComponent();

  MPFVideoJob videoJob("Testing", inVideoFile, start, stop,
     {{"KF_DISABLED","1"},
      {"DETECTION_FRAME_BATCH_SIZE","16"},
      {"MAX_INFER_CONCURRENCY", "4"},
      {"ENABLE_TRTIS", "true"},
      {"TRACKING_DISABLE_MOSSE_TRACKER","0"},
      {"TRTIS_SERVER","triton:8001"},
      {"TRTIS_USE_SHM", "true"},
      {"MODEL_NAME", "yolo"},
      {"NET_INPUT_IMAGE_SIZE", "416"}
     }, { });

  Config cfg(videoJob.job_properties);

  GOUT("Configuration:" << cfg);

  auto start_time = chrono::high_resolution_clock::now();
  vector<MPFVideoTrack> found_tracks = component.GetDetections(videoJob);
  auto end_time = chrono::high_resolution_clock::now();
  EXPECT_FALSE(found_tracks.empty());
  double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
  time_taken = time_taken * 1e-9;
  GOUT("\tVideoJob processing time: " << fixed << setprecision(5) << time_taken << "[sec] or "  << (stop - start)/time_taken << "[FPS]");

  GOUT("\tWriting detected video to files.");
  VideoGeneration video_generation;
  //video_generation.WriteTrackOutputVideo(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile));
  write_track_output_video(inVideoFile, found_tracks, outVideoFile, videoJob);
  GOUT("\tWriting test tracks to files.");
  WriteDetectionsToFile::WriteVideoTracks(outTrackFile, found_tracks);

  GOUT("\tClosing down detection.");
  EXPECT_TRUE(component.Close());

    return;
}
