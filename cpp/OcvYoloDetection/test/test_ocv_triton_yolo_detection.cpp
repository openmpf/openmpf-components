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

#include "Config.h"

#include "TestUtils.h"

using namespace MPF::COMPONENT;

class OcvTritonYoloDetectionTestFixture : public ::testing::Test {
protected:
    // Called once for all of the tests in this fixture.
    static void SetUpTestCase() {
        init_logging();
    }
};


TEST_F(OcvTritonYoloDetectionTestFixture, TestImageTriton) {
    MPFImageJob job("Test", "data/dog.jpg", getTritonYoloConfig(), {});

    auto detections = initComponent().GetDetections(job);
    ASSERT_EQ(3, detections.size());

    {
        const auto& dogDetection = findDetectionWithClass("dog", detections);
        ASSERT_NEAR(133, dogDetection.x_left_upper, 2);
        ASSERT_NEAR(228, dogDetection.y_left_upper, 2);
        ASSERT_NEAR(176, dogDetection.width, 2);
        ASSERT_NEAR(315, dogDetection.height, 2);
        ASSERT_NEAR(0.9899, dogDetection.confidence, 0.01);
        ASSERT_EQ("dog", dogDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto& bikeDetection = findDetectionWithClass("bicycle", detections);
        ASSERT_NEAR(121, bikeDetection.x_left_upper, 2);
        ASSERT_NEAR(136, bikeDetection.y_left_upper, 2);
        ASSERT_NEAR(455, bikeDetection.width, 2);
        ASSERT_NEAR(274, bikeDetection.height, 2);
        ASSERT_NEAR(0.9906, bikeDetection.confidence, 0.01);
        ASSERT_EQ("bicycle", bikeDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto& carDetection = findDetectionWithClass("truck", detections);
        ASSERT_NEAR(439, carDetection.x_left_upper, 2);
        ASSERT_NEAR(78, carDetection.y_left_upper, 3);
        ASSERT_NEAR(258, carDetection.width, 2);
        ASSERT_NEAR(89, carDetection.height, 2);
        ASSERT_NEAR(0.9686, carDetection.confidence, 0.01);
        ASSERT_EQ("truck", carDetection.detection_properties.at("CLASSIFICATION"));
    }
}


TEST_F(OcvTritonYoloDetectionTestFixture, TestVideoTriton) {
    auto jobProps = getTritonYoloConfig(0.92);
    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 2, 10,
                    jobProps, {});

    std::vector<MPFVideoTrack> expectedTracks =
      read_track_output("data/lp-ferrari-texas-shortened.tracks");

    auto foundTracks = initComponent().GetDetections(job);

    write_track_output_video(job.data_uri, foundTracks, "TestVideoTriton.avi", job);
    write_track_output(foundTracks, "TestVideoTriton.tracks", job);

    ASSERT_EQ(expectedTracks.size(), foundTracks.size());

    int numMatching = 0;
    int numNotMatching = 0;
    for(int et = 0; et < expectedTracks.size(); ++et){
      for(int ft = et; ft < foundTracks.size(); ++ft){
        float confDiff = 0;
        float aveIou = 0;
        if(same(expectedTracks.at(et), foundTracks.at(ft),
                0.001, 0.01, confDiff, aveIou)) {
          GOUT("Expected track #" << et << " matches found track #" << ft
                << " with confidence diff: " << setprecision(3) << confDiff
                << " and average iou: " << setprecision(3) << aveIou);
          ++numMatching;
        }else{
          ++numNotMatching;
        }
      }
    }

    ASSERT_EQ(numMatching, expectedTracks.size());

    ASSERT_EQ(numNotMatching,
       expectedTracks.size() * (expectedTracks.size() -1) / 2);
}


// TODO: Should we remove this?
TEST_F(OcvTritonYoloDetectionTestFixture, DISABLED_TestTritonPerformance) {

  int start = 0;
  int stop = 336;
  int rate = 1;
  string inVideoFile  = "data/Stockholm_Marathon_9_km.webm";
  string outTrackFile = "Stockholm_Marathon_9_km.tracks";
  string outVideoFile = "Stockholm_Marathon_9_km.tracks.avi";
  float comparison_score_threshold = 0.6;

  GOUT("Start:\t"    << start);
  GOUT("Stop:\t"     << stop);
  GOUT("Rate:\t"     << rate);
  GOUT("outTrack:\t" << outTrackFile);
  GOUT("inVideo:\t"  << inVideoFile);
  GOUT("outVideo:\t" << outVideoFile);
  GOUT("comparison threshold:\t" << comparison_score_threshold);

  auto component = initComponent();

  auto jobProp = getTritonYoloConfig();
  jobProp["TRACKING_DFT_SIZE"] = "128";
  jobProp["DETECTION_FRAME_BATCH_SIZE"] = "16";
  jobProp["TRITON_MAX_INFER_CONCURRENCY"] = "4";


  MPFVideoJob videoJob("Testing", inVideoFile, start, stop, jobProp, {});

  Config cfg(videoJob.job_properties);

  auto start_time = chrono::high_resolution_clock::now();
  vector<MPFVideoTrack> found_tracks = component.GetDetections(videoJob);
  auto end_time = chrono::high_resolution_clock::now();

  EXPECT_FALSE(found_tracks.empty());
  double time_taken =
    chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
  time_taken = time_taken * 1e-9;

  GOUT("\tVideoJob processing time: " << fixed << setprecision(5) << time_taken
     << "[sec] for " << stop-start << " frames or "
     << (stop - start)/time_taken << "[FPS]");

  GOUT("\t" <<found_tracks.size() << " tracks: " << outTrackFile);
  write_track_output(found_tracks, outTrackFile, videoJob);

  GOUT("\toverlay video: " << outVideoFile);
  write_track_output_video(inVideoFile, found_tracks, outVideoFile, videoJob);

  EXPECT_TRUE(component.Close());
}