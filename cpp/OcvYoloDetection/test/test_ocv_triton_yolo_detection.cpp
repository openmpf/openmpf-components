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

using namespace std;
using namespace MPF::COMPONENT;

class OcvTritonYoloDetectionTestFixture : public ::testing::Test {
protected:
    // Called once for all of the tests in this fixture.
    static void SetUpTestCase() {
        init_logging();
    }
};

std::string TRITON_SERVER = "ocv-yolo-detection-server:8001";


// Expected output based on yolov4.dim608.bs16.cuda11.3.trt7.2.3.nvidia_geforce_rtx_2080_ti.engine.1.0.0
TEST_F(OcvTritonYoloDetectionTestFixture, TestImageTriton) {
    MPFImageJob job("Test", "data/dog.jpg", getTritonYoloConfig(TRITON_SERVER), {});

    auto detections = initComponent().GetDetections(job);
    ASSERT_EQ(3, detections.size());

    {
        const auto &dogDetection = findDetectionWithClass("dog", detections);
        ASSERT_NEAR(131, dogDetection.x_left_upper, 2);
        ASSERT_NEAR(226, dogDetection.y_left_upper, 2);
        ASSERT_NEAR(179, dogDetection.width, 2);
        ASSERT_NEAR(313, dogDetection.height, 2);
        ASSERT_NEAR(0.9853, dogDetection.confidence, 0.01);
        ASSERT_EQ("dog", dogDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto &bikeDetection = findDetectionWithClass("bicycle", detections);
        ASSERT_NEAR(122, bikeDetection.x_left_upper, 2);
        ASSERT_NEAR(124, bikeDetection.y_left_upper, 2);
        ASSERT_NEAR(449, bikeDetection.width, 2);
        ASSERT_NEAR(299, bikeDetection.height, 2);
        ASSERT_NEAR(0.935, bikeDetection.confidence, 0.01);
        ASSERT_EQ("bicycle", bikeDetection.detection_properties.at("CLASSIFICATION"));
    }
    {
        const auto &carDetection = findDetectionWithClass("truck", detections);
        ASSERT_NEAR(468, carDetection.x_left_upper, 2);
        ASSERT_NEAR(76, carDetection.y_left_upper, 3);
        ASSERT_NEAR(214, carDetection.width, 2);
        ASSERT_NEAR(93, carDetection.height, 2);
        ASSERT_NEAR(0.987, carDetection.confidence, 0.01);
        ASSERT_EQ("truck", carDetection.detection_properties.at("CLASSIFICATION"));
    }
}


// Expected output based on yolov4.dim608.bs16.cuda11.3.trt7.2.3.nvidia_geforce_rtx_2080_ti.engine.1.0.0
TEST_F(OcvTritonYoloDetectionTestFixture, TestVideoTriton) {
    auto jobProps = getTritonYoloConfig(TRITON_SERVER, 0.92);
    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 2, 10,
                    jobProps, {});

    auto tracks = initComponent().GetDetections(job);

    ASSERT_NEAR(7, tracks.size(), 2);

    for (auto &track : tracks) {
        std::string clazz = track.detection_properties["CLASSIFICATION"];
        if (clazz == "car") {
            ASSERT_NEAR(9, track.frame_locations.size(), 2);
        } else if (clazz == "person") {
            ASSERT_NEAR(5, track.frame_locations.size(), 4);
        } else {
            FAIL() << "Unexpected classification: " << clazz;
        }
    }
}


// Disabled as a unit test. Keeping as a development tool.
// Uncomment the lines in the OUTPUT sections to generate a track list and markup output.
TEST_F(OcvTritonYoloDetectionTestFixture, DISABLED_TestTritonPerformance) {

    int start = 0;
    int stop = 335;
    string inVideoFile = "data/Stockholm_Marathon_9_km.webm";
    // string outTrackFile = "Stockholm_Marathon_9_km.tracks"; // OUTPUT
    // string outVideoFile = "Stockholm_Marathon_9_km.tracks.avi";
    float comparison_score_threshold = 0.6;

    GOUT("Start:\t" << start);
    GOUT("Stop:\t" << stop);
    GOUT("inVideo:\t" << inVideoFile);
    // GOUT("outTrack:\t" << outTrackFile); // OUTPUT
    // GOUT("outVideo:\t" << outVideoFile);
    GOUT("comparison threshold:\t" << comparison_score_threshold);

    auto component = initComponent();

    auto jobProp = getTritonYoloConfig(TRITON_SERVER);

    MPFVideoJob videoJob("Testing", inVideoFile, start, stop, jobProp, {});

    Config cfg(videoJob.job_properties);

    auto start_time = chrono::high_resolution_clock::now();
    vector<MPFVideoTrack> found_tracks = component.GetDetections(videoJob);
    auto end_time = chrono::high_resolution_clock::now();

    EXPECT_FALSE(found_tracks.empty());
    double time_taken =
            chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    time_taken = time_taken * 1e-9;

    int detections = 0;
    for (const auto& track: found_tracks) {
        detections += track.frame_locations.size();
    }
    GOUT("Found " << detections << " total detections.");

    GOUT("\tVideoJob processing time: " << fixed << setprecision(5) << time_taken
                                        << "[sec] for " << stop - start << " frames or "
                                        << (stop - start) / time_taken << "[FPS]");

    // GOUT("\t" << found_tracks.size() << " tracks: " << outTrackFile); // OUTPUT
    // write_track_output(found_tracks, outTrackFile, videoJob);

    // GOUT("\toverlay video: " << outVideoFile); // OUTPUT
    // write_track_output_video(inVideoFile, found_tracks, outVideoFile, videoJob);

    EXPECT_TRUE(component.Close());
}


TEST_F(OcvTritonYoloDetectionTestFixture, TestImageLocalAndTriton) {

    auto localJobProp = getYoloConfig();
    MPFImageJob localImageJob("LocalTest", "data/dog.jpg", localJobProp, {});

    auto tritonJobProp = getTritonYoloConfig(TRITON_SERVER);
    MPFImageJob tritonImageJob("TritonTest", "data/dog.jpg", tritonJobProp, {});

    auto component = initComponent();

    auto tracks = component.GetDetections(localImageJob);
    ASSERT_EQ(3, tracks.size());

    tracks = component.GetDetections(tritonImageJob);
    ASSERT_EQ(3, tracks.size());

    tracks = component.GetDetections(localImageJob);
    ASSERT_EQ(3, tracks.size());

    tracks = component.GetDetections(tritonImageJob);
    ASSERT_EQ(3, tracks.size());

    tracks = component.GetDetections(localImageJob);
    ASSERT_EQ(3, tracks.size());
    tracks = component.GetDetections(localImageJob);
    ASSERT_EQ(3, tracks.size());

    tracks = component.GetDetections(tritonImageJob);
    ASSERT_EQ(3, tracks.size());
    tracks = component.GetDetections(tritonImageJob);
    ASSERT_EQ(3, tracks.size());

    EXPECT_TRUE(component.Close());
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    if (argc > 1) {
        TRITON_SERVER = argv[1];
    }
    GOUT("Using TRITON_SERVER: " << TRITON_SERVER);
    return RUN_ALL_TESTS();
}
