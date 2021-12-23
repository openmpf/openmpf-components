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

#include <MPFImageReader.h>

#include "Config.h"
#include "Frame.h"
#include "DetectionLocation.h"
#include "Track.h"
#include "yolo_network/YoloNetwork.h"
#include "OcvYoloDetection.h"

#include "TestUtils.h"

using namespace MPF::COMPONENT;

class OcvLocalYoloDetectionTestFixture : public ::testing::Test {
protected:
    // Called once for all of the tests in this fixture.
    static void SetUpTestCase() {
        init_logging();
    }
};


/** ***************************************************************************
*   Test phase correlator and similarity score images
**************************************************************************** */
// TODO: Determine if this is worth saving. If it is, then clean it up.
TEST_F(OcvLocalYoloDetectionTestFixture, TestCorrelator) {
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
    modelSettings.ocvDnnNetworkConfigFile = "OcvYoloDetection/models/yolov4-tiny.cfg";
    modelSettings.ocvDnnWeightsFile = "OcvYoloDetection/models/yolov4-tiny.weights";
    modelSettings.namesFile = "OcvYoloDetection/models/coco.names";

    std::vector<Frame> frameBatch = {frame1};
    std::vector<std::vector<DetectionLocation>> detections;
    YoloNetwork(modelSettings, cfg).GetDetections(
      frameBatch,
      [&detections](std::vector<std::vector<DetectionLocation>> detectionsVec,
        std::vector<Frame>::const_iterator,
        std::vector<Frame>::const_iterator){
        detections = std::move(detectionsVec);
      },
      cfg);

    EXPECT_FALSE(detections[0].empty());

    auto dogItr=detections[0].begin();
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


TEST_F(OcvLocalYoloDetectionTestFixture, TestImage) {
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


TEST_F(OcvLocalYoloDetectionTestFixture, TestVideo) {
    auto jobProps = getTinyYoloConfig(0.92);
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


TEST_F(OcvLocalYoloDetectionTestFixture, TestInvalidModel) {
    ModelSettings modelSettings;
    modelSettings.ocvDnnNetworkConfigFile = "fake config";
    modelSettings.ocvDnnWeightsFile = "fake weights";
    modelSettings.namesFile = "fake names";
    Config config({});

    try {
        YoloNetwork network(modelSettings, config);
        FAIL() << "Expected exception not thrown.";
    }
    catch (const MPFDetectionException &ex) {
        ASSERT_EQ(MPF_COULD_NOT_READ_DATAFILE, ex.error_code);
    }
}


TEST_F(OcvLocalYoloDetectionTestFixture, TestWhitelist) {
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
