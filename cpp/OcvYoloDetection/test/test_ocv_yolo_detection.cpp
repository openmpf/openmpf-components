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

#include <gtest/gtest.h>
#include <MPFDetectionObjects.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>

#include "DetectionLocation.h"
#include "OcvYoloDetection.h"
#include "YoloNetwork.h"
#include "Track.h"

using namespace MPF::COMPONENT;

/** ***************************************************************************
*   macros for "pretty" gtest messages
**************************************************************************** */
#define GTEST_BOX "[          ] "
#define GOUT(MSG){                                                            \
  std::cout << GTEST_BOX << MSG << std::endl;                                 \
}


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

// TODO Figure out how to fix. Caused by using ocv tracker in OcvYoloDetection::ProcessFrameDetections
TEST(OcvYoloDetection, DISABLED_TestTrackingError) {
    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 1, 2,
                    getTinyYoloConfig(0.95), {});

    auto tracks = initComponent().GetDetections(job);
    for (const auto& track : tracks) {
        for (const auto& imgLocPair : track.frame_locations) {
            ASSERT_EQ(1, imgLocPair.second.detection_properties.count("CLASSIFICATION"));
        }
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

