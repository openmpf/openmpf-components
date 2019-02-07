/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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
#include <MPFDetectionComponent.h>

#include <gtest/gtest.h>

#include "CaffeDetection.h"

using namespace MPF::COMPONENT;


Properties getGoogleNetProperties() {
    return {
            { "MODEL_NAME", "googlenet" },
            { "RESIZE_HEIGHT", "224" },
            { "RESIZE_WIDTH", "224" },
            { "SUBTRACT_BLUE_VALUE", "104.0" },
            { "SUBTRACT_GREEN_VALUE", "117.0" },
            { "SUBTRACT_RED_VALUE", "123.0" },
    };
}

Properties getVehicleColorProperties() {
    return {
            { "MODEL_NAME", "vehicle_color" },
            { "RESIZE_HEIGHT", "227" },
            { "RESIZE_WIDTH", "227" },
            { "SUBTRACT_BLUE_VALUE", "102.1" },
            { "SUBTRACT_GREEN_VALUE", "104.7" },
            { "SUBTRACT_RED_VALUE", "106.5" },
    };
}


bool containsObject(const std::string &object_name, const Properties &props) {
    auto class_prop_iter = props.find("CLASSIFICATION");
    return class_prop_iter != props.end() && class_prop_iter->second == object_name;
}


bool containsObject(const std::string &object_name, const std::vector<MPFImageLocation> &locations) {
    return std::any_of(locations.begin(), locations.end(),
                       [&](const MPFImageLocation &location) {
                           return containsObject(object_name, location.detection_properties);
                       });
}


void assertObjectDetectedInImage(const std::string &expected_object, const std::string &image_path,
                                 CaffeDetection &caffe) {
    MPFImageJob job("Test", image_path, getGoogleNetProperties(), {});

    std::vector<MPFImageLocation> image_locations;
    MPFDetectionError rc = caffe.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_object, image_locations))
                                << "Expected Caffe to detect a \"" << expected_object << "\" in " << image_path;
}

void assertCorrectColorDetectionInImage(const std::string &expected_color, const std::string &image_path,
                                 CaffeDetection &caffe) {
    MPFImageJob job("Test", image_path, getVehicleColorProperties(), {});

    std::vector<MPFImageLocation> image_locations;
    MPFDetectionError rc = caffe.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_color, image_locations))
                                << "Expected Caffe to detect a \"" << expected_color << "\" in " << image_path;
}

TEST(CAFFE, ImageTest) {

    CaffeDetection caffe;
    caffe.SetRunDirectory("../plugin");

    ASSERT_TRUE(caffe.Init());

    assertObjectDetectedInImage("digital clock", "test/digital-clock.jpg", caffe);
    assertObjectDetectedInImage("sundial", "test/sundial.jpg", caffe);
    assertCorrectColorDetectionInImage("red", "test/red.jpg", caffe);

    ASSERT_TRUE(caffe.Close());
}


bool containsObject(const std::string &object_name, const std::vector<MPFVideoTrack> &tracks) {
    return std::any_of(tracks.begin(), tracks.end(),
                       [&](const MPFVideoTrack &track) {
                           return containsObject(object_name, track.detection_properties);
                       });
}


void assertObjectDetectedInVideo(const std::string &object_name, const Properties &job_props, CaffeDetection &caffe) {
    MPFVideoJob job("TEST", "test/ff-region-object-motion.avi", 10, 15, job_props, {});

    std::vector<MPFVideoTrack> tracks;
    MPFDetectionError rc = caffe.GetDetections(job, tracks);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(tracks.empty());
    ASSERT_TRUE(containsObject(object_name, tracks));
}


TEST(CAFFE, VideoTest) {
    CaffeDetection caffe;
    caffe.SetRunDirectory("../plugin");

    ASSERT_TRUE(caffe.Init());

    Properties job_props = getGoogleNetProperties();
    job_props["SEARCH_REGION_ENABLE_DETECTION"] = "true";
    job_props["SEARCH_REGION_BOTTOM_RIGHT_X_DETECTION"] = "340";
    assertObjectDetectedInVideo("digital clock", job_props, caffe);

    Properties job_props2 = getGoogleNetProperties();
    job_props2["SEARCH_REGION_ENABLE_DETECTION"] = "true";
    job_props2["SEARCH_REGION_TOP_LEFT_X_DETECTION"] = "340";
    assertObjectDetectedInVideo("sundial", job_props2, caffe);

    ASSERT_TRUE(caffe.Close());
}


TEST(CAFFE, SpectralHashTest) {
    CaffeDetection caffe;
    caffe.SetRunDirectory("../plugin");
    ASSERT_TRUE(caffe.Init());

    Properties job_props = getGoogleNetProperties();
    job_props["SPECTRAL_HASH_FILE_LIST"] =
            "../plugin/CaffeDetection/models/bvlc_googlenet_spectral_hash.json; fake_hash_file.asdf";
    job_props["ACTIVATION_LAYER_LIST"] = "prob;inception_3a/relu_1x1";

    MPFImageJob job("Test", "test/sundial.jpg", job_props, {});


    std::vector<MPFImageLocation> image_locations;
    MPFDetectionError rc = caffe.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_EQ(image_locations.size(), 1);

    MPFImageLocation &location = image_locations.front();

    ASSERT_TRUE(containsObject("sundial", location.detection_properties));

    std::string expected_hash = "1110111011111000110010010100000011101010110001000001010011000011";
    std::string actual_hash = location.detection_properties["LOSS3/CLASSIFIER SPECTRAL HASH VALUE"];
    ASSERT_EQ(actual_hash, expected_hash);

    ASSERT_EQ(location.detection_properties["INVALID SPECTRAL HASH FILENAME LIST"], "fake_hash_file.asdf");

    std::string prob_activation = location.detection_properties["PROB ACTIVATION MATRIX"];
    ASSERT_NE(prob_activation.find("activation values"), std::string::npos);
    ASSERT_NE(prob_activation.find("opencv-matrix"), std::string::npos);

    std::string reluActivation = location.detection_properties["INCEPTION_3A/RELU_1X1 ACTIVATION MATRIX"];
    ASSERT_NE(reluActivation.find("activation values"), std::string::npos);
    ASSERT_NE(reluActivation.find("opencv-nd-matrix"), std::string::npos);
}
