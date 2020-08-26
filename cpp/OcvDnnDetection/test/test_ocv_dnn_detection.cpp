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
#include <MPFDetectionComponent.h>

#include <gtest/gtest.h>

#include "OcvDnnDetection.h"

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
            { "MODEL_OUTPUT_LAYER", "softmax_tensor" },
            { "MODEL_INPUT_NAME", "input_placeholder" },
            { "RESIZE_HEIGHT", "224" },
            { "RESIZE_WIDTH", "224" },
            { "SUBTRACT_BLUE_VALUE", "92.81" },
            { "SUBTRACT_GREEN_VALUE", "88.55" },
            { "SUBTRACT_RED_VALUE", "84.77" },
    };
}


bool containsObject(const std::string &object_name, const Properties &props) {
    auto class_prop_iter = props.find("CLASSIFICATION");
    return class_prop_iter != props.end() && class_prop_iter->second == object_name;
}

bool containsObject(const std::string &object_name,
                    const std::vector<MPFImageLocation> &locations) {
    return std::any_of(
        locations.begin(),
        locations.end(),
        [&](const MPFImageLocation &location) {
            return containsObject(object_name, location.detection_properties);
        }
    );
}

bool containsObject(const std::string &object_name,
                    const std::vector<MPFVideoTrack> &tracks) {
    return std::any_of(
        tracks.begin(),
        tracks.end(),
        [&](const MPFVideoTrack &track) {
            return containsObject(object_name, track.detection_properties);
        }
    );
}


void assertObjectDetectedInImage(const std::string &expected_object,
                                 const std::string &image_path,
                                 OcvDnnDetection &ocv_dnn_component) {
    MPFImageJob job("Test", image_path, getGoogleNetProperties(), {});

    std::vector<MPFImageLocation> image_locations = ocv_dnn_component.GetDetections(job);

    ASSERT_FALSE(image_locations.empty());
    ASSERT_TRUE(containsObject(expected_object, image_locations))
                << "Expected GoogleNet to detect a \"" << expected_object << "\" in " << image_path;
}

TEST(OCVDNN, GoogleNetImageTest) {

    OcvDnnDetection ocv_dnn_component;
    ocv_dnn_component.SetRunDirectory("../plugin");

    ASSERT_TRUE(ocv_dnn_component.Init());

    ASSERT_NO_FATAL_FAILURE(assertObjectDetectedInImage("digital clock", "data/digital-clock.jpg", ocv_dnn_component));
    ASSERT_NO_FATAL_FAILURE(assertObjectDetectedInImage("sundial", "data/sundial.jpg", ocv_dnn_component));

    ASSERT_TRUE(ocv_dnn_component.Close());
}


void assertObjectDetectedInVideo(const std::string &object_name, const Properties &job_props, OcvDnnDetection &ocv_dnn_component) {
    MPFVideoJob job("TEST", "data/ff-region-object-motion.avi", 10, 15, job_props, {});

    std::vector<MPFVideoTrack> tracks = ocv_dnn_component.GetDetections(job);

    ASSERT_FALSE(tracks.empty());
    ASSERT_TRUE(containsObject(object_name, tracks));
}

TEST(OCVDNN, GoogleNetVideoTest) {
    OcvDnnDetection ocv_dnn_component;
    ocv_dnn_component.SetRunDirectory("../plugin");

    ASSERT_TRUE(ocv_dnn_component.Init());

    Properties job_props = getGoogleNetProperties();
    job_props["SEARCH_REGION_ENABLE_DETECTION"] = "true";
    job_props["SEARCH_REGION_BOTTOM_RIGHT_X_DETECTION"] = "340";
    ASSERT_NO_FATAL_FAILURE(assertObjectDetectedInVideo("digital clock", job_props, ocv_dnn_component));

    Properties job_props2 = getGoogleNetProperties();
    job_props2["SEARCH_REGION_ENABLE_DETECTION"] = "true";
    job_props2["SEARCH_REGION_TOP_LEFT_X_DETECTION"] = "340";
    ASSERT_NO_FATAL_FAILURE(assertObjectDetectedInVideo("sundial", job_props2, ocv_dnn_component));

    ASSERT_TRUE(ocv_dnn_component.Close());
}


TEST(OCVDNN, GoogleNetSpectralHashTest) {
    OcvDnnDetection ocv_dnn_component;
    ocv_dnn_component.SetRunDirectory("../plugin");
    ASSERT_TRUE(ocv_dnn_component.Init());

    Properties job_props = getGoogleNetProperties();
    job_props["SPECTRAL_HASH_FILE_LIST"] =
            "../plugin/OcvDnnDetection/models/bvlc_googlenet_spectral_hash.json; fake_hash_file.asdf";
    job_props["ACTIVATION_LAYER_LIST"] = "prob;inception_3a/relu_1x1";

    MPFImageJob job("Test", "data/sundial.jpg", job_props, {});


    std::vector<MPFImageLocation> image_locations = ocv_dnn_component.GetDetections(job);

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


void assertVehicleColorDetectedInImage(const std::string &expected_color,
                                       const std::string &image_path,
                                       OcvDnnDetection &ocv_dnn_component) {
    MPFImageJob job("Test", image_path, getVehicleColorProperties(), {});

    std::vector<MPFImageLocation> image_locations = ocv_dnn_component.GetDetections(job);

    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_color, image_locations))
                << "Expected Vehicle Color model to detect a " << expected_color << " vehicle in " << image_path;
}

TEST(OCVDNN, VehicleColorImageTest) {

    OcvDnnDetection ocv_dnn_component;
    ocv_dnn_component.SetRunDirectory("../plugin");

    ASSERT_TRUE(ocv_dnn_component.Init());

    ASSERT_NO_FATAL_FAILURE(assertVehicleColorDetectedInImage("blue", "data/blue-car.jpg", ocv_dnn_component));
    ASSERT_NO_FATAL_FAILURE(assertVehicleColorDetectedInImage("red", "data/red-car.jpg", ocv_dnn_component));
    ASSERT_NO_FATAL_FAILURE(assertVehicleColorDetectedInImage("yellow", "data/yellow-car.jpg", ocv_dnn_component));

    ASSERT_TRUE(ocv_dnn_component.Close());
}

TEST(OCVDNN, FeedForwardImageTest) {

    OcvDnnDetection ocv_dnn_component;
    ocv_dnn_component.SetRunDirectory("../plugin");

    ASSERT_TRUE(ocv_dnn_component.Init());

    std::string expected_color = "blue";
    std::string image_path = "data/blue-car.jpg";
    std::string classification_type = "COLOR";

    MPFImageLocation person_location = {10, 20, 100, 200, 0.5,
                                        {{"CLASSIFICATION", "person"},
                                         {"CLASSIFICATION LIST", "person; gorilla; cat"},
                                         {"CLASSIFICATION CONFIDENCE LIST", "0.8; 0.1; 0.05"}}};

    MPFImageLocation vehicle_location = {10, 20, 100, 200, 0.5,
                                         {{"CLASSIFICATION", "car"},
                                          {"CLASSIFICATION LIST", "car; truck; bus"},
                                          {"CLASSIFICATION CONFIDENCE LIST", "0.6; 0.3; 0.05"}}};

    Properties props = getVehicleColorProperties();
    props["CLASSIFICATION_TYPE"] = classification_type;
    props["FEED_FORWARD_TYPE"] = "FRAME";
    props["FEED_FORWARD_WHITELIST_FILE"] = "vehicle-whitelist.txt";

    // Test feed-forward person pass-through

    props["FEED_FORWARD_EXCLUDE_BEHAVIOR"] = "PASS_THROUGH";
    MPFImageJob pass_job("Test", image_path, props, {});
    pass_job.feed_forward_location = person_location;
    pass_job.has_feed_forward_location = true;

    std::vector<MPFImageLocation> image_locations = ocv_dnn_component.GetDetections(pass_job);

    ASSERT_EQ(1, image_locations.size());
    ASSERT_EQ(person_location.detection_properties, image_locations.at(0).detection_properties);

    // Test feed-forward person drop

    props["FEED_FORWARD_EXCLUDE_BEHAVIOR"] = "DROP";
    MPFImageJob drop_job("Test", image_path, props, {});
    drop_job.feed_forward_location = person_location;
    drop_job.has_feed_forward_location = true;

    image_locations = ocv_dnn_component.GetDetections(drop_job);

    ASSERT_TRUE(image_locations.empty());

    // Test feed-forward vehicle processing

    MPFImageJob color_job("Test", image_path, props, {});
    color_job.feed_forward_location = vehicle_location;
    color_job.has_feed_forward_location = true;

    image_locations = ocv_dnn_component.GetDetections(color_job);

    ASSERT_EQ(1, image_locations.size());
    Properties &feed_forward_props = vehicle_location.detection_properties;
    Properties &loc_props = image_locations.at(0).detection_properties;

    // "CLASSIFICATION*" props from feed-forward location and new "COLOR*" props
    ASSERT_EQ(6, loc_props.size());

    ASSERT_EQ(feed_forward_props.at("CLASSIFICATION"),
              loc_props.at("CLASSIFICATION"));
    ASSERT_EQ(feed_forward_props.at("CLASSIFICATION LIST"),
              loc_props.at("CLASSIFICATION LIST"));
    ASSERT_EQ(feed_forward_props.at("CLASSIFICATION CONFIDENCE LIST"),
              loc_props.at("CLASSIFICATION CONFIDENCE LIST"));

    ASSERT_EQ(expected_color, loc_props.at(classification_type))
                                << "Expected Vehicle Color model to detect a " << expected_color << " vehicle in "
                                << image_path;
    ASSERT_GT(loc_props.count(classification_type + " LIST"), 0);
    ASSERT_GT(loc_props.count(classification_type + " CONFIDENCE LIST"), 0);

    ASSERT_TRUE(ocv_dnn_component.Close());
}

TEST(OCVDNN, FeedForwardVideoTest) {

    OcvDnnDetection ocv_dnn_component;
    ocv_dnn_component.SetRunDirectory("../plugin");

    ASSERT_TRUE(ocv_dnn_component.Init());

    int end_frame = 2;
    std::string expected_color = "red";
    std::string video_path = "data/lp-ferrari-texas-shortened.mp4";
    std::string classification_type = "COLOR";

    Properties feed_forward_track_props = {{"CLASSIFICATION", "car"}};
    MPFVideoTrack vehicle_track = {0, end_frame, 0.4, feed_forward_track_props};

    for (int i = 0; i <= end_frame; i++) {
        Properties feed_forward_location_props = {{"CLASSIFICATION",                 "car"},
                                                  {"CLASSIFICATION LIST",            "car; truck; bus"},
                                                  {"CLASSIFICATION CONFIDENCE LIST", std::to_string(0.2 + i / 10.0) +
                                                                                     "; 0.1; 0.05"}};
        vehicle_track.frame_locations[i] = {10, 20, 100, 200, 0.5, feed_forward_location_props};;
    }

    Properties props = getVehicleColorProperties();
    props["CLASSIFICATION_TYPE"] = classification_type;
    props["FEED_FORWARD_TYPE"] = "FRAME";
    props["FEED_FORWARD_WHITELIST_FILE"] = "vehicle-whitelist.txt";

    // Test feed-forward vehicle processing

    MPFVideoJob job("Test", video_path, 0, end_frame, props, {});
    job.feed_forward_track = vehicle_track;
    job.has_feed_forward_track = true;

    std::vector<MPFVideoTrack> tracks = ocv_dnn_component.GetDetections(job);

    ASSERT_EQ(1, tracks.size());

    // "CLASSIFICATION" prop from feed-forward track and new "COLOR" prop
    Properties &track_props = tracks.at(0).detection_properties;
    ASSERT_EQ(2, track_props.size());
    ASSERT_EQ(vehicle_track.detection_properties.at("CLASSIFICATION"),
              track_props.at("CLASSIFICATION"));
    ASSERT_EQ(expected_color, track_props.at(classification_type))
                                << "Expected Vehicle Color model to detect a " << expected_color << " vehicle in "
                                << video_path;

    ASSERT_EQ(end_frame + 1, tracks.at(0).frame_locations.size());
    for (int i = 0; i <= end_frame; i++) {
        Properties &feed_forward_props = vehicle_track.frame_locations.at(i).detection_properties;
        Properties &loc_props = tracks.at(0).frame_locations.at(i).detection_properties;

        // "CLASSIFICATION*" props from feed-forward location and new "COLOR*" props
        ASSERT_EQ(6, loc_props.size());

        ASSERT_EQ(feed_forward_props.at("CLASSIFICATION"),
                  loc_props.at("CLASSIFICATION"));
        ASSERT_EQ(feed_forward_props.at("CLASSIFICATION LIST"),
                  loc_props.at("CLASSIFICATION LIST"));
        ASSERT_EQ(feed_forward_props.at("CLASSIFICATION CONFIDENCE LIST"),
                  loc_props.at("CLASSIFICATION CONFIDENCE LIST"));

        ASSERT_EQ(expected_color, loc_props.at(classification_type))
                                    << "Expected Vehicle Color model to detect a " << expected_color << " vehicle in "
                                    << video_path << " in frame " << i;
        ASSERT_GT(loc_props.count(classification_type + " LIST"), 0);
        ASSERT_GT(loc_props.count(classification_type + " CONFIDENCE LIST"), 0);
    }

    ASSERT_TRUE(ocv_dnn_component.Close());
}