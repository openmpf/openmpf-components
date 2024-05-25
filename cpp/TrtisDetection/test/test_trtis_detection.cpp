/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2024 The MITRE Corporation                                       *
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
#include <log4cxx/basicconfigurator.h>

#include "TrtisDetection.h"

using namespace MPF::COMPONENT;
using namespace std;

//------------------------------------------------------------------------------
Properties getProperties_ip_irv2_coco() {

    return {
            {"TRTIS_SERVER", ""}, // use env. var.
            {"MODEL_NAME", "ip_irv2_coco"}
    };
}

//------------------------------------------------------------------------------
bool containsObject(const string &object_name,
                    const Properties &props) {
    auto class_prop_iter = props.find("CLASSIFICATION");
    return class_prop_iter != props.end() && class_prop_iter->second == object_name;
}

//------------------------------------------------------------------------------
bool containsObject(const string &object_name,
                    const MPFImageLocationVec &locations) {
    return any_of(locations.begin(), locations.end(),
                  [&](const MPFImageLocation &location) {
                      return containsObject(object_name, location.detection_properties);
                  });
}

//------------------------------------------------------------------------------
void assertObjectDetectedInImage(const string &expected_object,
                                 const string &image_path,
                                 TrtisDetection &trtisDet) {
    MPFImageJob job("Test", image_path, getProperties_ip_irv2_coco(), {});

    MPFImageLocationVec image_locations = trtisDet.GetDetections(job);

    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_object, image_locations))
            << "Expected Trtis to detect a \"" << expected_object << "\" in " << image_path;
}

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();

//------------------------------------------------------------------------------
TEST(TRTIS, InitTest) {
    std::remove("../Testing/log/trtis-detection.log");
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");
    ASSERT_TRUE(trtisDet.Init());
    ASSERT_TRUE(trtisDet.Close());
}

//------------------------------------------------------------------------------
TEST(TRTIS, ImageTest) {
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    ASSERT_TRUE(trtisDet.Init());

    assertObjectDetectedInImage("clock", "test/digital-clock.jpg", trtisDet);
    assertObjectDetectedInImage("car", "test/traffic.jpg", trtisDet);

    ASSERT_TRUE(trtisDet.Close());
}

//------------------------------------------------------------------------------
bool containsObject(const string &object_name,
                    const MPFVideoTrackVec &tracks) {
    return any_of(tracks.begin(), tracks.end(),
                  [&](const MPFVideoTrack &track) {
                      return containsObject(object_name, track.detection_properties);
                  });
}

//------------------------------------------------------------------------------
void assertObjectDetectedInVideo(const string &object_name,
                                 const Properties &job_props,
                                 TrtisDetection &trtisDet) {
    MPFVideoJob job("TEST", "test/ff-region-object-motion.avi", 11, 12, job_props, {});

    MPFVideoTrackVec tracks = trtisDet.GetDetections(job);

    ASSERT_FALSE(tracks.empty());
    ASSERT_TRUE(containsObject(object_name, tracks));
}

//------------------------------------------------------------------------------
TEST(TRTIS, VideoTest) {
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    ASSERT_TRUE(trtisDet.Init());

    Properties job_props = getProperties_ip_irv2_coco();
    job_props["USER_FEATURE_ENABLE"] = "true";
    assertObjectDetectedInVideo("clock", job_props, trtisDet);

    ASSERT_TRUE(trtisDet.Close());
}
