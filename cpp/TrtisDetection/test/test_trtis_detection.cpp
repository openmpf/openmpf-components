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
#include <fstream>
#include <MPFDetectionComponent.h>
#include <MPFImageReader.h>
#include <gtest/gtest.h>

#include "TrtisDetection.hpp"

using namespace MPF::COMPONENT;
using namespace std;

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

//------------------------------------------------------------------------------
bool containsObject(const string     &object_name,
                    const Properties &props       ) {
    auto class_prop_iter = props.find("CLASSIFICATION");
    return class_prop_iter != props.end() && class_prop_iter->second == object_name;
}

//------------------------------------------------------------------------------
bool containsObject(const string              &object_name,
                    const MPFImageLocationVec &locations   ) {
    return any_of(locations.begin(), locations.end(),
                  [&](const MPFImageLocation &location) {
                  return containsObject(object_name, location.detection_properties);
                 });
}

//------------------------------------------------------------------------------
void assertObjectDetectedInImage(const string   &expected_object,
                                 const string   &image_path,
                                 TrtisDetection &trtisDet           ){
    MPFImageJob job("Test", image_path, {}, {});

    MPFImageLocationVec image_locations;

    try {
        image_locations = trtisDet.GetDetections(job);
    } catch (const MPFDetectionException &ex) {
        FAIL() << " GetDetections failed to process test image.";
    }

    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_object, image_locations))
                                << "Expected Trtis to detect a \"" << expected_object << "\" in " << image_path;
}

//------------------------------------------------------------------------------
TEST(TRTIS, InitTest) {

    try{
      std::remove("../Testing/log/trtis-detection.log");
    }catch(...){}
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
 TEST(TRTIS, ImageTimeoutTest) {

    MPFImageLocationVec image_locations;
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");
    ASSERT_TRUE(trtisDet.Init());

    for(int t : {1, 2, 5}){
      MPFImageJob job("Test", "test/traffic.jpg",{{"CONTEXT_WAIT_TIMEOUT_SEC",to_string(t)}},{});
      try {
        GOUT("Testing context timeout value " << to_string(t) << " [sec].");
        image_locations = trtisDet.GetDetections(job);
      }catch (const MPFDetectionException &ex) {
        GOUT("Got exception:" << ex.what());
      }
    }
 }

//------------------------------------------------------------------------------
bool containsObject(const string           &object_name,
                    const MPFVideoTrackVec &tracks) {
    return any_of(tracks.begin(), tracks.end(),
                  [&](const MPFVideoTrack &track) {
                  return containsObject(object_name, track.detection_properties);
                 });
}

//------------------------------------------------------------------------------
void assertObjectDetectedInVideo(const string     &object_name,
                                 const Properties &job_props,
                                 TrtisDetection   &trtisDet) {
    MPFVideoJob job("TEST", "test/ff-region-object-motion.avi", 11, 12, job_props, {});

    MPFVideoTrackVec tracks;
    try {
        tracks = trtisDet.GetDetections(job);
    } catch (const MPFDetectionException &ex) {
        FAIL() << " GetDetections failed to process test video.";
    }

    ASSERT_FALSE(tracks.empty());
    ASSERT_TRUE(containsObject(object_name, tracks));
}

//------------------------------------------------------------------------------
TEST(TRTIS, VideoTest) {
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    ASSERT_TRUE(trtisDet.Init());
    assertObjectDetectedInVideo("clock", {{"USER_FEATURE_ENABLE","true"}}, trtisDet);
    ASSERT_TRUE(trtisDet.Close());
}
//------------------------------------------------------------------------------
TEST(TRTIS, VideoTimeoutTest) {
    MPFImageLocationVec image_locations;
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");
    ASSERT_TRUE(trtisDet.Init());

    for(int t : {1, 2, 5}){
      MPFVideoJob job("TEST", "test/ff-region-object-motion.avi", 0, 12,
       {{"CONTEXT_WAIT_TIMEOUT_SEC",to_string(t)},
        {"USER_FEATURE_ENABLE","true"}}, {});
      MPFVideoTrackVec tracks;
      try {
        GOUT("Testing with " << to_string(t) << " [sec] timeout.");
        tracks = trtisDet.GetDetections(job);
      } catch (const MPFDetectionException &ex) {
         GOUT("Got exception:" << ex.what());
      }
    }
}