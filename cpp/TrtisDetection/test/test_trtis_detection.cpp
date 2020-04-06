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

#define private public

#include "TrtisDetection.hpp"

using namespace MPF::COMPONENT;
using namespace std;

//------------------------------------------------------------------------------
Properties getProperties_ip_irv2_coco() {

    return {
            { "TRTIS_SERVER", "localhost:8001" },
            { "MODEL_NAME", "ip_irv2_coco" },
    };
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
    MPFImageJob job("Test", image_path, getProperties_ip_irv2_coco(), {});

    MPFImageLocationVec image_locations;
    MPFDetectionError rc = trtisDet.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
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
TEST(TRTIS, GetInferenceContextTest) {

    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    TrtisJobConfig cfg(MPFImageJob("Test", "foo.jpg", getProperties_ip_irv2_coco(), {}));

    ASSERT_TRUE(trtisDet.Init());
    trtisDet._niGetInferContexts(cfg);
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
//  TEST(TRTIS, PerfClientData) {
//
//     TrtisDetection trtisDet;
//     trtisDet.SetRunDirectory("../plugin");
//
//     ASSERT_TRUE(trtisDet.Init());
//
//     MPFImageJob job("Test", "test/digital-clock.jpg", getProperties_ip_irv2_coco(), {});
//
//     MPFImageReader image_reader(job);
//     cv::Mat img = image_reader.GetImage();
//     TrtisIpIrv2CocoJobConfig cfg(job, img.cols, img.rows);
//     cfg.maxInferConcurrency = 1;
//     uPtrInferCtx* ctx  = trtisDet._niGetInferContexts(cfg)[0];
//
//     LngVec shape;
//     BytVec imgDat;
//     trtisDet._ip_irv2_coco_prepImageData(cfg, img, ctx, shape, imgDat);
//
//     ofstream myfile;
//     myfile.open("image_input", ios::out | ios::binary);
//     myfile.write((char*)&imgDat[0],imgDat.size());
//     myfile.close();
//
//     FltVec bbox = {0.0f,0.0f,1.0f,1.0f};
//     myfile.open("bbox_input",ios::out | ios::binary);
//     myfile.write((char*)&bbox[0], sizeof(bbox));
//     myfile.close();
//
// }
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
    MPFVideoJob job("TEST", "test/ff-region-object-motion.avi", 10, 15, job_props, {});

    MPFVideoTrackVec tracks;
    MPFDetectionError rc = trtisDet.GetDetections(job, tracks);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(tracks.empty());
    ASSERT_TRUE(containsObject(object_name, tracks));
}
//------------------------------------------------------------------------------
/*
TEST(TRTIS, VideoTest) {
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    ASSERT_TRUE(trtisDet.Init());

    Properties job_props = getProperties_ip_irv2_coco();
    job_props["USER_FEATURE_ENABLE"] = "true";
    assertObjectDetectedInVideo("clock", job_props, trtisDet);

    ASSERT_TRUE(trtisDet.Close());
}
//------------------------------------------------------------------------------
TEST(TRTIS, VideoTest2) {
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    ASSERT_TRUE(trtisDet.Init());

    Properties job_props = getProperties_ip_irv2_coco();
    job_props["USER_FEATURE_ENABLE"] = "true";
    job_props["MAX_INFER_CONCURRENCY"] = "16";
    job_props["CONTEXT_WAIT_TIMEOUT_SEC"] = "60";
    MPFVideoJob job("TEST", "test/big_buck_bunny.mp4", 1, 1440, job_props, {});

    MPFVideoTrackVec tracks;
    ASSERT_EQ(trtisDet.GetDetections(job, tracks), MPF_DETECTION_SUCCESS);

    ASSERT_FALSE(tracks.empty());

    ASSERT_TRUE(trtisDet.Close());
}
*/
