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

#include "SceneChangeDetection.h"

using namespace MPF::COMPONENT;



MPFVideoJob createSceneJob(const std::string &uri){
	Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("Testing Scene Change");
    MPFVideoJob job(job_name, uri, 0,251,algorithm_properties, media_properties);
    return job;
}


void assertScenesDetected(const int &expected, const std::string &image_path,
                                 SceneChangeDetection &scenechange) {
    MPFVideoJob job = createSceneJob(image_path);

    std::vector<MPFVideoTrack> detections;
    MPFDetectionError rc = scenechange.GetDetections(job, detections);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(detections.empty());

    ASSERT_TRUE(detections.size() == expected)
                                << "Expected " << std::to_string(expected) << " scenes in " << image_path;
}


TEST(SCENECHANGE, VideoTest) {

    SceneChangeDetection scenechange;
    scenechange.SetRunDirectory("../plugin");

    ASSERT_TRUE(scenechange.Init());

    assertScenesDetected(2,"test/scene_change.mp4", scenechange);

    ASSERT_TRUE(scenechange.Close());
}







