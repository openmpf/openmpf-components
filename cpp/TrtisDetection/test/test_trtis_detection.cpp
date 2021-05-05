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
#include <MPFDetectionComponent.h>
#include <VideoGeneration.h>
#include <WriteDetectionsToFile.h>
#include <ImageGeneration.h>

#include <gtest/gtest.h>
#include <log4cxx/propertyconfigurator.h>
#include "TrtisDetection.h"

using namespace MPF::COMPONENT;
using namespace std;

/** ***************************************************************************
*   macros for "pretty" gtest messages
**************************************************************************** */
#define ANSI_TXT_GRN "\033[0;32m"
#define ANSI_TXT_MGT "\033[0;35m" //Magenta
#define ANSI_TXT_DFT "\033[0;0m"  //Console default
#define GTEST_BOX "[          ] "
#define GOUT(MSG)                               \
  {                                             \
    std::cout << GTEST_BOX << MSG << std::endl; \
  }
#define GOUT_MGT(MSG)                                                           \
  {                                                                             \
    std::cout << ANSI_TXT_MGT << GTEST_BOX << MSG << ANSI_TXT_DFT << std::endl; \
  }
#define GOUT_GRN(MSG)                                                           \
  {                                                                             \
    std::cout << ANSI_TXT_GRN << GTEST_BOX << MSG << ANSI_TXT_DFT << std::endl; \
  }

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
                    const vector<MPFImageLocation> &locations) {
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

    vector<MPFImageLocation> image_locations = trtisDet.GetDetections(job);

    ImageGeneration image_generation;
    image_generation.WriteDetectionOutputImage(image_path,
                                               image_locations,
                                               "test/detections.png");

    ASSERT_FALSE(image_locations.empty());

    ASSERT_TRUE(containsObject(expected_object, image_locations))
            << "Expected Trtis to detect a \"" << expected_object << "\" in " << image_path;
}

bool init_logging() {
    log4cxx::PropertyConfigurator::configure("test/log4cxx.properties");
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
                    const vector<MPFVideoTrack> &tracks) {
    return any_of(tracks.begin(), tracks.end(),
                  [&](const MPFVideoTrack &track) {
                      return containsObject(object_name, track.detection_properties);
                  });
}

//------------------------------------------------------------------------------
void assertObjectDetectedInVideo(const string &object_name,
                                 const Properties &job_props,
                                 TrtisDetection &trtisDet) {
    MPFVideoJob job("TEST", "test/ff-region-object-motion.avi", 0, 12, job_props, {});

    vector<MPFVideoTrack> tracks = trtisDet.GetDetections(job);

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
//------------------------------------------------------------------------------
TEST(TRTIS, VideoTest2) {
    TrtisDetection trtisDet;
    trtisDet.SetRunDirectory("../plugin");

    ASSERT_TRUE(trtisDet.Init());

    Properties job_props = getProperties_ip_irv2_coco();
    job_props["USER_FEATURE_ENABLE"] = "false";
    job_props["FRAME_FEATURE_ENABLE"] = "false";
    job_props["EXTRA_FEATURE_ENABLE"] = "false";
    job_props["MAX_INFER_CONCURRENCY"] = "10";
    job_props["CONTEXT_WAIT_TIMEOUT_SEC"] = "60";
    MPFVideoJob job("TEST",
                    //"test/big_buck_bunny.mp4",
                    "test/ped_short.mp4",
                        0,
                      50,
                     //144,
                     //1440,
                     job_props,
                     //{{"FPS","23.96"}});
                     {{"FPS","24.0"}});

    vector<MPFVideoTrack> tracks;
    auto start_time = chrono::high_resolution_clock::now();
    tracks = trtisDet.GetDetections(job);
    auto end_time = chrono::high_resolution_clock::now();
    double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count();
    time_taken = time_taken * 1e-9;
    int frame_count = job.stop_frame - job.start_frame + 1;
    double fps = frame_count / time_taken;
    GOUT("\tVideoJob processing time for "<< frame_count << " frames : " << fixed << setprecision(3) << time_taken << "[sec]");
    GOUT("\tVideoJob processing speed:" << fixed << setprecision(2) << fps << " [FPS] or " << setprecision(3) << 1000.0f/fps << "[ms] per inference");
    ASSERT_FALSE(tracks.empty());

    GOUT("\tWriting detected video to files.");
    VideoGeneration video_generation;
    video_generation.WriteTrackOutputVideo(job.data_uri, tracks, "tracks.avi");
    //write_track_output_video(inVideoFile, found_tracks, (test_output_dir + "/" + outVideoFile), videoJob);
    GOUT("\tWriting test tracks to files.");
    WriteDetectionsToFile::WriteVideoTracks("tracks.txt", tracks);

    GOUT("\tClosing down detection.")

    ASSERT_TRUE(trtisDet.Close());
}
