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


#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv/cv.hpp>

#include <MPFDetectionComponent.h>
#include <fstream>
#include <unordered_set>
#include <MPFVideoCapture.h>
#include <ModelsIniParser.h>

#include "DarknetDetection.h"
#include "Trackers.h"
#include "include/DarknetInterface.h"

using namespace MPF::COMPONENT;


Properties get_yolo_tiny_config(float confidence = 0.5) {
    return {
            { "MODEL_NAME", "tiny yolo" },
            { "CONFIDENCE_THRESHOLD", std::to_string(confidence) }
    };
}

bool almost_equal(float f1, float f2) {
    return std::abs(f1 - f2) < 0.0001;
}

bool object_found(const std::string &expected_object_name, const Properties &detection_properties) {
    return expected_object_name == detection_properties.at("CLASSIFICATION");

}

bool object_found(const std::string &expected_object_name, const std::vector<MPFImageLocation> &detections) {
    for (const auto &detection : detections) {
        if (object_found(expected_object_name, detection.detection_properties)) {
            return true;
        }
    }
    return false;
}


bool object_found(const std::string &expected_object_name, int frame_number,
                  const std::vector<MPFVideoTrack> &tracks) {

    for (const auto &track : tracks) {
        if (track.start_frame == frame_number
                && object_found(expected_object_name, track.detection_properties)
                && object_found(expected_object_name, track.frame_locations.at(frame_number).detection_properties)) {
            return true;
        }
    }
    return false;
}



DarknetDetection init_component() {
    DarknetDetection component;
    component.SetRunDirectory("../plugin/");
    component.Init();
    return component;
}


TEST(Darknet, ImageTest) {
    MPFImageJob job("Test", "data/dog.jpg", get_yolo_tiny_config(), {});


    DarknetDetection component = init_component();

    std::vector<MPFImageLocation> results;
    MPFDetectionError rc = component.GetDetections(job, results);
    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);

    ASSERT_TRUE(object_found("dog", results));
    ASSERT_TRUE(object_found("car", results));
    ASSERT_TRUE(object_found("bicycle", results));
}



TEST(Darknet, VideoTest) {
    int end_frame = 4;
    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, get_yolo_tiny_config(), { });

    DarknetDetection component = init_component();
    std::vector<MPFVideoTrack> results;
    MPFDetectionError rc = component.GetDetections(job, results);
    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);

    for (int i = 0; i <= end_frame; i++) {
        ASSERT_TRUE(object_found("person", i, results));
        ASSERT_TRUE(object_found("car", i, results));
    }
}



bool object_found_in_all_frames(const std::string &object_type, const MPFVideoTrack &track, int start, int stop) {
    bool track_valid = object_found(object_type, track.detection_properties)
                       && track.start_frame == start
                       && track.stop_frame == stop;
    if (!track_valid) {
        return false;
    }

    for (int i = start; i <= stop; i++) {
        const MPFImageLocation &loc = track.frame_locations.at(i);
        if (!object_found(object_type, loc.detection_properties)) {
            return false;
        }
    }
    return true;
}


TEST(Darknet, UsePreprocessorVideoTest) {
    int end_frame = 4;
    Properties job_properties = get_yolo_tiny_config();
    job_properties.emplace("USE_PREPROCESSOR", "TRUE");

    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, job_properties, { });

    DarknetDetection component = init_component();
    std::vector<MPFVideoTrack> results;
    MPFDetectionError rc = component.GetDetections(job, results);
    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);

    ASSERT_EQ(results.size(), 2);

    ASSERT_TRUE(object_found_in_all_frames("car", results.at(0), 0, end_frame)
                || object_found_in_all_frames("car", results.at(1), 0, end_frame));

    ASSERT_TRUE(object_found_in_all_frames("person", results.at(0), 0, end_frame)
                || object_found_in_all_frames("person", results.at(1), 0, end_frame));
}


DarknetResult CreateDetection(const std::string &object_type, float confidence) {
    return DarknetResult { cv::Rect(0, 0, 10, 10), { { confidence, object_type } } };
}


bool TrackMatches(const MPFVideoTrack &track, const std::string &object_type, int start, int stop, float confidence) {
    bool track_fields_match = track.detection_properties.at("CLASSIFICATION") == object_type
           && track.start_frame == start
           && track.stop_frame == stop
           && almost_equal(track.confidence, confidence);
    if (!track_fields_match) {
        return false;
    }

    for (int i = start; i <= stop; i++) {
        if (object_type != track.frame_locations.at(i).detection_properties.at("CLASSIFICATION")) {
            return false;
        }
    }
    return true;
}

bool ContainsTrack(const std::vector<MPFVideoTrack> &tracks, const std::string &object_type, int start, int stop,
                   float confidence) {
    for (const auto& track : tracks) {
        if (TrackMatches(track, object_type, start, stop, confidence))  {
            return true;
        }
    }
    return false;
}

TEST(Darknet, PreprocessorTrackerTest) {

    PreprocessorTracker tracker;

    tracker.ProcessFrameDetections(
            { CreateDetection("person", 0.5), CreateDetection("dog", 0.5) },
            0);

    tracker.ProcessFrameDetections(
            { CreateDetection("person", 0.6), CreateDetection("dog", 0.5)},
            1
    );

    tracker.ProcessFrameDetections(
            { CreateDetection("person", 0.8), CreateDetection("dog", 0.5), CreateDetection("cat", 0.5)},
            2
    );

    tracker.ProcessFrameDetections(
            { CreateDetection("dog", 0.5), CreateDetection("cat", 0.95)},
            3
    );

    tracker.ProcessFrameDetections(
            { CreateDetection("dog", 0.5), CreateDetection("cat", 0.5), CreateDetection("person", 0.65)},
            4
    );

    tracker.ProcessFrameDetections(
            { CreateDetection("dog", 0.5), CreateDetection("person", 0.9), CreateDetection("person", 0.3)},
            5
    );

    std::vector<MPFVideoTrack> tracks = tracker.GetTracks();
    ASSERT_EQ(4, tracks.size());
    ASSERT_TRUE(ContainsTrack(tracks, "person", 0, 2, 0.8));
    ASSERT_TRUE(ContainsTrack(tracks, "person", 4, 5, 0.93));
    ASSERT_TRUE(ContainsTrack(tracks, "cat", 2, 4, 0.95));
    ASSERT_TRUE(ContainsTrack(tracks, "dog", 0, 5, 0.5));
}



bool has_track_with_confidence(const std::vector<MPFVideoTrack> &tracks, const std::string &type, float confidence) {
    for (auto &track : tracks) {
        if (track.detection_properties.at("CLASSIFICATION") == type && almost_equal(track.confidence, confidence)) {
            return true;
        }
    }
    return false;
}

bool has_image_location_with_confidence(const std::vector<MPFVideoTrack> &tracks, const std::string &type,
                                        float confidence) {
    for (auto &track : tracks) {
        for (auto &im_loc_pair : track.frame_locations) {
            auto &im_loc = im_loc_pair.second;
            if (im_loc.detection_properties.at("CLASSIFICATION") == type && almost_equal(im_loc.confidence, confidence)) {
                return true;
            }
        }
    }
    return false;
}


TEST(Darknet, TestPreprocessorConfidenceCalculation) {
    float p1_confidence = 0.45;
    float p2_confidence = 0.75;
    float prob_not_p1_and_not_p2 = (1 - p1_confidence) * (1 - p2_confidence);
    float prob_p1_or_p2 = 1 - prob_not_p1_and_not_p2;

    float p3_confidence = 0.25;
    float prob_not_p1_and_not_p2_and_not_p3 = (1 - p1_confidence) * (1 - p2_confidence) * (1 - p3_confidence);
    float prob_p1_or_p2_or_p3 = 1 - prob_not_p1_and_not_p2_and_not_p3;

    float d1_confidence = 0.65;

    std::vector<DarknetResult> detections0(2);
    detections0.at(0).object_type_probs = { { p1_confidence, "person" }, { d1_confidence, "dog" } };
    detections0.at(1).object_type_probs = { { p2_confidence, "person" } };

    {
        PreprocessorTracker tracker;
        tracker.ProcessFrameDetections(detections0, 0);
        auto tracks = tracker.GetTracks();

        ASSERT_EQ(tracks.size(), 2);

        ASSERT_TRUE(has_track_with_confidence(tracks, "person", prob_p1_or_p2));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "person", prob_p1_or_p2));

        ASSERT_TRUE(has_track_with_confidence(tracks, "dog", d1_confidence));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "dog", d1_confidence));
    }

    {
        DarknetResult detections0_other_location;
        detections0_other_location.object_type_probs = { { p3_confidence, "person" } };
        auto detections_copy = detections0;
        detections_copy.push_back(detections0_other_location);

        PreprocessorTracker tracker;
        tracker.ProcessFrameDetections(detections_copy, 0);
        auto tracks = tracker.GetTracks();

        ASSERT_EQ(tracks.size(), 2);

        // Verify at P(A or B) or P(C) is the same as P(A or B or C)
        ASSERT_TRUE(has_track_with_confidence(tracks, "person", prob_p1_or_p2_or_p3));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "person", prob_p1_or_p2_or_p3));

        ASSERT_TRUE(has_track_with_confidence(tracks, "dog", d1_confidence));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "dog", d1_confidence));
    }


    {
        PreprocessorTracker tracker;
        tracker.ProcessFrameDetections(detections0, 0);

        DarknetResult detections1;
        detections1.object_type_probs = { { p3_confidence, "person" } };

        tracker.ProcessFrameDetections({ detections1 }, 1);

        auto tracks = tracker.GetTracks();

        ASSERT_EQ(tracks.size(), 2);

        ASSERT_TRUE(has_track_with_confidence(tracks, "person", prob_p1_or_p2));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "person", prob_p1_or_p2));
        // p3 is in a different frame so the confidence should not be changed
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "person", p3_confidence));

        ASSERT_TRUE(has_track_with_confidence(tracks, "dog", d1_confidence));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "dog", d1_confidence));
    }
}


bool has_confidence_values(const MPFVideoTrack &track, std::vector<float> expected_confidences) {

    std::istringstream iss(track.detection_properties.at("CLASSIFICATION CONFIDENCE LIST"));

    for (auto &expected_confidence : expected_confidences) {
        std::string temp;
        if (!std::getline(iss, temp, ';')) {
            return false;
        }
        float actual_confidence = std::stof(temp);
        if (!almost_equal(expected_confidence, actual_confidence)) {
            return false;
        }
    }
    return true;
}


TEST(Darknet, TestNumberOfClassifications) {
    DarknetResult detections1;
    detections1.object_type_probs = { { .1, "dog" }, { .2, "person" }, {.3, "cat"}, {.25, "apple"} };

    DarknetResult detections2;
    detections2.object_type_probs = { { .1, "person" }, { .25, "dog" }, {.25, "cat"}, {.1, "apple"} };

    SingleDetectionPerTrackTracker tracker(3);
    tracker.ProcessFrameDetections({ detections1, detections2 }, 0);

    auto tracks = tracker.GetTracks();
    ASSERT_EQ(tracks.size(), 2);

    auto &track1 = tracks.at(0);
    auto &track2 = tracks.at(1);

    ASSERT_EQ(track1.frame_locations.size(), 1);
    ASSERT_EQ(track2.frame_locations.size(), 1);

    ASSERT_FLOAT_EQ(track1.confidence, .3);
    ASSERT_EQ(track1.detection_properties.at("CLASSIFICATION"), "cat");
    ASSERT_EQ(track1.detection_properties.at("CLASSIFICATION LIST"), "cat; apple; person");
    ASSERT_TRUE(has_confidence_values(track1, { .3, .25, .2 }));

    ASSERT_FLOAT_EQ(track2.confidence, .25);
    ASSERT_EQ(track2.detection_properties.at("CLASSIFICATION"), "cat");
    ASSERT_EQ(track2.detection_properties.at("CLASSIFICATION LIST"), "cat; dog; apple");
    ASSERT_TRUE(has_confidence_values(track2, { .25, .25, .1 }));
}



TEST(Darknet, TestModelsIniParser) {
    ModelsIniParser<ModelSettings> parser;
    ModelSettings settings = parser.Init("../plugin/DarknetDetection/models")
            .RegisterField("network_config", &ModelSettings::network_config_file)
            .RegisterField("names", &ModelSettings::names_file)
            .RegisterField("weights", &ModelSettings::weights_file)
            .ParseIni("tiny yolo", "/opt/share/models/Darknet/");

    ASSERT_EQ(settings.network_config_file, "../plugin/DarknetDetection/models/tiny-yolo.cfg");
    ASSERT_EQ(settings.names_file, "../plugin/DarknetDetection/models/coco.names");
    ASSERT_EQ(settings.weights_file, "../plugin/DarknetDetection/models/tiny-yolo.weights");
}



TEST(Darknet, TestWhitelist) {
    Properties job_props = get_yolo_tiny_config();
    DarknetDetection component = init_component();

    {
        job_props["CLASS_WHITELIST_FILE"] = "data/test-whitelist.txt";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});

        std::vector<MPFImageLocation> results;
        MPFDetectionError rc = component.GetDetections(job, results);
        ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);

        ASSERT_TRUE(object_found("dog", results));
        ASSERT_TRUE(object_found("bicycle", results));
        ASSERT_FALSE(object_found("car", results));
    }

    {
        int end_frame = 2;
        setenv("TEST_ENV_VAR", "data", true);
        setenv("TEST_ENV_VAR2", "whitelist", true);
        job_props["CLASS_WHITELIST_FILE"] = "$TEST_ENV_VAR/test-${TEST_ENV_VAR2}.txt";

        MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, job_props, {});

        std::vector<MPFVideoTrack> results;
        MPFDetectionError rc = component.GetDetections(job, results);
        ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);

        for (int i = 0; i <= end_frame; i++) {
            ASSERT_TRUE(object_found("person", i, results));
            ASSERT_FALSE(object_found("car", i, results));
        }
    }
}



TEST(Darknet, TestInvalidWhitelist) {
    Properties job_props = get_yolo_tiny_config();

    DarknetDetection component = init_component();
    std::vector<MPFImageLocation> results;

    {
        job_props["CLASS_WHITELIST_FILE"] = "data/NOTICE";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});
        MPFDetectionError rc = component.GetDetections(job, results);
        ASSERT_EQ(rc, MPF_COULD_NOT_READ_DATAFILE);
    }
    {
        job_props["CLASS_WHITELIST_FILE"] = "FAKE_PATH";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});
        MPFDetectionError rc = component.GetDetections(job, results);
        ASSERT_EQ(rc, MPF_COULD_NOT_OPEN_DATAFILE);
    }
    {
        job_props["CLASS_WHITELIST_FILE"] = "$THIS_ENV_VAR_SHOULD_NOT_EXIST/FAKE_PATH";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});
        MPFDetectionError rc = component.GetDetections(job, results);
        ASSERT_EQ(rc, MPF_INVALID_PROPERTY);
    }
}
