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


#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <log4cxx/basicconfigurator.h>

#include <MPFDetectionComponent.h>
#include <MPFVideoCapture.h>

#include <fstream>
#include <unordered_set>
#include <MPFVideoCapture.h>
#include <ModelsIniParser.h>

#include "DarknetDetection.h"
#include "DarknetStreamingDetection.h"
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
        if (frame_number >= track.start_frame && frame_number <= track.stop_frame
                && object_found(expected_object_name, track.detection_properties)
                && object_found(expected_object_name, track.frame_locations.at(frame_number).detection_properties)) {
            return true;
        }
    }
    return false;
}

bool object_found(const std::string &expected_object_name, int frame_number, const MPFVideoTrack &track) {
    return object_found(expected_object_name, frame_number, std::vector<MPFVideoTrack>{ track });
}

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();

DarknetDetection init_component() {
    DarknetDetection component;
    component.SetRunDirectory("../plugin/");
    component.Init();
    return component;
}


TEST(Darknet, ImageTest) {
    for (bool use_preprocessor : { true, false }) {
        Properties job_properties = get_yolo_tiny_config();
        if (use_preprocessor) {
            job_properties.emplace("USE_PREPROCESSOR", "TRUE");
        }
        MPFImageJob job("Test", "data/dog.jpg", job_properties, {});

        DarknetDetection component = init_component();

        std::vector<MPFImageLocation> results = component.GetDetections(job);

        ASSERT_TRUE(object_found("dog", results));
        ASSERT_TRUE(object_found("car", results));
        ASSERT_TRUE(object_found("bicycle", results));
    }
}


TEST(Darknet, VideoTest) {
    int end_frame = 4;
    MPFVideoJob job("Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, get_yolo_tiny_config(), { });

    DarknetDetection component = init_component();
    std::vector<MPFVideoTrack> results = component.GetDetections(job);

    for (int i = 0; i <= end_frame; i++) {
        ASSERT_TRUE(object_found("person", i, results));
        ASSERT_TRUE(object_found("car", i, results));
    }
}


TEST(DarknetStreaming, VideoTest) {
    int end_frame = 4;
    MPFStreamingVideoJob job("Test", "../plugin/", get_yolo_tiny_config(), {});
    DarknetStreamingDetection component(job);
    int frame_number = 0;

    // Runs Darknet on same video frames twice, but treats them as separate segments.
    // Same tracks should be found in both segments since each segment processes the same frames.
    for (int segment = 0; segment < 2; segment++) {
        VideoSegmentInfo segment_info(segment, frame_number, frame_number + end_frame, 100, 100);
        component.BeginSegment(segment_info);

        MPFVideoCapture cap({"Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, {}, { }});

        int true_count = 0;
        cv::Mat frame;
        while (cap.Read(frame)) {
            if (component.ProcessFrame(frame, frame_number)) {
                true_count++;
            }
            frame_number++;
        }
        ASSERT_EQ(1, true_count);

        std::vector<MPFVideoTrack> results = component.EndSegment();
        for (int i = segment_info.start_frame; i <= segment_info.end_frame; i++) {
            ASSERT_TRUE(object_found("person", i, results));
            ASSERT_TRUE(object_found("car", i, results));
        }
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
    std::vector<MPFVideoTrack> results = component.GetDetections(job);

    ASSERT_EQ(results.size(), 2);

    ASSERT_TRUE(object_found_in_all_frames("car", results.at(0), 0, end_frame)
                || object_found_in_all_frames("car", results.at(1), 0, end_frame));

    ASSERT_TRUE(object_found_in_all_frames("person", results.at(0), 0, end_frame)
                || object_found_in_all_frames("person", results.at(1), 0, end_frame));
}



TEST(DarknetStreaming, UsePreprocessorVideoTest) {
    int end_frame = 4;
    Properties job_properties = get_yolo_tiny_config();
    job_properties.emplace("USE_PREPROCESSOR", "TRUE");

    MPFStreamingVideoJob job("Test", "../plugin/", job_properties, {});
    DarknetStreamingDetection component(job);
    int frame_number = 0;

    // Runs Darknet on same video frames twice, but treats them as separate segments.
    // Same tracks should be found in both segments since each segment processes the same frames.
    for (int segment = 0; segment < 2; segment++) {
        VideoSegmentInfo segment_info(segment, frame_number, frame_number + end_frame, 100, 100);
        component.BeginSegment(segment_info);
        MPFVideoCapture cap({"Test", "data/lp-ferrari-texas-shortened.mp4", 0, end_frame, job_properties, { }});

        int true_count = 0;
        cv::Mat frame;
        while (cap.Read(frame)) {
            if (component.ProcessFrame(frame, frame_number)) {
                true_count++;
            }
            frame_number++;
        }
        ASSERT_EQ(1, true_count);

        std::vector<MPFVideoTrack> results = component.EndSegment();

        ASSERT_EQ(results.size(), 2);

        ASSERT_TRUE(
            object_found_in_all_frames("car", results.at(0), segment_info.start_frame, segment_info.end_frame)
                || object_found_in_all_frames("car", results.at(1), segment_info.start_frame, segment_info.end_frame));

        ASSERT_TRUE(
            object_found_in_all_frames("person", results.at(0), segment_info.start_frame, segment_info.end_frame)
                || object_found_in_all_frames("person", results.at(1), segment_info.start_frame, segment_info.end_frame));
    }
}



DarknetResult CreateDetection(const std::string &object_type, float confidence, int frame_number) {
    return DarknetResult(frame_number, cv::Rect(0, 0, 10, 10), { { confidence, object_type } });
}

DarknetResult CreateDetection(const cv::Rect &location, const std::string &object_type, float confidence,
                              int frame_number) {
    return DarknetResult(frame_number, location, { { confidence, object_type } });
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
    std::vector<MPFVideoTrack> tracks = PreprocessorTracker::GetTracks({
            CreateDetection("person", 0.5, 0), CreateDetection("dog", 0.5, 0),

            CreateDetection("person", 0.6, 1), CreateDetection("dog", 0.5, 1),

            CreateDetection("person", 0.8, 2), CreateDetection("dog", 0.5, 2), CreateDetection("cat", 0.5, 2),

            CreateDetection("dog", 0.5, 3), CreateDetection("cat", 0.95, 3),

            CreateDetection("dog", 0.5, 4), CreateDetection("cat", 0.5, 4), CreateDetection("person", 0.65, 4),

            CreateDetection("dog", 0.5, 5), CreateDetection("person", 0.9, 5), CreateDetection("person", 0.3, 5)
    });

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

using class_prob_vec_t = decltype(DarknetResult::object_type_probs);

TEST(Darknet, TestPreprocessorConfidenceCalculation) {
    float p1_confidence = 0.45;
    float p2_confidence = 0.75;
    float prob_not_p1_and_not_p2 = (1 - p1_confidence) * (1 - p2_confidence);
    float prob_p1_or_p2 = 1 - prob_not_p1_and_not_p2;

    float p3_confidence = 0.25;
    float prob_not_p1_and_not_p2_and_not_p3 = (1 - p1_confidence) * (1 - p2_confidence) * (1 - p3_confidence);
    float prob_p1_or_p2_or_p3 = 1 - prob_not_p1_and_not_p2_and_not_p3;

    float d1_confidence = 0.65;


    std::vector<DarknetResult> initial_detections {
            DarknetResult(0, cv::Rect(1, 1, 1, 1), { { p1_confidence, "person" }, { d1_confidence, "dog" } }),
            DarknetResult(0, cv::Rect(1, 1, 1, 1), { { p2_confidence, "person" } })
    };

    {
        auto detections_copy = initial_detections;
        auto tracks = PreprocessorTracker::GetTracks(std::move(detections_copy));

        ASSERT_EQ(tracks.size(), 2);

        ASSERT_TRUE(has_track_with_confidence(tracks, "person", prob_p1_or_p2));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "person", prob_p1_or_p2));

        ASSERT_TRUE(has_track_with_confidence(tracks, "dog", d1_confidence));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "dog", d1_confidence));
    }

    {
        auto detections_copy = initial_detections;
        detections_copy.emplace_back(0, cv::Rect(1, 1, 1, 1), class_prob_vec_t{ { p3_confidence, "person" } });

        auto tracks = PreprocessorTracker::GetTracks(std::move(detections_copy));

        ASSERT_EQ(tracks.size(), 2);

        // Verify at P(A or B) or P(C) is the same as P(A or B or C)
        ASSERT_TRUE(has_track_with_confidence(tracks, "person", prob_p1_or_p2_or_p3));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "person", prob_p1_or_p2_or_p3));

        ASSERT_TRUE(has_track_with_confidence(tracks, "dog", d1_confidence));
        ASSERT_TRUE(has_image_location_with_confidence(tracks, "dog", d1_confidence));
    }


    {
        auto detections_copy = initial_detections;
        detections_copy.emplace_back(1, cv::Rect(1, 1, 1, 1), class_prob_vec_t{ { p3_confidence, "person" } });
        auto tracks = PreprocessorTracker::GetTracks(std::move(detections_copy));

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

    std::istringstream iss(track.frame_locations.begin()->second
                                   .detection_properties.at("CLASSIFICATION CONFIDENCE LIST"));

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
    std::vector<DarknetResult> detections {
        DarknetResult(0, cv::Rect(0, 0, 1, 1), { { .1, "dog" }, { .2, "person" }, {.3, "cat"}, {.25, "apple"} }),
        DarknetResult(0, cv::Rect(4, 4, 1, 1), { { .1, "person" }, { .25, "dog" }, {.25, "cat"}, {.1, "apple"} })
    };

    auto tracks = DefaultTracker::GetTracks(3, 0.5, std::move(detections));

    ASSERT_EQ(tracks.size(), 2);

    MPFVideoTrack track1;
    MPFVideoTrack track2;
    if (tracks.at(0).frame_locations.begin()->second.x_left_upper == 0) {
        track1 = tracks.at(0);
        track2 = tracks.at(1);
    }
    else {
        track1 = tracks.at(1);
        track2 = tracks.at(0);
    }

    ASSERT_EQ(track1.frame_locations.size(), 1);
    ASSERT_EQ(track2.frame_locations.size(), 1);

    ASSERT_FLOAT_EQ(track1.confidence, .3);
    ASSERT_EQ(track1.detection_properties.at("CLASSIFICATION"), "cat");
    ASSERT_EQ(track1.frame_locations.begin()->second.detection_properties.at("CLASSIFICATION LIST"),
              "cat; apple; person");
    ASSERT_TRUE(has_confidence_values(track1, { .3, .25, .2 }));

    ASSERT_FLOAT_EQ(track2.confidence, .25);
    ASSERT_EQ(track2.detection_properties.at("CLASSIFICATION"), "cat");
    ASSERT_EQ(track2.frame_locations.begin()->second.detection_properties.at("CLASSIFICATION LIST"),
              "cat; dog; apple");
    ASSERT_TRUE(has_confidence_values(track2, { .25, .25, .1 }));
}



TEST(Darknet, TestModelsIniParser) {
    ModelsIniParser<ModelSettings> parser;
    ModelSettings settings = parser.Init("../plugin/DarknetDetection/models")
            .RegisterPathField("network_config", &ModelSettings::network_config_file)
            .RegisterPathField("names", &ModelSettings::names_file)
            .RegisterPathField("weights", &ModelSettings::weights_file)
            .ParseIni("tiny yolo", "/opt/share/models/Darknet/");

    ASSERT_EQ(settings.network_config_file, "../plugin/DarknetDetection/models/yolov3-tiny.cfg");
    ASSERT_EQ(settings.names_file, "../plugin/DarknetDetection/models/coco.names");
    ASSERT_EQ(settings.weights_file, "../plugin/DarknetDetection/models/yolov3-tiny.weights");
}



TEST(Darknet, TestWhitelist) {
    Properties job_props = get_yolo_tiny_config();
    DarknetDetection component = init_component();

    {
        job_props["CLASS_WHITELIST_FILE"] = "data/test-whitelist.txt";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});

        std::vector<MPFImageLocation> results = component.GetDetections(job);

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

        std::vector<MPFVideoTrack> results = component.GetDetections(job);

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

    try {
        job_props["CLASS_WHITELIST_FILE"] = "data/NOTICE";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});
        component.GetDetections(job);
        FAIL() << "Expected MPFDetectionException to be thrown.";
    }
    catch (const MPFDetectionException &ex) {
        ASSERT_EQ(ex.error_code, MPF_COULD_NOT_READ_DATAFILE);
    }

    try {
        job_props["CLASS_WHITELIST_FILE"] = "FAKE_PATH";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});
        component.GetDetections(job);
        FAIL() << "Expected MPFDetectionException to be thrown.";
    }
    catch (const MPFDetectionException &ex) {
        ASSERT_EQ(ex.error_code, MPF_COULD_NOT_OPEN_DATAFILE);
    }

    try {
        job_props["CLASS_WHITELIST_FILE"] = "$THIS_ENV_VAR_SHOULD_NOT_EXIST/FAKE_PATH";
        MPFImageJob job("Test", "data/dog.jpg", job_props, {});
        component.GetDetections(job);
        FAIL() << "Expected MPFDetectionException to be thrown.";
    }
    catch (const MPFDetectionException &ex) {
        ASSERT_EQ(ex.error_code, MPF_INVALID_PROPERTY);
    }
}


TEST(Darknet, DefaultTrackerFiltersOnIntersectionRatio) {
    std::vector<DarknetResult> detections {
            CreateDetection({5, 5, 20, 20}, "object", 0.5, 0),
            CreateDetection({8, 8, 20, 20}, "object", 0.5, 1),
            CreateDetection({20, 20, 20, 20}, "object", 0.5, 2)
    };

    auto tracks = DefaultTracker::GetTracks(5, 0.5, std::move(detections));
    ASSERT_EQ(2, tracks.size());

    MPFVideoTrack track1;
    MPFVideoTrack track2;
    if (tracks.at(0).start_frame == 0) {
        track1 = tracks.at(0);
        track2 = tracks.at(1);
    }
    else {
        track1 = tracks.at(1);
        track2 = tracks.at(0);
    }

    ASSERT_EQ(0, track1.start_frame);
    ASSERT_EQ(1, track1.stop_frame);
    ASSERT_EQ(2, track1.frame_locations.size());
    ASSERT_TRUE(object_found("object", 0, track1));
    ASSERT_TRUE(object_found("object", 1, track1));

    ASSERT_EQ(2, track2.start_frame);
    ASSERT_EQ(2, track2.stop_frame);
    ASSERT_EQ(1, track2.frame_locations.size());
    ASSERT_TRUE(object_found("object", 2, track2));
}


TEST(Darknet, DefaultTrackerIgnoresOverlapWhenOverlapRatioNotPositive) {
    for (double overlap_ratio : { 0.0, -0.5, -1.0 }) {

        std::vector<DarknetResult> detections {
                CreateDetection(cv::Rect(0, 0, 1, 1), "object", 0.5, 0),
                CreateDetection(cv::Rect(5, 5, 1, 1), "object", 0.5, 1),
                CreateDetection(cv::Rect(0, 0, 1, 1), "other", 0.5, 1)
        };

        auto tracks = DefaultTracker::GetTracks(5, overlap_ratio, std::move(detections));
        ASSERT_EQ(2, tracks.size());

        MPFVideoTrack object_track;
        MPFVideoTrack other_track;
        if (tracks.at(0).frame_locations.size() == 2) {
            object_track = tracks.at(0);
            other_track = tracks.at(1);
        }
        else {
            object_track = tracks.at(1);
            other_track = tracks.at(0);
        }

        ASSERT_EQ(2, object_track.frame_locations.size());
        ASSERT_TRUE(object_found("object", 0, object_track));
        ASSERT_TRUE(object_found("object", 1, object_track));
        ASSERT_EQ(0, object_track.start_frame);
        ASSERT_EQ(1, object_track.stop_frame);

        ASSERT_EQ(1, other_track.frame_locations.size());
        ASSERT_TRUE(object_found("other", 1, other_track));
        ASSERT_EQ(1, other_track.start_frame);
        ASSERT_EQ(1, other_track.stop_frame);
    }
}



TEST(Darknet, DefaultTrackerOnlyCombinesExactMatchWhenOverlapIsOne) {
    std::vector<DarknetResult> detections {
            CreateDetection({5, 5, 5, 6}, "object", 0.5, 0),
            CreateDetection({5, 5, 5, 5}, "object", 0.5, 1),
            CreateDetection({5, 5, 5, 5}, "other", 0.5, 1),
            CreateDetection({5, 5, 5, 5}, "other", 0.5, 2)
    };

    auto tracks = DefaultTracker::GetTracks(5, 1, std::move(detections));
    ASSERT_EQ(3, tracks.size());

    MPFVideoTrack object_track1;
    MPFVideoTrack object_track2;
    MPFVideoTrack other_track;
    for (auto& track : tracks) {
        if (track.start_frame == 0 && track.stop_frame == 0) {
            object_track1 = track;
        }
        else if (track.start_frame == 1 && track.stop_frame == 1) {
            object_track2 = track;
        }
        else {
            other_track = track;
        }
    }

    ASSERT_EQ(1, object_track1.frame_locations.size());
    ASSERT_TRUE(object_found("object", 0, object_track1));

    ASSERT_EQ(1, object_track2.frame_locations.size());
    ASSERT_TRUE(object_found("object", 1, object_track2));

    ASSERT_EQ(1, other_track.start_frame);
    ASSERT_EQ(2, other_track.stop_frame);
    ASSERT_EQ(2, other_track.frame_locations.size());
    ASSERT_TRUE(object_found("other", 1, other_track));
    ASSERT_TRUE(object_found("other", 2, other_track));
}


TEST(Darknet, DefaultTrackerDoesNotCombineWhenOverlapIsGreaterThanOne) {
    std::vector<DarknetResult> detections {
            CreateDetection({5, 5, 5, 6}, "object", 0.5, 0),
            CreateDetection({5, 5, 5, 5}, "object", 0.5, 1),
            CreateDetection({5, 5, 5, 5}, "other", 0.5, 1),
            CreateDetection({5, 5, 5, 5}, "other", 0.5, 2)
    };


    auto tracks = DefaultTracker::GetTracks(5, 1.1, std::move(detections));
    ASSERT_EQ(4, tracks.size());
    ASSERT_TRUE(object_found("object", 0, tracks));
    ASSERT_TRUE(object_found("object", 1, tracks));
    ASSERT_TRUE(object_found("other", 1, tracks));
    ASSERT_TRUE(object_found("other", 2, tracks));

    for (const auto &track : tracks) {
        ASSERT_EQ(1, track.frame_locations.size());
        ASSERT_EQ(track.start_frame, track.stop_frame);
    }
}



TEST(Darknet, DefaultTrackerDoesNotCombineDetectionsWhenNonContiguousFrames) {
    std::vector<DarknetResult> detections {
            CreateDetection(cv::Rect(0, 0, 1, 1), "object", 0.5, 0),
            CreateDetection(cv::Rect(0, 0, 1, 1), "object", 0.5, 1),

            CreateDetection(cv::Rect(0, 0, 1, 1), "object", 0.5, 3),
            CreateDetection(cv::Rect(0, 0, 1, 1), "object", 0.5, 4),
            CreateDetection(cv::Rect(0, 0, 1, 1), "object", 0.5, 5)
    };

    auto tracks = DefaultTracker::GetTracks(5, 0, std::move(detections));
    ASSERT_EQ(2, tracks.size());
    MPFVideoTrack track0to1;
    MPFVideoTrack track3to5;
    if (tracks.at(0).start_frame == 0) {
        track0to1 = tracks.at(0);
        track3to5 = tracks.at(1);
    }
    else {
        track0to1 = tracks.at(1);
        track3to5 = tracks.at(0);
    }

    ASSERT_EQ(0, track0to1.start_frame);
    ASSERT_EQ(1, track0to1.stop_frame);
    ASSERT_EQ(2, track0to1.frame_locations.size());
    ASSERT_TRUE(object_found("object", 0, track0to1));
    ASSERT_TRUE(object_found("object", 1, track0to1));

    ASSERT_EQ(3, track3to5.start_frame);
    ASSERT_EQ(5, track3to5.stop_frame);
    ASSERT_EQ(3, track3to5.frame_locations.size());
    ASSERT_TRUE(object_found("object", 3, track3to5));
    ASSERT_TRUE(object_found("object", 4, track3to5));
    ASSERT_TRUE(object_found("object", 5, track3to5));
}


