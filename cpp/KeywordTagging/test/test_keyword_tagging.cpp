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

#include <gtest/gtest.h>
#include "KeywordTagging.h"

using namespace MPF::COMPONENT;

void setAlgorithmProperties(Properties &algorithm_properties, const std::map<std::string, std::string> &custom) {

    algorithm_properties["TAGGING_FILE"] = "config/test-text-tags-foreign.json";

    for (auto const &x : custom) {
        algorithm_properties[x.first] = x.second;
    }
}

MPFGenericJob createGenericJob(const std::string &uri, const std::map<std::string, std::string> &custom = {}) {

    Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("Tagger_test");
    setAlgorithmProperties(algorithm_properties, custom);

    MPFGenericJob job(job_name, uri, algorithm_properties, media_properties);
    return job;
}

bool containsProp(const std::string &exp_text, const std::vector<MPFGenericTrack> &tracks,
                  const std::string &property, int index = -1) {
    if (index != -1) {
        if (tracks[index].detection_properties.count(property) == 0) {
            return true;
        }

        std::string text = tracks[index].detection_properties.at(property);


        return text.find(exp_text) != std::string::npos;
    }

    for (int i = 0; i < tracks.size(); i++) {
        if (tracks[i].detection_properties.count(property) == 0) {
            continue;
        }

        std::string text = tracks[i].detection_properties.at(property);


        if (text.find(exp_text) != std::string::npos)
            return true;
    }

    return false;
}

void assertInText(const std::string &image_path, const std::string &expected_value,
                  const std::vector<MPFGenericTrack> &tracks, const std::string &prop, int index = -1) {
    ASSERT_TRUE(containsProp(expected_value, tracks, prop, index))
                                << "Expected tagger to detect " << prop << " \"" << expected_value << "\" in " << image_path;
}

void assertNotInText(const std::string &file_path, const std::string &expected_text,
                     const std::vector<MPFGenericTrack> &tracks, const std::string &prop, int index = -1) {
    ASSERT_FALSE(containsProp(expected_text, tracks, prop, index))
                                << "Expected tagger to NOT detect "<< prop << " \""  << expected_text << "\" in "
                                << file_path;
}

void runKeywordTagging(const std::string &uri_path, KeywordTagging &tagger,
                       std::vector<MPFGenericTrack> &text_tags,
                       const std::map<std::string, std::string> &custom = {}) {
    MPFGenericJob job = createGenericJob(uri_path, custom);
    text_tags = tagger.GetDetections(job);

    ASSERT_FALSE(text_tags.empty());
}

TEST(KEYWORDTAGGING, TaggingTest) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

    std::vector<MPFImageLocation> result;

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    // Test basic tagging
    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/text-demo.txt", tagger, results, custom_properties));
    assertNotInText("data/text-demo.txt", "personal", results, "TAGS");
    ASSERT_TRUE(results[0].detection_properties.at("TAGS").length() == 0);
    results.clear();

    // Test escaped backslash text tagging.
    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/test-backslash.txt", tagger, results, custom_properties));
    assertInText("data/test-backslash.txt", "backslash; personal", results, "TAGS");
    assertInText("data/test-backslash.txt", "TEXT; \\", results, "TRIGGER_WORDS");
    assertInText("data/test-backslash.txt", "7-10; 0, 12, 15, 16, 18, 19", results, "TRIGGER_WORDS_OFFSET");

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, MulitpleTagsTest) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-keyword.txt", tagger, results, custom_properties));
    assertInText("data/tags-keyword.txt", "Passenger Passport", results, "TEXT");
    assertInText("data/tags-keyword.txt", "identity document; travel", results, "TAGS");
    assertInText("data/tags-keyword.txt", "Passenger; Passport", results, "TRIGGER_WORDS");
    assertInText("data/tags-keyword.txt", "0-8; 10-17", results, "TRIGGER_WORDS_OFFSET");
    results.clear();

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-regex.txt", tagger, results, custom_properties));
    assertInText("data/tags-regex.txt", "case-insensitive-tag; financial; personal", results, "TAGS");
    assertInText("data/tags-regex.txt", "122-123-1234; financ", results, "TRIGGER_WORDS");
    assertInText("data/tags-regex.txt", "17-28; 0-5", results, "TRIGGER_WORDS_OFFSET");
    results.clear();

    // Test multiple text tagging w/ delimiter tag.
    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-regex-delimiter.txt", tagger, results, custom_properties));
    assertInText("data/tags-regex-delimiter.txt", "case-insensitive-tag; delimiter-test; financial; personal",
                  results, "TAGS");
    assertInText("data/tags-regex-delimiter.txt", "122-123-1234; a[[;] ]b; financ", results, "TRIGGER_WORDS");
    assertInText("data/tags-regex-delimiter.txt", "22-33; 15-20; 0-5", results, "TRIGGER_WORDS_OFFSET");

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, FullSearch) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-keywordregex.txt", tagger, results, custom_properties));
    assertInText("data/tags-keywordregex.txt", "case-insensitive-tag; case-sensitive-tag; financial; personal; vehicle",
                 results, "TAGS");
    assertInText("data/tags-keywordregex.txt", "01/01/20; Financ; Text; Vehicle", results, "TRIGGER_WORDS");
    assertInText("data/tags-keywordregex.txt", "20-27; 37-42; 10-13, 15-18; 29-35", results, "TRIGGER_WORDS_OFFSET");
    results.clear();

    // With full regex search disabled, number of reported triggers and offsets will decrease.
    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-keywordregex.txt", tagger, results, custom_properties_disabled));
    assertInText("data/tags-keywordregex.txt", "case-insensitive-tag; case-sensitive-tag; financial; personal; vehicle",
                  results, "TAGS");
    assertInText("data/tags-keywordregex.txt", "01/01/20; Financ; Vehicle", results, "TRIGGER_WORDS");
    assertInText("data/tags-keywordregex.txt", "20-27; 37-42; 29-35", results, "TRIGGER_WORDS_OFFSET");

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, LanguageTest) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/eng-bul.txt", tagger, results, custom_properties));
    assertInText("data/eng-bul.txt", "foreign-text", results, "TAGS", 0);
    assertInText("data/eng-bul.txt", "свободни", results, "TRIGGER_WORDS", 0);
    assertInText("data/eng-bul.txt", "106-113", results, "TRIGGER_WORDS_OFFSET", 0);
    assertInText("data/eng-bul.txt", "Всички хора се раждат свободни", results, "TEXT", 0);

    ASSERT_TRUE(tagger.Close());
}


TEST(KEYWORDTAGGING, MissingPropertyToProcessTest) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    MPFImageLocation location(1, 2, 3, 4, 5,
                              {{"SOME_PROP_1", "SOME_VAL_1"},
                               {"SOME_PROP_2", "SOME_VAL_2"}}); // no TEXT or TRANSCRIPT provided
    MPFImageJob job("JOB NAME", "/some/path", location, {}, {});

    std::vector<MPFImageLocation> results = tagger.GetDetections(job);

    // detection is unchanged
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(location.x_left_upper, results.at(0).x_left_upper);
    ASSERT_EQ(location.y_left_upper, results.at(0).y_left_upper);
    ASSERT_EQ(location.width, results.at(0).width);
    ASSERT_EQ(location.height, results.at(0).height);
    ASSERT_EQ(location.confidence, results.at(0).confidence);
    ASSERT_EQ(location.detection_properties, results.at(0).detection_properties);

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, MissingTextToProcessTest) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    MPFImageLocation location(1, 2, 3, 4, 5,
                              {{"TEXT", ""},
                               {"SOME_PROP_2", "SOME_VAL_2"}});
    MPFImageJob job("JOB NAME", "/some/path", location, {}, {});

    std::vector<MPFImageLocation> results = tagger.GetDetections(job);

    // detection is unchanged
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(location.x_left_upper, results.at(0).x_left_upper);
    ASSERT_EQ(location.y_left_upper, results.at(0).y_left_upper);
    ASSERT_EQ(location.width, results.at(0).width);
    ASSERT_EQ(location.height, results.at(0).height);
    ASSERT_EQ(location.confidence, results.at(0).confidence);
    ASSERT_EQ(location.detection_properties, results.at(0).detection_properties);

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, ProcessFirstPropertyOnlyTest) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    {
        MPFImageLocation location(1, 2, 3, 4, 5,
                                  {{"TRANSCRIPT", "cash"},
                                   {"TEXT", "car"}}); // TEXT should be considered before TRANSCRIPT
        MPFImageJob job("JOB NAME", "/some/path", location, {}, {});

        std::vector<MPFImageLocation> results = tagger.GetDetections(job);
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(location.x_left_upper, results.at(0).x_left_upper);
        ASSERT_EQ(location.y_left_upper, results.at(0).y_left_upper);
        ASSERT_EQ(location.width, results.at(0).width);
        ASSERT_EQ(location.height, results.at(0).height);
        ASSERT_EQ(location.confidence, results.at(0).confidence);

        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("cash", props["TRANSCRIPT"]);
        ASSERT_EQ("car", props["TEXT"]);
        ASSERT_EQ("vehicle", props["TAGS"]);
        ASSERT_EQ("car", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-2", props["TRIGGER_WORDS_OFFSET"]);
    }

    {
        MPFAudioTrack track(1000, 5000, 0.9,
                            {{"TRANSCRIPT", "cash"},
                             {"TEXT", "car"}});
        MPFAudioJob job("JOB NAME", "/some/path", 100, 100000, track,
                        { { "FEED_FORWARD_PROP_TO_PROCESS", "TRANSCRIPT,TEXT" } }, {}); // TRANSCRIPT should be considered before TEXT

        std::vector<MPFAudioTrack> results = tagger.GetDetections(job);
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(track.start_time, results.at(0).start_time);
        ASSERT_EQ(track.stop_time, results.at(0).stop_time);
        ASSERT_EQ(track.confidence, results.at(0).confidence);

        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("cash", props["TRANSCRIPT"]);
        ASSERT_EQ("car", props["TEXT"]);
        ASSERT_EQ("financial", props["TAGS"]);
        ASSERT_EQ("cash", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-3", props["TRIGGER_WORDS_OFFSET"]);
    }

    {
        MPFGenericTrack track(0.9,
                              {{"BAR", "cash"},
                               {"FOO", "car"}});
        MPFGenericJob job("JOB NAME", "/some/path", track,
                          { { "FEED_FORWARD_PROP_TO_PROCESS", "FOO,BAR" } }, {}); // user-specified properties

        std::vector<MPFGenericTrack> results = tagger.GetDetections(job);
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(track.confidence, results.at(0).confidence);

        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("cash", props["BAR"]);
        ASSERT_EQ("car", props["FOO"]);
        ASSERT_EQ("vehicle", props["TAGS"]);
        ASSERT_EQ("car", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-2", props["TRIGGER_WORDS_OFFSET"]);
    }

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, ProcessTrackAndDetectionProperties) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    {
        MPFImageLocation location1(1, 2, 3, 4, 5,
                                   {{"TEXT", "car"},
                                    {"SOME_PROP_1", "SOME_VAL_1"}});

        MPFImageLocation location2(11, 12, 13, 14, 15,
                                   {{"TEXT", "username"},
                                    {"SOME_PROP_2", "SOME_VAL_2"}});

        MPFVideoTrack track(10, 12, 0.5,
                            {{"TEXT", "airport"},
                             {"SOME_PROP_3", "SOME_VAL_3"}});
        track.frame_locations.emplace(10, location1);
        track.frame_locations.emplace(12, location2);

        MPFVideoJob job("JOB NAME", "/some/path", 0, 100, track, {}, {});

        std::vector<MPFVideoTrack> results = tagger.GetDetections(job);

        ASSERT_EQ(1, results.size());
        ASSERT_EQ(track.start_frame, results.at(0).start_frame);
        ASSERT_EQ(track.stop_frame, results.at(0).stop_frame);
        ASSERT_EQ(track.confidence, results.at(0).confidence);
        ASSERT_EQ(2, results.at(0).frame_locations.size());

        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("airport", props["TEXT"]);
        ASSERT_EQ("SOME_VAL_3", props["SOME_PROP_3"]);
        ASSERT_EQ("travel", props["TAGS"]);
        ASSERT_EQ("airport", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-6", props["TRIGGER_WORDS_OFFSET"]);

        MPFImageLocation location = results.at(0).frame_locations.at(10);
        ASSERT_EQ(location1.x_left_upper, location.x_left_upper);
        ASSERT_EQ(location1.y_left_upper, location.y_left_upper);
        ASSERT_EQ(location1.width, location.width);
        ASSERT_EQ(location1.height, location.height);
        ASSERT_EQ(location1.confidence, location.confidence);

        props = location.detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("SOME_VAL_1", props["SOME_PROP_1"]);
        ASSERT_EQ("car", props["TEXT"]);
        ASSERT_EQ("vehicle", props["TAGS"]);
        ASSERT_EQ("car", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-2", props["TRIGGER_WORDS_OFFSET"]);

        location = results.at(0).frame_locations.at(12);
        ASSERT_EQ(location2.x_left_upper, location.x_left_upper);
        ASSERT_EQ(location2.y_left_upper, location.y_left_upper);
        ASSERT_EQ(location2.width, location.width);
        ASSERT_EQ(location2.height, location.height);
        ASSERT_EQ(location2.confidence, location.confidence);

        props = location.detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("SOME_VAL_2", props["SOME_PROP_2"]);
        ASSERT_EQ("username", props["TEXT"]);
        ASSERT_EQ("personal", props["TAGS"]);
        ASSERT_EQ("username", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-7", props["TRIGGER_WORDS_OFFSET"]);
    }

    {
        MPFImageLocation location1(1, 2, 3, 4, 5,
                                   {{"SOME_PROP_1", "SOME_VAL_1"}}); // no property to process

        MPFImageLocation location2(11, 12, 13, 14, 15,
                                   {{"TRANSCRIPT", "username"}});

        MPFVideoTrack track(10, 12, 0.5,
                            {{"SOME_PROP_3", "SOME_VAL_3"}}); // no property to process
        track.frame_locations.emplace(10, location1);
        track.frame_locations.emplace(12, location2);

        MPFVideoJob job("JOB NAME", "/some/path", 0, 100, track, {}, {});

        std::vector<MPFVideoTrack> results = tagger.GetDetections(job);

        // track fields are unchanged, except for the content of frame_locations
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(track.start_frame, results.at(0).start_frame);
        ASSERT_EQ(track.stop_frame, results.at(0).stop_frame);
        ASSERT_EQ(track.confidence, results.at(0).confidence);
        ASSERT_EQ(track.detection_properties, results.at(0).detection_properties);
        ASSERT_EQ(2, results.at(0).frame_locations.size());

        // detection is unchanged
        MPFImageLocation location = results.at(0).frame_locations.at(10);
        ASSERT_EQ(location1.x_left_upper, location.x_left_upper);
        ASSERT_EQ(location1.y_left_upper, location.y_left_upper);
        ASSERT_EQ(location1.width, location.width);
        ASSERT_EQ(location1.height, location.height);
        ASSERT_EQ(location1.confidence, location.confidence);
        ASSERT_EQ(location1.detection_properties, location.detection_properties);

        location = results.at(0).frame_locations.at(12);
        ASSERT_EQ(location2.x_left_upper, location.x_left_upper);
        ASSERT_EQ(location2.y_left_upper, location.y_left_upper);
        ASSERT_EQ(location2.width, location.width);
        ASSERT_EQ(location2.height, location.height);
        ASSERT_EQ(location2.confidence, location.confidence);

        Properties props = location.detection_properties;
        ASSERT_EQ(4, props.size());
        ASSERT_EQ("username", props["TRANSCRIPT"]);
        ASSERT_EQ("personal", props["TAGS"]);
        ASSERT_EQ("username", props["TRIGGER_WORDS"]);
        ASSERT_EQ("0-7", props["TRIGGER_WORDS_OFFSET"]);
    }

    ASSERT_TRUE(tagger.Close());
}