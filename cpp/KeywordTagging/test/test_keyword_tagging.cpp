/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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
#include <log4cxx/basicconfigurator.h>

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
            return false;
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

void assertInText(const std::string &file_path, const std::string &expected_value,
                  const std::vector<MPFGenericTrack> &tracks, const std::string &prop, int index = -1) {
    ASSERT_TRUE(containsProp(expected_value, tracks, prop, index))
                                << "Expected tagger to detect " << prop << " \"" << expected_value << "\" in " << file_path;
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

bool init_logging() {
    log4cxx::BasicConfigurator::configure();
    return true;
}
bool logging_initialized = init_logging();


TEST(KEYWORDTAGGING, TaggingTest) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
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
    assertInText("data/test-backslash.txt", "\\", results, "TEXT BACKSLASH TRIGGER WORDS");
    assertInText("data/test-backslash.txt", "0, 12, 15, 16, 18, 19, 20, 21", results, "TEXT BACKSLASH TRIGGER WORDS OFFSET");
    assertInText("data/test-backslash.txt", "TEXT", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/test-backslash.txt", "7-10", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, MulitpleTagsTest) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-keyword.txt", tagger, results, custom_properties));
    assertInText("data/tags-keyword.txt", "Passenger Passport", results, "TEXT");
    assertInText("data/tags-keyword.txt", "identity document; travel", results, "TAGS");
    assertInText("data/tags-keyword.txt", "Passport", results, "TEXT IDENTITY DOCUMENT TRIGGER WORDS");
    assertInText("data/tags-keyword.txt", "10-17", results, "TEXT IDENTITY DOCUMENT TRIGGER WORDS OFFSET");
    assertInText("data/tags-keyword.txt", "Passenger", results, "TEXT TRAVEL TRIGGER WORDS");
    assertInText("data/tags-keyword.txt", "0-8", results, "TEXT TRAVEL TRIGGER WORDS OFFSET");
    results.clear();

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-regex.txt", tagger, results, custom_properties));
    assertInText("data/tags-regex.txt", "case-insensitive-tag; financial; personal", results, "TAGS");
    assertInText("data/tags-regex.txt", "financ", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS");
    assertInText("data/tags-regex.txt", "0-5", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS OFFSET");
    assertInText("data/tags-regex.txt", "financ", results, "TEXT FINANCIAL TRIGGER WORDS");
    assertInText("data/tags-regex.txt", "0-5", results, "TEXT FINANCIAL TRIGGER WORDS OFFSET");
    assertInText("data/tags-regex.txt", "122-123-1234", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/tags-regex.txt", "17-28", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");
    results.clear();

    // Test multiple text tagging w/ delimiter tag.
    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-regex-delimiter.txt", tagger, results, custom_properties));
    assertInText("data/tags-regex-delimiter.txt", "case-insensitive-tag; delimiter-test; financial; personal",
                  results, "TAGS");
    assertInText("data/tags-regex-delimiter.txt", "financ", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS");
    assertInText("data/tags-regex-delimiter.txt", "0-5", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS OFFSET");
    assertInText("data/tags-regex-delimiter.txt", "a[[;] ]b", results, "TEXT DELIMITER-TEST TRIGGER WORDS");
    assertInText("data/tags-regex-delimiter.txt", "15-20", results, "TEXT DELIMITER-TEST TRIGGER WORDS OFFSET");
    assertInText("data/tags-regex-delimiter.txt", "financ", results, "TEXT FINANCIAL TRIGGER WORDS");
    assertInText("data/tags-regex-delimiter.txt", "0-5", results, "TEXT FINANCIAL TRIGGER WORDS OFFSET");
    assertInText("data/tags-regex-delimiter.txt", "122-123-1234", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/tags-regex-delimiter.txt", "22-33", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");

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
    assertInText("data/tags-keywordregex.txt", "Financ", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "37-42", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS OFFSET");
    assertInText("data/tags-keywordregex.txt", "Financ", results, "TEXT CASE-SENSITIVE-TAG TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "37-42", results, "TEXT CASE-SENSITIVE-TAG TRIGGER WORDS OFFSET");
    assertInText("data/tags-keywordregex.txt", "Financ", results, "TEXT FINANCIAL TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "37-42", results, "TEXT FINANCIAL TRIGGER WORDS OFFSET");
    assertInText("data/tags-keywordregex.txt", "01/01/20; Text", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "20-27; 10-13, 15-18", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");
    assertInText("data/tags-keywordregex.txt", "Vehicle", results, "TEXT VEHICLE TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "29-35", results, "TEXT VEHICLE TRIGGER WORDS OFFSET");
    results.clear();

    // With full regex search disabled, number of reported triggers and offsets will decrease.
    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/tags-keywordregex.txt", tagger, results, custom_properties_disabled));
    assertInText("data/tags-keywordregex.txt", "case-insensitive-tag; case-sensitive-tag; financial; personal; vehicle",
                  results, "TAGS");
    assertInText("data/tags-keywordregex.txt", "Financ", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "37-42", results, "TEXT CASE-INSENSITIVE-TAG TRIGGER WORDS OFFSET");
        assertInText("data/tags-keywordregex.txt", "Financ", results, "TEXT CASE-SENSITIVE-TAG TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "37-42", results, "TEXT CASE-SENSITIVE-TAG TRIGGER WORDS OFFSET");
        assertInText("data/tags-keywordregex.txt", "Financ", results, "TEXT FINANCIAL TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "37-42", results, "TEXT FINANCIAL TRIGGER WORDS OFFSET");
        assertInText("data/tags-keywordregex.txt", "01/01/20", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "20-27", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");
        assertInText("data/tags-keywordregex.txt", "Vehicle", results, "TEXT VEHICLE TRIGGER WORDS");
    assertInText("data/tags-keywordregex.txt", "29-35", results, "TEXT VEHICLE TRIGGER WORDS OFFSET");

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, LanguageTest) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/eng-bul.txt", tagger, results, custom_properties));
    assertInText("data/eng-bul.txt", "foreign-text", results, "TAGS");
    assertInText("data/eng-bul.txt", "свободни", results, "TEXT FOREIGN-TEXT TRIGGER WORDS");
    assertInText("data/eng-bul.txt", "106-113", results, "TEXT FOREIGN-TEXT TRIGGER WORDS OFFSET");
    assertInText("data/eng-bul.txt", "Всички хора се раждат свободни", results, "TEXT");

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

    {
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
    }

    {
        MPFImageLocation location(1, 2, 3, 4, 5,
                                  {{"TEXT", ""}, {"TRANSCRIPT", "   "},
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
    }

    ASSERT_TRUE(tagger.Close());
}


TEST(KEYWORDTAGGING, ProcessAllProperties) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    {
        MPFImageLocation location(1, 2, 3, 4, 5,
                                  {{"TRANSLATION", "cash"},
                                   {"TEXT", "car"}});
        MPFImageJob job("JOB NAME", "/some/path", location, {}, {});

        std::vector<MPFImageLocation> results = tagger.GetDetections(job);
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(location.x_left_upper, results.at(0).x_left_upper);
        ASSERT_EQ(location.y_left_upper, results.at(0).y_left_upper);
        ASSERT_EQ(location.width, results.at(0).width);
        ASSERT_EQ(location.height, results.at(0).height);
        ASSERT_EQ(location.confidence, results.at(0).confidence);

        // default FEED_FORWARD_PROP_TO_PROCESS is used (TEXT, TRANSCRIPT) so tagging should run only on TEXT
        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(5, props.size());
        ASSERT_EQ("cash", props["TRANSLATION"]);
        ASSERT_EQ("car", props["TEXT"]);
        ASSERT_EQ("vehicle", props["TAGS"]);
        ASSERT_EQ("car", props["TEXT VEHICLE TRIGGER WORDS"]);
        ASSERT_EQ("0-2", props["TEXT VEHICLE TRIGGER WORDS OFFSET"]);
    }

    {
        MPFAudioTrack track(1000, 5000, 0.9,
                            {{"TRANSLATION", "cash"},
                             {"TEXT", "car"}});
        MPFAudioJob job("JOB NAME", "/some/path", 100, 100000, track,
                        { { "FEED_FORWARD_PROP_TO_PROCESS", "TRANSLATION, TEXT" } }, {});

        std::vector<MPFAudioTrack> results = tagger.GetDetections(job);
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(track.start_time, results.at(0).start_time);
        ASSERT_EQ(track.stop_time, results.at(0).stop_time);
        ASSERT_EQ(track.confidence, results.at(0).confidence);

        // TEXT AND TRANSLATION specified as props to process so tagging should run on both
        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(7, props.size());
        ASSERT_EQ("cash", props["TRANSLATION"]);
        ASSERT_EQ("car", props["TEXT"]);
        ASSERT_EQ("financial; vehicle", props["TAGS"]); // tags added in alphabetical order
        ASSERT_EQ("cash", props["TRANSLATION FINANCIAL TRIGGER WORDS"]);
        ASSERT_EQ("0-3", props["TRANSLATION FINANCIAL TRIGGER WORDS OFFSET"]);
        ASSERT_EQ("car", props["TEXT VEHICLE TRIGGER WORDS"]);
        ASSERT_EQ("0-2", props["TEXT VEHICLE TRIGGER WORDS OFFSET"]);
    }

    {
        MPFGenericTrack track(0.9,
                              {{"FOO", "car"},
                               {"BAR", "cash"}});
        MPFGenericJob job("JOB NAME", "/some/path", track,
                          { { "FEED_FORWARD_PROP_TO_PROCESS", "FOO,BAR" } }, {}); // user-specified properties

        std::vector<MPFGenericTrack> results = tagger.GetDetections(job);
        ASSERT_EQ(1, results.size());
        ASSERT_EQ(track.confidence, results.at(0).confidence);

        // should run tagging on both foo and bar
        Properties props = results.at(0).detection_properties;
        ASSERT_EQ(7, props.size());
        ASSERT_EQ("cash", props["BAR"]);
        ASSERT_EQ("car", props["FOO"]);
        ASSERT_EQ("financial; vehicle", props["TAGS"]); // tags added in alphabetical order
        ASSERT_EQ("car", props["FOO VEHICLE TRIGGER WORDS"]);
        ASSERT_EQ("0-2", props["FOO VEHICLE TRIGGER WORDS OFFSET"]);
        ASSERT_EQ("cash", props["BAR FINANCIAL TRIGGER WORDS"]);
        ASSERT_EQ("0-3", props["BAR FINANCIAL TRIGGER WORDS OFFSET"]);
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
        ASSERT_EQ("airport", props["TEXT TRAVEL TRIGGER WORDS"]);
        ASSERT_EQ("0-6", props["TEXT TRAVEL TRIGGER WORDS OFFSET"]);

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
        ASSERT_EQ("car", props["TEXT VEHICLE TRIGGER WORDS"]);
        ASSERT_EQ("0-2", props["TEXT VEHICLE TRIGGER WORDS OFFSET"]);

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
        ASSERT_EQ("username", props["TEXT PERSONAL TRIGGER WORDS"]);
        ASSERT_EQ("0-7", props["TEXT PERSONAL TRIGGER WORDS OFFSET"]);
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
        ASSERT_EQ("username", props["TRANSCRIPT PERSONAL TRIGGER WORDS"]);
        ASSERT_EQ("0-7", props["TRANSCRIPT PERSONAL TRIGGER WORDS OFFSET"]);
    }

    ASSERT_TRUE(tagger.Close());
}


TEST(KEYWORDTAGGING, ProcessRepeatTags) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    MPFImageLocation location(1, 2, 3, 4, 5,
                              {{"TEXT", "cash-car-suv"},
                               {"OTHER TEXT", "car-cash-suv"},
                               {"MORE TEXT", "cash cash"},
                               {"BLANK TEXT", " "}});
    MPFImageJob job("JOB NAME", "/some/path", location,
                      { { "FEED_FORWARD_PROP_TO_PROCESS", "TEXT, OTHER TEXT, MORE TEXT, BLANK TEXT" } }, {});

    std::vector<MPFImageLocation> results = tagger.GetDetections(job);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(location.x_left_upper, results.at(0).x_left_upper);
    ASSERT_EQ(location.y_left_upper, results.at(0).y_left_upper);
    ASSERT_EQ(location.width, results.at(0).width);
    ASSERT_EQ(location.height, results.at(0).height);
    ASSERT_EQ(location.confidence, results.at(0).confidence);

    Properties props = results.at(0).detection_properties;
    ASSERT_EQ(15, props.size());

    ASSERT_EQ("cash-car-suv", props["TEXT"]);
    ASSERT_EQ("car-cash-suv", props["OTHER TEXT"]);
    ASSERT_EQ("cash cash", props["MORE TEXT"]);
    ASSERT_EQ(" ", props["BLANK TEXT"]);

    ASSERT_EQ("financial; vehicle", props["TAGS"]); // tags added in alphabetical order

    ASSERT_EQ("cash", props["TEXT FINANCIAL TRIGGER WORDS"]); // words added in alphabetical order
    ASSERT_EQ("0-3", props["TEXT FINANCIAL TRIGGER WORDS OFFSET"]); // offsets line up with words
    ASSERT_EQ("car; suv", props["TEXT VEHICLE TRIGGER WORDS"]);
    ASSERT_EQ("5-7; 9-11", props["TEXT VEHICLE TRIGGER WORDS OFFSET"]);

    ASSERT_EQ("cash", props["OTHER TEXT FINANCIAL TRIGGER WORDS"]); // words added in alphabetical order
    ASSERT_EQ("4-7", props["OTHER TEXT FINANCIAL TRIGGER WORDS OFFSET"]); // offsets line up with words
    ASSERT_EQ("car; suv", props["OTHER TEXT VEHICLE TRIGGER WORDS"]);
    ASSERT_EQ("0-2; 9-11", props["OTHER TEXT VEHICLE TRIGGER WORDS OFFSET"]);

    ASSERT_EQ("cash", props["MORE TEXT FINANCIAL TRIGGER WORDS"]);
    ASSERT_EQ("0-3, 5-8", props["MORE TEXT FINANCIAL TRIGGER WORDS OFFSET"]); // offsets are in ascending order

    // "BLANK TEXT TRIGGER WORDS" and "BLANK TEXT TRIGGER WORDS OFFSET" are omitted since "BLANK TEXT"
    // is only whitespace.

    ASSERT_TRUE(tagger.Close());
}

TEST(KEYWORDTAGGING, FeedForwardTags) {
    KeywordTagging tagger;
    tagger.SetRunDirectory("../plugin");
    ASSERT_TRUE(tagger.Init());

    MPFGenericTrack track(0.9,
                          {{"TAGS", "FeedForwardTag"},
                          {"BAR", "cash"}});
    MPFGenericJob job("JOB NAME", "/some/path", track,
                      { { "FEED_FORWARD_PROP_TO_PROCESS", "FOO,BAR" } }, {});

    std::vector<MPFGenericTrack> results = tagger.GetDetections(job);
    ASSERT_EQ(1, results.size());
    ASSERT_EQ(track.confidence, results.at(0).confidence);

    Properties props = results.at(0).detection_properties;
    ASSERT_EQ(4, props.size());
    ASSERT_EQ("feedforwardtag; financial", props["TAGS"]);
}


TEST(KEYWORDTAGGING, NewLines) {
    KeywordTagging tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/test-newlines.txt", tagger, results, custom_properties));
    assertInText("data/test-newlines.txt", "identity document", results, "TAGS");
    assertInText("data/test-newlines.txt", "address", results, "TEXT IDENTITY DOCUMENT TRIGGER WORDS");
    assertInText("data/test-newlines.txt", "37-43", results, "TEXT IDENTITY DOCUMENT TRIGGER WORDS OFFSET");
    assertInText("data/test-newlines.txt", "personal", results, "TAGS");
    assertInText("data/test-newlines.txt", "777-777-7777", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/test-newlines.txt", "83-94", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");
    assertInText("data/test-newlines.txt", "564-456-46", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/test-newlines.txt", "145-154", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");
    assertInText("data/test-newlines.txt", "Text", results, "TEXT PERSONAL TRIGGER WORDS");
    assertInText("data/test-newlines.txt", "19-22", results, "TEXT PERSONAL TRIGGER WORDS OFFSET");

    ASSERT_TRUE(tagger.Close());
}
