/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
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
#include "KeywordTaggingComponent.h"

using namespace MPF::COMPONENT;


void setAlgorithmProperties(Properties &algorithm_properties, const std::map<std::string, std::string> &custom) {

    algorithm_properties["TAGGING_FILE"] = "config/test-text-tags-foreign.json";
    algorithm_properties["STRUCTURED_TEXT_SHARPEN"] = "-1.0";

    for (auto const &x : custom) {
        algorithm_properties[x.first] = x.second;
    }
}

MPFImageJob createImageJob(const std::string &uri, const std::map<std::string, std::string> &custom = {}) {
    Properties algorithm_properties;
    Properties media_properties;

    std::string job_name("Tagger_feed_forawrd_test");
    setAlgorithmProperties(algorithm_properties, custom);

    MPFImageLocation text_tags;
    text_tags.detection_properties["TEXT"] = "Testing 123:\n"
                                             "\n"
                                             "This package contains an OCR engine -\n"
                                             "libtesseract and a command line\n"
                                             "program - tesseract.\n"
                                             "\n"
                                             "The lead developer is Ray Smith. The\n"
                                             "maintainer is Zdenko Podobny. For a\n"
                                             "lsit of contributors see AUTHORS and\n"
                                             "GitHub's log of contributors.";

    MPFImageJob job(job_name, uri, text_tags, algorithm_properties, media_properties);
    return job;

}

MPFGenericJob createGenericJob(const std::string &uri, const std::map<std::string, std::string> &custom = {},
                           bool wild_mode = false) {

    Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("Tagger_test");
    setAlgorithmProperties(algorithm_properties, custom);

    if (wild_mode) {
        MPFGenericTrack text_tags;
        text_tags.detection_properties["TEXT_TYPE"] = "UNSTRUCTURED";
        MPFGenericJob job(job_name, uri, text_tags, algorithm_properties, media_properties);
        return job;
    }

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

bool containsPropFeedForward(const std::string &exp_text, const std::vector<MPFImageLocation> &locations,
                  const std::string &property, int index = -1) {
    if (index != -1) {
        if (locations[index].detection_properties.count(property) == 0) {
            return false;
        }

        std::string text = locations[index].detection_properties.at(property);
        return text.find(exp_text) != std::string::npos;
    }

    for (int i = 0; i < locations.size(); i++) {
        if (locations[i].detection_properties.count(property) == 0) {
            continue;
        }

        std::string text = locations[i].detection_properties.at(property);
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

void assertNotInTextFeedForward(const std::string &file_path, const std::string &expected_text,
                                const std::vector<MPFImageLocation> &location, const std::string &prop, int index = -1) {
    ASSERT_FALSE(containsPropFeedForward(expected_text, location, prop, index))
                                << "Expected tagger to NOT detect "<< prop << " \""  << expected_text << "\" in "
                                << file_path;
}

void runKeywordTaggingFeedForward(const std::string &image, KeywordTagger &tagger,
        std::vector<MPFImageLocation> &text_tags,
        const std::map<std::string, std::string> &custom = {}) {

    MPFImageJob job = createImageJob(image, custom);
    MPFDetectionError rc = tagger.GetDetections(job, text_tags);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(text_tags.empty());

}

void runKeywordTagging(const std::string &uri_path, KeywordTagger &tagger,
                       std::vector<MPFGenericTrack> &text_tags,
                       const std::map<std::string, std::string> &custom = {},
                       bool wild_mode = false) {
    MPFGenericJob job = createGenericJob(uri_path, custom, wild_mode);
    MPFDetectionError rc = tagger.GetDetections(job, text_tags);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(text_tags.empty());
}

TEST(KEYWORDTAGGING, TaggingTest) {
    KeywordTagger tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

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
    KeywordTagger tagger;
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
    KeywordTagger tagger;
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
    KeywordTagger tagger;
    std::vector<MPFGenericTrack> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    ASSERT_NO_FATAL_FAILURE(runKeywordTagging("data/eng-bul.txt", tagger, results, custom_properties));
    assertInText("data/eng-bul.txt", "foreign-text", results, "TAGS", 0);
    assertInText("data/eng-bul.txt", "свободни", results, "TRIGGER_WORDS", 0);
    assertInText("data/eng-bul.txt", "107-114", results, "TRIGGER_WORDS_OFFSET", 0);
    assertInText("data/eng-bul.txt", "Всички хора се раждат свободни", results, "TEXT", 0);

    // Also test mult-keyword phrase tag.
    assertInText("data/eng-bul.txt", "key-phrase", results, "TAGS", 1);
    assertInText("data/eng-bul.txt", "brotherhood", results, "TRIGGER_WORDS", 1);
    //assertInText("data/eng-bul.txt", "436-457", results, "TRIGGER_WORDS_OFFSET", 1);
    //assertInText("data/eng-bul.txt", "All human beings are born free", results, "TEXT", 1);

    ASSERT_TRUE(tagger.Close());

}


TEST(KEYWORDTAGGING, FeedForwardTest) {
    KeywordTagger tagger;
    std::vector<MPFImageLocation> results;
    std::map<std::string, std::string> custom_properties_disabled = {{"FULL_REGEX_SEARCH", "false"}};
    std::map<std::string, std::string> custom_properties = {{}};

    tagger.SetRunDirectory("../plugin");

    ASSERT_TRUE(tagger.Init());

    // Test basic tagging
    ASSERT_NO_FATAL_FAILURE(runKeywordTaggingFeedForward("data/text-demo.txt", tagger, results, custom_properties));
    assertNotInTextFeedForward("data/text-demo.txt", "personal", results, "TAGS");
    ASSERT_TRUE(results[0].detection_properties.at("TAGS").length() == 0);
    results.clear();
}