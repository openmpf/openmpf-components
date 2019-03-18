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

#include <string>
#include <MPFDetectionComponent.h>

#include <gtest/gtest.h>

#include "TesseractOCRTextDetection.h"

using namespace MPF::COMPONENT;


void setAlgorithmProperties(Properties &algorithm_properties, const std::map<std::string,std::string> & custom) {
    algorithm_properties["TAGGING_FILE"] = "config/test-text-tags-foreign.json";
    algorithm_properties["SHARPEN"] = "1.0";
    algorithm_properties["TESSERACT_LANGUAGE"] = "eng";
    algorithm_properties["THRS_FILTER"] = "false";
    algorithm_properties["HIST_FILTER"] = "false";
    for (auto const& x : custom) {
            algorithm_properties[x.first] = x.second;
    }
}

MPFImageJob createImageJob(const std::string &uri, const std::map<std::string,std::string> & custom = {}){
    Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("OCR_test");
    setAlgorithmProperties(algorithm_properties, custom);
    MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
    return job;
}

MPFGenericJob createPDFJob(const std::string &uri, const std::map<std::string,std::string> & custom = {}){
    Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("OCR_test");
    setAlgorithmProperties(algorithm_properties, custom);
    MPFGenericJob job(job_name, uri, algorithm_properties, media_properties);
    return job;
}


void runImageDetection(const std::string &image_path, TesseractOCRTextDetection &ocr, std::vector<MPFImageLocation> &image_locations, const std::map<std::string,std::string> & custom = {}) {
    MPFImageJob job = createImageJob(image_path, custom);
    MPFDetectionError rc = ocr.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(image_locations.empty());
}

void runDocumentDetection(const std::string &image_path, TesseractOCRTextDetection &ocr, std::vector<MPFGenericTrack> &generic_tracks, const std::map<std::string,std::string> & custom = {}) {
      MPFGenericJob job = createPDFJob(image_path, custom);
      MPFDetectionError rc = ocr.GetDetections(job, generic_tracks);
      ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
      ASSERT_FALSE(generic_tracks.empty());
}

void assertEmptyDetection(const std::string &image_path, TesseractOCRTextDetection &ocr, std::vector<MPFImageLocation> &image_locations, const std::map<std::string,std::string> & custom = {}) {
    MPFImageJob job = createImageJob(image_path, custom);
    MPFDetectionError rc = ocr.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_TRUE(image_locations.empty());
}

bool containsText(const std::string &exp_text, const std::vector<MPFImageLocation> &locations, int index = -1) {

    if (index != -1) {
        std::string text = locations[index].detection_properties.at("TEXT");
        return text.find(exp_text) != std::string::npos;
    }
    for (int i = 0; i < locations.size(); i++) {
        std::string text = locations[i].detection_properties.at("TEXT");
        if(text.find(exp_text) != std::string::npos)
            return true;
    }
    return false;
}

bool containsTag(const std::string &exp_tag, const std::vector<MPFImageLocation> &locations, int index = -1) {
    if (index != -1) {
        std::string text = locations[index].detection_properties.at("TAGS");
        return text.find(exp_tag) != std::string::npos;
    }
    for (int i = 0; i < locations.size(); i++) {
        std::string text = locations[i].detection_properties.at("TAGS");
        if(text.find(exp_tag) != std::string::npos)
            return true;
    }
    return false;
}

void assertTextInImage(const std::string &image_path, const std::string &expected_text, const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_TRUE(containsText(expected_text, locations, index))
                               << "Expected OCR to detect text \"" << expected_text << "\" in " << image_path;
}

void assertTagInImage(const std::string &image_path, const std::string &expected_tag, const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_TRUE(containsTag(expected_tag, locations, index))
                               << "Expected OCR to detect tag \"" << expected_tag << "\" in " << image_path;
}

void assertTextNotInImage(const std::string &image_path, const std::string &expected_text, const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_FALSE(containsText(expected_text, locations, index))
                               << "Expected OCR to NOT detect text \"" << expected_text << "\" in " << image_path;
}

void assertTagNotInImage(const std::string &image_path, const std::string &expected_tag, const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_FALSE(containsTag(expected_tag, locations, index))
                               << "Expected OCR to NOT detect keyword tag \"" << expected_tag << "\" in " << image_path;
}

void convert_results(std::vector<MPFImageLocation>& im_track, const std::vector<MPFGenericTrack>& gen_track ) {
    for (int i = 0; i < gen_track.size(); i++) {
        MPFImageLocation image_location(0, 0, 1, 1);
        image_location.detection_properties = gen_track[i].detection_properties;
        im_track.push_back(image_location);
    }
}

TEST(TESSERACTOCR, ImageTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Test basic keyword and text detection.
    runImageDetection("data/text-demo.png", ocr, results);
    assertTextInImage("data/text-demo.png", "TESTING 123", results);
    assertTextNotInImage("data/text-demo.png", "Ponies", results);
    assertTagNotInImage("data/text-demo.png", "personal", results);
    results.clear();

    // Test multiple keyword tagging.
    runImageDetection("data/tags-keyword.png", ocr, results);
    assertTextInImage("data/tags-keyword.png", "Passenger Passport", results);
    assertTagInImage("data/tags-keyword.png", "identity document, travel", results);
    results.clear();

    // Test keyword and regex tagging.
    // Keyword and regex tag match to same category for personal.
    // Regex tagging also picks up financial.
    // Keyword tagging picks up vehicle.
    // Three tags should be detected in total.
    runImageDetection("data/tags-keywordregex.png", ocr, results);
    assertTagInImage("data/tags-keywordregex.png", "financial, personal, vehicle", results);
    results.clear();

    // Test multiple regex tagging.
    runImageDetection("data/tags-regex.png", ocr, results);
    assertTagInImage("data/tags-regex.png", "financial, personal", results);
    results.clear();
    }

TEST(TESSERACTOCR, FilterTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Check no text detected for blank image.
    assertEmptyDetection("data/blank.png", ocr, results);
    results.clear();

    // Check no text detected for image with junk text when hist filter is enabled.
    runImageDetection("data/junk-text.png", ocr, results);
    results.clear();
    std::map<std::string,std::string> custom_properties = {{"HIST_FILTER","true"}, {"TESSERACT_OEM","0"}};
    assertEmptyDetection("data/junk-text.png", ocr, results,  custom_properties);
    results.clear();

    // Check no text detected for image with junk text when thrs filter is enabled.
    runImageDetection("data/junk-text.png", ocr, results);
    results.clear();
    std::map<std::string,std::string> custom_properties2 = {{"THRS_FILTER","true"}, {"TESSERACT_OEM","0"}};
    assertEmptyDetection("data/junk-text.png", ocr, results, custom_properties2);
    results.clear();
}

TEST(TESSERACTOCR, ModeTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results_old, results_new;
    ASSERT_TRUE(ocr.Init());

    // Check that PSM and OEM settings impact text extraction behavior.
    std::map<std::string,std::string> custom_properties = {{"TESSERACT_OEM","0"}};
    runImageDetection("data/junk-text.png", ocr, results_old,  custom_properties);

    custom_properties = {{"TESSERACT_OEM","3"}};
    runImageDetection("data/junk-text.png", ocr, results_new,  custom_properties);

    ASSERT_FALSE(results_old[0].detection_properties.at("TEXT") == results_new[0].detection_properties.at("TEXT"));

    results_old.clear();
    results_new.clear();

    custom_properties = {{"TESSERACT_PSM","3"}};
    runImageDetection("data/junk-text.png", ocr, results_old,  custom_properties);

    custom_properties = {{"TESSERACT_PSM","13"}};
    runImageDetection("data/junk-text.png", ocr, results_new,  custom_properties);

    ASSERT_FALSE(results_old[0].detection_properties.at("TEXT") == results_new[0].detection_properties.at("TEXT"));
}

TEST(TESSERACTOCR, OSDTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Check that OSD works.

    // Check script detection.
    std::map<std::string,std::string> custom_properties = {{"ENABLE_OSD","true"}, {"ENABLE_OSD_TRACK_REPORT","false"},
        {"MIN_SCRIPT_CONFIDENCE","0.80"}};
    runImageDetection("data/eng.png", ocr, results,  custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("LANGUAGE") == "script/Latin");

    results.clear();

    // Check orientation detection.
    runImageDetection("data/eng_rotated.png", ocr, results,  custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("LANGUAGE") == "script/Latin");
    assertTextInImage("data/eng_rotated.png", "All human beings", results);
}


TEST(TESSERACTOCR, LanguageTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string,std::string> custom_properties3 = {{"TESSERACT_LANGUAGE", "eng, bul"}};

    // Test multilanguage text extraction.
    runImageDetection("data/eng-bul.png", ocr, results, custom_properties3);
    assertTagInImage("data/eng-bul.png", "foreign-text", results, 1);
    assertTextInImage("data/eng-bul.png", "Всички хора се раждат свободни", results, 1);
    // Also test mult-keyword phrase tag.
    assertTagInImage("data/eng-bul.png", "key-phrase", results, 0);
    assertTextInImage("data/eng-bul.png", "All human beings are born free", results, 0);
    results.clear();
}

TEST(TESSERACTOCR, DocumentTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    std::vector<MPFGenericTrack> results_pdf;
    ASSERT_TRUE(ocr.Init());

    // Test document text extraction.
    runDocumentDetection("data/test.pdf", ocr, results_pdf);
    convert_results(results, results_pdf);
    assertTagInImage("data/test.pdf", "personal", results, 0);
    assertTextInImage("data/test.pdf", "This is a test", results, 0);
    assertTagInImage("data/test.pdf", "vehicle", results, 1);
    assertTextInImage("data/test.pdf", "Vehicle", results, 1);
}
