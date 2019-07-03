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
#include <unistd.h>
#include <gtest/gtest.h>

#define BOOST_NO_CXX11_SCOPED_ENUMS

#include <boost/filesystem.hpp>

#undef BOOST_NO_CXX11_SCOPED_ENUMS

#include "TesseractOCRTextDetection.h"

using namespace MPF::COMPONENT;


void setAlgorithmProperties(Properties &algorithm_properties, const std::map<std::string, std::string> &custom) {
    algorithm_properties["TAGGING_FILE"] = "config/test-text-tags-foreign.json";
    algorithm_properties["SHARPEN"] = "1.0";
    algorithm_properties["TESSERACT_LANGUAGE"] = "eng";
    algorithm_properties["THRS_FILTER"] = "false";
    algorithm_properties["HIST_FILTER"] = "false";
    for (auto const &x : custom) {
        algorithm_properties[x.first] = x.second;
    }
}

MPFImageJob createImageJob(const std::string &uri, const std::map<std::string, std::string> &custom = {}) {
    Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("OCR_test");
    setAlgorithmProperties(algorithm_properties, custom);
    MPFImageJob job(job_name, uri, algorithm_properties, media_properties);
    return job;
}

MPFGenericJob createPDFJob(const std::string &uri, const std::map<std::string, std::string> &custom = {}) {
    Properties algorithm_properties;
    Properties media_properties;
    std::string job_name("OCR_test");
    setAlgorithmProperties(algorithm_properties, custom);
    MPFGenericJob job(job_name, uri, algorithm_properties, media_properties);
    return job;
}

void runImageDetection(const std::string &image_path, TesseractOCRTextDetection &ocr,
                       std::vector<MPFImageLocation> &image_locations,
                       const std::map<std::string, std::string> &custom = {}) {
    MPFImageJob job = createImageJob(image_path, custom);
    MPFDetectionError rc = ocr.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(image_locations.empty());
}

void runDocumentDetection(const std::string &image_path, TesseractOCRTextDetection &ocr,
                          std::vector<MPFGenericTrack> &generic_tracks,
                          const std::map<std::string, std::string> &custom = {}) {
    MPFGenericJob job = createPDFJob(image_path, custom);
    MPFDetectionError rc = ocr.GetDetections(job, generic_tracks);
    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_FALSE(generic_tracks.empty());
}

void assertEmptyDetection(const std::string &image_path, TesseractOCRTextDetection &ocr,
                          std::vector<MPFImageLocation> &image_locations,
                          const std::map<std::string, std::string> &custom = {}) {
    MPFImageJob job = createImageJob(image_path, custom);
    MPFDetectionError rc = ocr.GetDetections(job, image_locations);

    ASSERT_EQ(rc, MPF_DETECTION_SUCCESS);
    ASSERT_TRUE(image_locations.empty());
}

bool containsText(const std::string &exp_text, const std::vector<MPFImageLocation> &locations, int index = -1) {
    if (index != -1) {
        if (locations[index].detection_properties.count("TEXT") == 0) {
            return false;
        }
        std::string text = locations[index].detection_properties.at("TEXT");
        return text.find(exp_text) != std::string::npos;
    }
    for (int i = 0; i < locations.size(); i++) {
        if (locations[i].detection_properties.count("TEXT") == 0) {
            continue;
        }
        std::string text = locations[i].detection_properties.at("TEXT");
        if (text.find(exp_text) != std::string::npos)
            return true;
    }
    return false;
}

bool containsTag(const std::string &exp_tag, const std::vector<MPFImageLocation> &locations, int index = -1) {
    if (index != -1) {
        if (locations[index].detection_properties.count("TAGS") == 0) {
            return false;
        }
        std::string text = locations[index].detection_properties.at("TAGS");
        return text.find(exp_tag) != std::string::npos;
    }
    for (int i = 0; i < locations.size(); i++) {
        if (locations[i].detection_properties.count("TAGS") == 0) {
            continue;
        }
        std::string text = locations[i].detection_properties.at("TAGS");
        if (text.find(exp_tag) != std::string::npos)
            return true;
    }
    return false;
}

void assertTextInImage(const std::string &image_path, const std::string &expected_text,
                       const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_TRUE(containsText(expected_text, locations, index))
                                << "Expected OCR to detect text \"" << expected_text << "\" in " << image_path;
}

void assertTagInImage(const std::string &image_path, const std::string &expected_tag,
                      const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_TRUE(containsTag(expected_tag, locations, index))
                                << "Expected OCR to detect tag \"" << expected_tag << "\" in " << image_path;
}

void assertTextNotInImage(const std::string &image_path, const std::string &expected_text,
                          const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_FALSE(containsText(expected_text, locations, index))
                                << "Expected OCR to NOT detect text \"" << expected_text << "\" in " << image_path;
}

void assertTagNotInImage(const std::string &image_path, const std::string &expected_tag,
                         const std::vector<MPFImageLocation> &locations, int index = -1) {
    ASSERT_FALSE(containsTag(expected_tag, locations, index))
                                << "Expected OCR to NOT detect keyword tag \"" << expected_tag << "\" in "
                                << image_path;
}

void convert_results(std::vector<MPFImageLocation> &im_track, const std::vector<MPFGenericTrack> &gen_track) {
    for (int i = 0; i < gen_track.size(); i++) {
        MPFImageLocation image_location(0, 0, 1, 1);
        image_location.detection_properties = gen_track[i].detection_properties;
        im_track.push_back(image_location);
    }
}

TEST(TESSERACTOCR, ModelTest) {

    // Ensure user can specify custom model directory locations.

    boost::filesystem::remove_all("data/model_dir");
    boost::filesystem::create_directories("data/model_dir/TesseractOCRTextDetection/tessdata");

    std::string model = boost::filesystem::absolute(
            "../plugin/TesseractOCRTextDetection/tessdata/eng.traineddata").string();
    symlink(model.c_str(), "data/model_dir/TesseractOCRTextDetection/tessdata/custom.traineddata");

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"TESSERACT_LANGUAGE",    "custom,eng"},
                                                            {"MODELS_DIR_PATH",       "data/model_dir"},
                                                            {"ENABLE_OSD_AUTOMATION", "false"}};

    runImageDetection("data/eng.png", ocr, results, custom_properties);

    boost::filesystem::remove_all("data/model_dir");
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "custom")
                                << "Shared models directory not loaded.";
    ASSERT_TRUE(results[1].detection_properties.at("TEXT_LANGUAGE") == "eng") << "Tessdata directory not loaded.";
    ASSERT_TRUE(results.size() == 2) << "Expected two models to be properly loaded into component.";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, TwoPassOCRTest) {

    // Ensure that two pass OCR correctly works to process upside down text even w/out OSD support.

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "false"},
                                                            {"ROTATE_AND_DETECT",     "true"}};

    runImageDetection("data/eng.png", ocr, results, custom_properties);
    ASSERT_TRUE(std::stoi(results[0].detection_properties.at("TWO_PASS_OCR_ROTATION")) == 0) << "Expected 0 degree text rotation.";
    assertTextInImage("data/eng.png", "All human beings", results);
    results.clear();

    runImageDetection("data/eng-rotated.png", ocr, results, custom_properties);
    ASSERT_TRUE(std::stoi(results[0].detection_properties.at("TWO_PASS_OCR_ROTATION")) == 180) << "Expected 180 degree text rotation.";
    assertTextInImage("data/eng-rotated.png", "All human beings", results);
    results.clear();

    // Test to make sure threshold parameter is working.
    custom_properties = {{"ENABLE_OSD_AUTOMATION",            "false"},
                         {"ROTATE_AND_DETECT_MIN_CONFIDENCE", "10"},
                         {"ROTATE_AND_DETECT",                "true"}};

    runImageDetection("data/eng-rotated.png", ocr, results, custom_properties);
    ASSERT_TRUE(std::stoi(results[0].detection_properties.at("TWO_PASS_OCR_ROTATION")) == 0) << "Expected 0 degree text rotation.";
    assertTextNotInImage("data/eng-rotated.png", "All human beings", results);
    ASSERT_TRUE(ocr.Close());

}



TEST(TESSERACTOCR, TrackFilterTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "false"},
                                                            {"ROTATE_AND_DETECT",     "true"},
                                                            {"TESSERACT_LANGUAGE",    "eng,chi_sim"},
                                                            {"MAX_TEXT_TRACKS",       "1"}};

    // Check that the top text track is properly returned after filtering.
    runImageDetection("data/eng.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "eng") << "Top text track not returned.";
    ASSERT_TRUE(results.size() == 1) << "Expected output track only.";
    results.clear();

    // Check both tracks are returned by default
    custom_properties = {{"ENABLE_OSD_AUTOMATION", "false"},
                         {"TESSERACT_LANGUAGE",    "eng,chi_sim"}};
    runImageDetection("data/eng.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "eng") << "Expected English track";
    ASSERT_TRUE(results[1].detection_properties.at("TEXT_LANGUAGE") == "chi_sim") << "Expected Chinese track";
    ASSERT_TRUE(results.size() == 2) << "Expected two output tracks.";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, ImageTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "false"}};

    // Test basic keyword and text detection.
    runImageDetection("data/text-demo.png", ocr, results, custom_properties);
    assertTextInImage("data/text-demo.png", "TESTING 123", results);
    assertTextNotInImage("data/text-demo.png", "Ponies", results);
    assertTagNotInImage("data/text-demo.png", "personal", results);
    results.clear();

    // Test multiple keyword tagging.
    runImageDetection("data/tags-keyword.png", ocr, results, custom_properties);
    assertTextInImage("data/tags-keyword.png", "Passenger Passport", results);
    assertTagInImage("data/tags-keyword.png", "identity document, travel", results);
    results.clear();

    // Test keyword and regex tagging.
    // Keyword and regex tag match to same category for personal.
    // Regex tagging also picks up financial.
    // Keyword tagging picks up vehicle.
    // Three tags should be detected in total.
    runImageDetection("data/tags-keywordregex.png", ocr, results, custom_properties);
    assertTagInImage("data/tags-keywordregex.png", "financial, personal, vehicle", results);
    results.clear();

    // Test multiple regex tagging.
    runImageDetection("data/tags-regex.png", ocr, results, custom_properties);
    assertTagInImage("data/tags-regex.png", "financial, personal", results);

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, FilterTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "false"}};

    // Check no text detected for blank image.
    assertEmptyDetection("data/blank.png", ocr, results, custom_properties);
    results.clear();

    // Check no text detected for image with junk text when hist filter is enabled.
    runImageDetection("data/junk-text.png", ocr, results, custom_properties);
    results.clear();
    custom_properties = {{"HIST_FILTER",           "true"},
                         {"TESSERACT_OEM",         "0"},
                         {"ENABLE_OSD_AUTOMATION", "false"}};
    assertEmptyDetection("data/junk-text.png", ocr, results, custom_properties);
    results.clear();

    // Check no text detected for image with junk text when thrs filter is enabled.
    runImageDetection("data/junk-text.png", ocr, results);
    results.clear();
    custom_properties = {{"THRS_FILTER",           "true"},
                         {"TESSERACT_OEM",         "0"},
                         {"ENABLE_OSD_AUTOMATION", "false"}};
    assertEmptyDetection("data/junk-text.png", ocr, results, custom_properties);

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, ModeTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results_old, results_new;
    ASSERT_TRUE(ocr.Init());

    // Check that PSM and OEM settings impact text extraction behavior.
    std::map<std::string, std::string> custom_properties = {{"TESSERACT_OEM",         "0"},
                                                            {"ENABLE_OSD_AUTOMATION", "false"}};
    runImageDetection("data/junk-text.png", ocr, results_old, custom_properties);

    custom_properties = {{"TESSERACT_OEM",         "3"},
                         {"ENABLE_OSD_AUTOMATION", "false"}};
    runImageDetection("data/junk-text.png", ocr, results_new, custom_properties);

    ASSERT_FALSE(results_old[0].detection_properties.at("TEXT") == results_new[0].detection_properties.at("TEXT"));

    results_old.clear();
    results_new.clear();

    custom_properties = {{"TESSERACT_PSM",         "3"},
                         {"ENABLE_OSD_AUTOMATION", "false"}};
    runImageDetection("data/junk-text.png", ocr, results_old, custom_properties);

    custom_properties = {{"TESSERACT_PSM",         "13"},
                         {"ENABLE_OSD_AUTOMATION", "false"}};
    runImageDetection("data/junk-text.png", ocr, results_new, custom_properties);

    ASSERT_FALSE(results_old[0].detection_properties.at("TEXT") == results_new[0].detection_properties.at("TEXT"));

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Check that OSD works.

    // Check script detection.
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION",             "true"},
                                                            {"MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE", "0.30"},
                                                            {"MIN_OSD_SCRIPT_SCORE",              "40"}};
    runImageDetection("data/eng.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Latin") << "Expected Latin script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Latin") << "Expected latin script.";
    results.clear();

    // Check orientation detection. Text should be properly extracted.
    runImageDetection("data/eng-rotated.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "180") << "Expected 180 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Latin") << "Expected Latin script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Latin") << "Expected latin script.";
    assertTextInImage("data/eng-rotated.png", "All human beings", results);

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDConfidenceFilteringTest) {
    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Check script confidence filtering.
    // In the OSDTest for eng.png, the component will accept the detected script and run it as the text_language model (script/Latin).
    // By setting the min script confidence level too high, the component must default back to original setting (eng).
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION",             "true"},
                                                            {"MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE", "100"},
                                                            {"MIN_OSD_SCRIPT_SCORE",              "40"}};
    runImageDetection("data/eng.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Latin") << "Expected Latin script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "eng")
                                << "Expected default language (eng) due to script confidence rejection.";
    assertTextInImage("data/eng.png", "All human beings", results);
    results.clear();

    // Check script score filtering
    // In the OSDTest for eng.png, the component will accept the detected script and run it as the text_language model (script/Latin).
    // By setting the min script score level too high, the component must default back to original setting (eng).
    custom_properties = {{"ENABLE_OSD_AUTOMATION",             "true"},
                         {"MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE", "0"},
                         {"MIN_OSD_SCRIPT_SCORE",              "60"}};
    runImageDetection("data/eng.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Latin") << "Expected Latin script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "eng")
                                << "Expected default language (eng) due to script score rejection.";
    assertTextInImage("data/eng.png", "All human beings", results);
    results.clear();

    // Check orientation confidence filtering.
    // In the OSDTest for eng-rotated.png, the component will accept a detected rotation of 180 degrees.
    // By setting the min rotation confidence level too high, the component must default back to original setting (0 degrees rotation).
    custom_properties = {{"TESSERACT_PSM",               "0"},
                         {"MIN_OSD_TEXT_ORIENTATION_CONFIDENCE", "200"}};
    runImageDetection("data/eng-rotated.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected default text rotation.";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDCommonTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Check that "Common" script is reported but secondary script is used to run OCR.
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION",              "true"},
                                                            {"MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE",  "0.0"},
                                                            {"MIN_OSD_SCRIPT_SCORE",               "0"},
                                                            {"MAX_OSD_SCRIPTS",                    "1"},
                                                            {"MIN_OSD_SECONDARY_SCRIPT_THRESHOLD", "0.0"}};
    runImageDetection("data/numeric.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Common") << "Expected Common script.";
    ASSERT_TRUE(results[0].detection_properties.at("SECONDARY_SCRIPTS") == "Cyrillic") << "Expected Cyrillic script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Cyrillic")
                                << "Expected Cyrillic script.";

    results.clear();

    // Check that "Common" script is reported but default language is used to run OCR.
    custom_properties = {{"ENABLE_OSD_AUTOMATION", "true"},
                         {"MAX_OSD_SCRIPTS",       "1"}};
    runImageDetection("data/numeric.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Common") << "Expected Common script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "eng") << "Expected default language (eng).";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDMultilanguageScriptTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());


    // Check multi-language script detection.
    // Set max scripts to three, however based on filters only two scripts should be processed as a single track.
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "true"},
                                                            {"MAX_OSD_SCRIPTS",       "3"},
                                                            {"MIN_OSD_SCRIPT_SCORE",  "45.0"}};
    runImageDetection("data/eng-bul.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic")
                                << "Expected Cyrillic as primary script.";
    ASSERT_TRUE(results[0].detection_properties.at("SECONDARY_SCRIPTS") == "Latin")
                                << "Expected Latin as secondary script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Cyrillic+script/Latin")
                                << "Expected both scripts to run together.";
    assertTextInImage("data/eng-bul.png", "Всички хора се раждат ", results);
    assertTextInImage("data/eng-bul.png", "All human beings", results);
    results.clear();

    // Check multi-language script detection.
    // Modified script behavior so that individual scripts are run separately.
    custom_properties = {{"ENABLE_OSD_AUTOMATION", "true"},
                         {"MAX_OSD_SCRIPTS",       "3"},
                         {"COMBINE_OSD_SCRIPTS",   "false"},
                         {"MIN_OSD_SCRIPT_SCORE",  "45.0"}};
    runImageDetection("data/eng-bul.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic")
                                << "Expected Cyrillic as primary script.";
    ASSERT_TRUE(results[0].detection_properties.at("SECONDARY_SCRIPTS") == "Latin")
                                << "Expected Latin as secondary script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Cyrillic")
                                << "Expected both scripts to run separately.";

    ASSERT_TRUE(results[1].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[1].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic")
                                << "Expected Cyrillic as primary script.";
    ASSERT_TRUE(results[1].detection_properties.at("SECONDARY_SCRIPTS") == "Latin")
                                << "Expected Latin as secondary script.";
    ASSERT_TRUE(results[1].detection_properties.at("TEXT_LANGUAGE") == "script/Latin")
                                << "Expected both scripts to run separately.";

    assertTextInImage("data/eng-bul.png", "Всички хора се раждат ", results, 0);
    assertTextInImage("data/eng-bul.png", "All human beings", results, 1);

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDMultiPageTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());

    // Check multiple page OSD processing (script and orientation detection).
    std::vector<MPFGenericTrack> results_pdf;
    std::map<std::string, std::string> custom_properties = {{"TESSERACT_PSM", "0"}};
    runDocumentDetection("data/osd-tests.pdf", ocr, results_pdf, custom_properties);
    convert_results(results, results_pdf);
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Han")
                                << "Expected Chinese/Han detected as primary script.";
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[1].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic")
                                << "Expected Cyrillic detected as primary script.";
    ASSERT_TRUE(results[1].detection_properties.at("ROTATION") == "180") << "Expected 180 degree text rotation.";
    results_pdf.clear();
    results.clear();

    custom_properties = {{"ENABLE_OSD_AUTOMATION",             "true"},
                         {"MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE", "0.30"},
                         {"MIN_OSD_SCRIPT_SCORE",              "40"}};

    // Check multiple page OSD processing.
    // Ensure that the default language is properly reset.

    runDocumentDetection("data/osd-check-defaults.pdf", ocr, results_pdf, custom_properties);
    convert_results(results, results_pdf);
    ASSERT_TRUE(results.size() == 2) << "Expected two text tracks (middle page is blank and should be ignored).";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic") << "Expected Cyrillic script.";
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Cyrillic")
                                << "Expected Cyrillic script.";
    ASSERT_TRUE(results[1].detection_properties.at("PRIMARY_SCRIPT") == "NULL")
                                << "Expected NULL script due to insufficient text.";
    ASSERT_TRUE(results[1].detection_properties.at("TEXT_LANGUAGE") == "eng")
                                << "Expected default language (eng) for second track.";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDSecondaryScriptThresholdTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    // Check multi-language script detection.
    // Set max scripts to two, two scripts should be picked up and processed.
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "true"},
                                                            {"MAX_OSD_SCRIPTS",       "2"},
                                                            {"MIN_OSD_SCRIPT_SCORE",  "30.0"}};
    runImageDetection("data/eng-bul.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic")
                                << "Expected Cyrillic as primary script.";
    ASSERT_TRUE(results[0].detection_properties.at("SECONDARY_SCRIPTS") == "Latin")
                                << "Expected Latin as secondary script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Cyrillic+script/Latin")
                                << "Expected both Cyrillic and Latin scripts";

    results.clear();

    // Set max scripts to two, however by setting min secondary script threshold too high, only one script should be processed.
    custom_properties = {{"ENABLE_OSD_AUTOMATION",              "true"},
                         {"MAX_OSD_SCRIPTS",                    "2"},
                         {"MIN_OSD_SCRIPT_SCORE",               "45.0"},
                         {"MIN_OSD_SECONDARY_SCRIPT_THRESHOLD", "0.99"}};
    runImageDetection("data/eng-bul.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Cyrillic")
                                << "Expected Cyrillic as primary script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/Cyrillic")
                                << "Expected only Cyrillic script due to threshold filtering.";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, OSDHanOrientationTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "true"},
                                                            {"MIN_OSD_SCRIPT_SCORE",  "45.0"},
                                                            {"MAX_TEXT_TRACKS",       "1"},
                                                            {"COMBINE_OSD_SCRIPTS",   "false"}};

    runImageDetection("data/chinese.png", ocr, results, custom_properties);
    ASSERT_TRUE(results[0].detection_properties.at("ROTATION") == "0") << "Expected 0 degree text rotation.";
    ASSERT_TRUE(results[0].detection_properties.at("PRIMARY_SCRIPT") == "Han") << "Expected Chinese as primary script.";
    ASSERT_TRUE(results[0].detection_properties.at("TEXT_LANGUAGE") == "script/HanS+script/HanT")
                                << "Expected Chinese horizontal text to be the correct output model";

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, LanguageTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"TESSERACT_LANGUAGE",    "eng, bul"},
                                                            {"ENABLE_OSD_AUTOMATION", "false"}};

    // Test multilanguage text extraction.
    runImageDetection("data/eng-bul.png", ocr, results, custom_properties);
    assertTagInImage("data/eng-bul.png", "foreign-text", results, 1);
    assertTextInImage("data/eng-bul.png", "Всички хора се раждат свободни", results, 1);
    // Also test mult-keyword phrase tag.
    assertTagInImage("data/eng-bul.png", "key-phrase", results, 0);
    assertTextInImage("data/eng-bul.png", "All human beings are born free", results, 0);

    ASSERT_TRUE(ocr.Close());
}

TEST(TESSERACTOCR, DocumentTest) {

    TesseractOCRTextDetection ocr;
    ocr.SetRunDirectory("../plugin");
    std::vector<MPFImageLocation> results;
    std::vector<MPFGenericTrack> results_pdf;
    ASSERT_TRUE(ocr.Init());
    std::map<std::string, std::string> custom_properties = {{"ENABLE_OSD_AUTOMATION", "false"}};

    // Test document text extraction.
    runDocumentDetection("data/test.pdf", ocr, results_pdf, custom_properties);
    convert_results(results, results_pdf);
    assertTagInImage("data/test.pdf", "personal", results, 0);
    assertTextInImage("data/test.pdf", "This is a test", results, 0);
    assertTagInImage("data/test.pdf", "vehicle", results, 1);
    assertTextInImage("data/test.pdf", "Vehicle", results, 1);

    ASSERT_TRUE(ocr.Close());
}
