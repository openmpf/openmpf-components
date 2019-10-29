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

#include <map>
#include <iostream>
#include <boost/regex.hpp>
#include <boost/locale.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/filesystem.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>
#include <algorithm>
#include <cctype>
#include "TesseractOCRTextDetection.h"
#include "detectionComponentUtils.h"
#include <Utils.h>
#include <fstream>
#include "JSON.h"
#include "MPFSimpleConfigLoader.h"
#include <tesseract/baseapi.h>
#include <tesseract/osdetect.h>
#include <unicharset.h>
#include <Magick++.h>
#include <thread>

using namespace MPF;
using namespace COMPONENT;
using namespace std;
using log4cxx::Logger;
using log4cxx::xml::DOMConfigurator;
typedef boost::multi_index::multi_index_container<string,
            boost::multi_index::indexed_by<
                boost::multi_index::sequenced<>,
                boost::multi_index::ordered_unique<boost::multi_index::identity<string>>
            >> unique_vector;


bool TesseractOCRTextDetection::Init() {
    // Set global C++ locale.
    // Required for boost function calls.
    // Also overwrites C locale:
    // https://en.cppreference.com/w/cpp/locale/locale/global.
    boost::locale::generator gen;
    locale loc = gen("");
    locale::global(loc);

    // Reset C locale back to default.
    // Required for Tesseract API calls:
    // https://github.com/tesseract-ocr/tesseract/commit/3292484f67af8bdda23aa5e510918d0115785291
    setlocale(LC_ALL, "C");

    // Determine where the executable is running.
    string run_dir = GetRunDirectory();
    if (run_dir == "") {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/TesseractOCRTextDetection";
    string config_path = plugin_path + "/config";
    cout << "looking for logger at " << plugin_path + "/config/Log4cxxConfig.xml" << endl;
    log4cxx::xml::DOMConfigurator::configure(plugin_path + "/config/Log4cxxConfig.xml");
    hw_logger_ = log4cxx::Logger::getLogger("TesseractOCRTextDetection");

    LOG4CXX_DEBUG(hw_logger_, "Plugin path: " << plugin_path);
    LOG4CXX_INFO(hw_logger_, "Initializing TesseractOCRTextDetection");
    set_default_parameters();
    default_ocr_fset.model_dir = "";

    // Once this is done - parameters will be set and set_read_config_parameters() can be called again to revert back
    // to the params read at initialization.
    string config_params_path = config_path + "/mpfOCR.ini";
    int rc = LoadConfig(config_params_path, parameters);
    if (rc) {
        LOG4CXX_ERROR(hw_logger_, "Could not parse config file: " << config_params_path);
        return false;
    }
    set_read_config_parameters();

    LOG4CXX_INFO(hw_logger_, "INITIALIZED COMPONENT.");
    return true;
}

// Ensure this is idempotent because it's called by the destructor.
bool TesseractOCRTextDetection::Close() {
    for (auto const &element : tess_api_map) {
        tesseract::TessBaseAPI *tess_api = element.second;
        tess_api->End();
        delete tess_api;
    }
    tess_api_map.clear();
    return true;
}

TesseractOCRTextDetection::~TesseractOCRTextDetection() {
    Close();
}

string TesseractOCRTextDetection::GetDetectionType() {
    return "TEXT";
}

bool TesseractOCRTextDetection::Supports(MPFDetectionDataType data_type) {

    // Supports images and documents, no audio or video data types.
    return data_type == MPFDetectionDataType::IMAGE || data_type == MPFDetectionDataType::UNKNOWN;
}

/*
 * Called during Init.
 * Initialize default parameters.
 */
void TesseractOCRTextDetection::set_default_parameters() {
    default_ocr_fset.psm = 3;
    default_ocr_fset.oem = 3;
    default_ocr_fset.sharpen = -1.0;
    default_ocr_fset.scale = 2.4;
    default_ocr_fset.invert = false;
    default_ocr_fset.enable_otsu_thrs = false;
    default_ocr_fset.enable_adaptive_thrs = false;
    default_ocr_fset.tesseract_lang = "eng";
    default_ocr_fset.adaptive_thrs_pixel = 11;
    default_ocr_fset.adaptive_thrs_c = 0;
    default_ocr_fset.enable_osd = true;
    default_ocr_fset.combine_detected_scripts = true;
    default_ocr_fset.min_script_confidence = 2.0;
    default_ocr_fset.min_script_score = 50.0;
    default_ocr_fset.min_orientation_confidence = 2.0;
    default_ocr_fset.max_scripts = 1;
    default_ocr_fset.max_text_tracks = 0;
    default_ocr_fset.min_secondary_script_thrs = 0.80;
    default_ocr_fset.rotate_and_detect = false;
    default_ocr_fset.rotate_and_detect_min_confidence = 95.0;

    default_ocr_fset.enable_hist_equalization = false;
    default_ocr_fset.enable_adaptive_hist_equalization = false;
    default_ocr_fset.min_height = -1;
    default_ocr_fset.adaptive_hist_tile_size = 5;
    default_ocr_fset.adaptive_hist_clip_limit = 2.0;

    default_ocr_fset.processing_wild_text = false;
    default_ocr_fset.full_regex_search = false;
    default_ocr_fset.enable_osd_fallback = true;
    default_ocr_fset.max_parallel_ocr_threads = 4;
    default_ocr_fset.max_parallel_pdf_threads = 4;
}

/*
 * Called during Init.
 * Copy over parameters values from .ini file.
 */
void TesseractOCRTextDetection::set_read_config_parameters() {

    // Load wild image preprocessing settings.
    if (parameters.contains("UNSTRUCTURED_TEXT_ENABLE_PREPROCESSING")) {
        default_ocr_fset.processing_wild_text = (parameters["UNSTRUCTURED_TEXT_ENABLE_PREPROCESSING"].toInt() > 0);
    }
    if (default_ocr_fset.processing_wild_text) {
        if (parameters.contains("UNSTRUCTURED_TEXT_ENABLE_OTSU_THRS")) {
            default_ocr_fset.enable_otsu_thrs = (parameters["UNSTRUCTURED_TEXT_ENABLE_OTSU_THRS"].toInt() > 0);
        }
        if (parameters.contains("UNSTRUCTURED_TEXT_ENABLE_ADAPTIVE_THRS")) {
            default_ocr_fset.enable_adaptive_thrs = (parameters["UNSTRUCTURED_TEXT_ENABLE_ADAPTIVE_THRS"].toInt() > 0);
        }
        if (parameters.contains("UNSTRUCTURED_TEXT_ENABLE_ADAPTIVE_HIST_EQUALIZATION")) {
            default_ocr_fset.enable_adaptive_hist_equalization= (parameters["UNSTRUCTURED_TEXT_ENABLE_ADAPTIVE_HIST_EQUALIZATION"].toInt() > 0);
        }
        if (parameters.contains("UNSTRUCTURED_TEXT_ENABLE_HIST_EQUALIZATION")) {
            default_ocr_fset.enable_hist_equalization= (parameters["UNSTRUCTURED_TEXT_ENABLE_HIST_EQUALIZATION "].toInt() > 0);
        }
        if (parameters.contains("UNSTRUCTURED_TEXT_SHARPEN")) {
            default_ocr_fset.sharpen = parameters["UNSTRUCTURED_TEXT_SHARPEN"].toDouble();
        }
        if (parameters.contains("UNSTRUCTURED_TEXT_SCALE")) {
            default_ocr_fset.scale = parameters["UNSTRUCTURED_TEXT_SCALE"].toDouble();
        }
    } else {
        // Load default image preprocessing settings.

        if (parameters.contains("STRUCTURED_TEXT_ENABLE_OTSU_THRS")) {
            default_ocr_fset.enable_otsu_thrs = (parameters["STRUCTURED_TEXT_ENABLE_OTSU_THRS"].toInt() > 0);
        }
        if (parameters.contains("STRUCTURED_TEXT_ENABLE_ADAPTIVE_THRS")) {
            default_ocr_fset.enable_adaptive_thrs = (parameters["STRUCTURED_TEXT_ENABLE_ADAPTIVE_THRS"].toInt() > 0);
        }
        if (parameters.contains("STRUCTURED_TEXT_ENABLE_ADAPTIVE_HIST_EQUALIZATION")) {
            default_ocr_fset.enable_adaptive_hist_equalization= (parameters["STRUCTURED_TEXT_ENABLE_ADAPTIVE_HIST_EQUALIZATION"].toInt() > 0);
        }
        if (parameters.contains("STRUCTURED_TEXT_ENABLE_HIST_EQUALIZATION")) {
            default_ocr_fset.enable_hist_equalization= (parameters["STRUCTURED_TEXT_ENABLE_HIST_EQUALIZATION "].toInt() > 0);
        }
        if (parameters.contains("STRUCTURED_TEXT_SHARPEN")) {
            default_ocr_fset.sharpen = parameters["STRUCTURED_TEXT_SHARPEN"].toDouble();
        }
        if (parameters.contains("STRUCTURED_TEXT_SCALE")) {
            default_ocr_fset.scale = parameters["STRUCTURED_TEXT_SCALE"].toDouble();
        }
    }
    if (parameters.contains("INVERT")) {
        default_ocr_fset.invert = (parameters["INVERT"].toInt() > 0);
    }
    if (parameters.contains("MIN_HEIGHT")) {
        default_ocr_fset.min_height = parameters["MIN_HEIGHT"].toInt();
    }
    if (parameters.contains("ADAPTIVE_HIST_TILE_SIZE")){
        default_ocr_fset.adaptive_hist_tile_size = parameters["ADAPTIVE_HIST_TILE_SIZE"].toInt();
    }
    if (parameters.contains("ADAPTIVE_HIST_CLIP_LIMIT")) {
        default_ocr_fset.adaptive_hist_clip_limit = parameters["ADAPTIVE_HIST_CLIP_LIMIT"].toDouble();
    }
    if (parameters.contains("ADAPTIVE_THRS_CONSTANT")) {
        default_ocr_fset.adaptive_thrs_c = parameters["ADAPTIVE_THRS_CONSTANT"].toDouble();
    }
    if (parameters.contains("ADAPTIVE_THRS_BLOCKSIZE")) {
        default_ocr_fset.adaptive_thrs_pixel = parameters["ADAPTIVE_THRS_BLOCKSIZE"].toInt();
    }
    if (parameters.contains("TESSERACT_LANGUAGE")) {
        default_ocr_fset.tesseract_lang = parameters["TESSERACT_LANGUAGE"].toStdString();
    }
    if (parameters.contains("TESSERACT_PSM")) {
        default_ocr_fset.psm = parameters["TESSERACT_PSM"].toInt();
    }
    if (parameters.contains("TESSERACT_OEM")) {
        default_ocr_fset.oem = parameters["TESSERACT_OEM"].toInt();
    }
    if (parameters.contains("ENABLE_OSD_AUTOMATION")) {
        default_ocr_fset.enable_osd = (parameters["ENABLE_OSD_AUTOMATION"].toInt() > 0);
    }
    if (parameters.contains("COMBINE_OSD_SCRIPTS")) {
        default_ocr_fset.combine_detected_scripts = parameters["COMBINE_OSD_SCRIPTS"].toInt() > 0;
    }
    if (parameters.contains("MAX_OSD_SCRIPTS")) {
        default_ocr_fset.max_scripts = parameters["MAX_OSD_SCRIPTS"].toInt();
    }
    if (parameters.contains("MAX_TEXT_TRACKS")) {
        default_ocr_fset.max_text_tracks = parameters["MAX_TEXT_TRACKS"].toInt();
    }
    if (parameters.contains("MIN_OSD_SECONDARY_SCRIPT_THRESHOLD")) {
        default_ocr_fset.min_secondary_script_thrs = parameters["MIN_OSD_SECONDARY_SCRIPT_THRESHOLD"].toDouble();
    }
    if (parameters.contains("MIN_OSD_TEXT_ORIENTATION_CONFIDENCE")) {
        default_ocr_fset.min_orientation_confidence = parameters["MIN_OSD_TEXT_ORIENTATION_CONFIDENCE"].toDouble();
    }
    if (parameters.contains("MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE")) {
        default_ocr_fset.min_script_confidence = parameters["MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE"].toDouble();
    }
    if (parameters.contains("MIN_OSD_SCRIPT_SCORE")) {
        default_ocr_fset.min_script_score = parameters["MIN_OSD_SCRIPT_SCORE"].toDouble();
    }
    if (parameters.contains("ROTATE_AND_DETECT")) {
        default_ocr_fset.rotate_and_detect = parameters["ROTATE_AND_DETECT"].toInt() > 0;
    }
    if (parameters.contains("ROTATE_AND_DETECT_MIN_OCR_CONFIDENCE")) {
        default_ocr_fset.rotate_and_detect_min_confidence = parameters["ROTATE_AND_DETECT_MIN_OCR_CONFIDENCE"].toDouble();
    }
    if (parameters.contains("FULL_REGEX_SEARCH")) {
        default_ocr_fset.full_regex_search = parameters["FULL_REGEX_SEARCH"].toInt() > 0;
    }
    if (parameters.contains("ENABLE_OSD_FALLBACK")) {
        default_ocr_fset.enable_osd_fallback = parameters["ENABLE_OSD_FALLBACK"].toInt() > 0;
    }
    if (parameters.contains("MAX_PARALLEL_THREADS_PER_OCR_SCRIPT")) {
        default_ocr_fset.max_parallel_ocr_threads = parameters["MAX_PARALLEL_THREADS_PER_OCR_SCRIPT"].toInt();
    }
    if (parameters.contains("MAX_PARALLEL_THREADS_PER_PDF_RUN")) {
            default_ocr_fset.max_parallel_pdf_threads = parameters["MAX_PARALLEL_THREADS_PER_PDF_RUN"].toInt();
    }
}

/*
 * Image preprocessing to improve text extraction.
 * Sharpens image.
 */
void TesseractOCRTextDetection::sharpen(cv::Mat &image, double weight = 1.0) {
    cv::Mat blurred, mask;
    cv::blur(image, blurred, cv::Size(2, 2));
    cv::threshold(blurred, mask, 48, 1, cv::THRESH_BINARY);
    cv::multiply(blurred, mask, blurred);
    cv::addWeighted(image, 1.0 + weight, blurred, -1.0, 0, image);
}

/*
 * Helper function for string processing.
 */
inline wstring to_lowercase(const wstring &data) {
    wstring d2(data);
    d2 = boost::locale::normalize(d2);
    d2 = boost::locale::to_lower(d2);
    return d2;
}

/*
 * Helper function for string processing.
 */
inline wstring trim_punc(const wstring &in) {
    wstring d2(in);
    boost::trim_if(d2, [](wchar_t c) { return iswpunct(c); });
    return d2;
}

/*
 * Helper function for string processing.
 */
wstring clean_whitespace(const wstring &input) {

    boost::wregex re(L"\n(\n|[[:space:]])+");
    boost::wregex re2(L"\\\\n(\\\\n|[[:space:]])+");
    wstring result = boost::regex_replace(input, re, L"\n");
    wstring result2 = boost::regex_replace(result, re2, L"\\\\n");
    result2 = boost::trim_copy(result2);
    return result2;
}

/*
 * Helper function for language input processing.
 * Removes trailing whitespace and duplicate language entries.
 */
string clean_lang(const string &input) {
    string trimmed_input = boost::trim_copy(input);
    unique_vector language_list;
    vector<string> languages;

    boost::algorithm::split(languages, trimmed_input, boost::algorithm::is_any_of("+"));

    for (const string &lang : languages) {
        trimmed_input = boost::trim_copy(lang);
        language_list.insert(language_list.end(), trimmed_input);
    }

    return boost::algorithm::join(language_list, "+");
}

/*
 * Helper function for language input processing.
 * Generates a set of input tesseract scripts.
 */
set<string> generate_lang_set(const string input) {
    vector<string> lang_list;
    set<string> output_set;
    boost::algorithm::split(lang_list, input, boost::algorithm::is_any_of(","));

    for (const string &lang: lang_list) {
        string c_lang = clean_lang(lang);
        output_set.insert(c_lang);
    }
    return output_set;
}

/*
 * Reads JSON Tag filter file.
 * Setup tags for split-string and regex filters.
 */
map<wstring, vector<pair<wstring, bool>>>
TesseractOCRTextDetection::parse_json(const MPFJob &job, const string &jsonfile_path, MPFDetectionError &job_status) {
    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
    ifstream ifs(jsonfile_path);
    if (!ifs.is_open()) {
        LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Error reading JSON file at " + jsonfile_path);
        job_status = MPF_COULD_NOT_READ_DATAFILE;
        return json_kvs_regex;
    }
    string j;
    stringstream buffer2;
    wstring x;
    buffer2 << ifs.rdbuf();
    j = buffer2.str();
    x = boost::locale::conv::utf_to_utf<wchar_t>(j);
    JSONValue *value = JSON::Parse(x.c_str());
    if (value == NULL) {
        LOG4CXX_ERROR(hw_logger_,
                      "[" + job.job_name + "] JSON is corrupted. File location: " + jsonfile_path);
        job_status = MPF_COULD_NOT_READ_DATAFILE;
        return json_kvs_regex;
    }
    JSONObject root;
    root = value->AsObject();


    // REGEX TAG LOADING
    if (root.find(L"TAGS_BY_REGEX") != root.end() && root[L"TAGS_BY_REGEX"]->IsObject()) {
        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Regex tags found.");
        JSONObject key_tags = root[L"TAGS_BY_REGEX"]->AsObject();
        vector<wstring> keys = root[L"TAGS_BY_REGEX"]->ObjectKeys();
        vector<wstring>::iterator iter = keys.begin();

        while (iter != keys.end()) {
            auto term = *iter;
            wstring term_temp(term);

            if (!key_tags[term]->IsArray()) {
                LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Invalid JSON Array in TAGS_BY_REGEX!");
                job_status = MPF_COULD_NOT_READ_DATAFILE;
                // There was a processing error, but continue checking the remaining terms.
                iter++;
                continue;
            }

            JSONArray regex_array = key_tags[term]->AsArray();
            for (unsigned int i = 0; i < regex_array.size(); i++) {

                if (regex_array[i]->IsString()) {
                    // Legacy JSON processing.
                    // Legacy regex patterns in the JSON tags file are listed as follows:
                    //
                    // "TAGS_BY_REGEX": {
                    //    "vehicle-tag-legacy-format": [
                    //        "auto",
                    //        "car"
                    //    ],
                    //  ...
                    // }

                    wstring temp = regex_array[i]->AsString();
                    json_kvs_regex[term].push_back(make_pair(temp, false));
                } else if (regex_array[i]->IsObject()) {
                    // Standard JSON format processing.
                    // Standard JSON regex patterns are listed as follows:
                    //
                    // "TAGS_BY_REGEX": {
                    //    "vehicle-tag-standard-format": [
                    //      {"pattern": "auto"},
                    //      {"pattern": "car"}
                    //    ],
                    //  ...
                    //}
                    JSONObject regex_entry = regex_array[i]->AsObject();
                    if (regex_entry.find(L"pattern") != regex_entry.end()) {
                        wstring temp = regex_entry[L"pattern"]->AsString();
                        bool case_sens = false;
                        if (regex_entry.find(L"caseSensitive") != regex_entry.end()) {
                            case_sens = regex_entry[L"caseSensitive"]->AsBool();
                        }
                        json_kvs_regex[term].push_back(make_pair(temp, case_sens));
                    }
                }
            }
            iter++;
        }
    } else {
        LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] TAGS_BY_REGEX NOT FOUND.");
    }


    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] successfully read JSON.");
    return json_kvs_regex;
}

void TesseractOCRTextDetection::process_regex_match(const boost::wsmatch &match, const wstring &full_text,
                                map<wstring, vector<string>> &trigger_words_offset) {
    // Find and return matching pattern.
    int start = match.position(0Lu);
    int end = match.position(0Lu) + match[0].length();

    // Trim trigger words.
    int trim_start = start, trim_end = end;
    while (trim_start < end && iswspace(full_text.at(trim_start))) {
        trim_start++;
    }
    if (trim_start != end) {
        while (start < trim_end && iswspace(full_text.at(trim_end - 1))) {
            trim_end--;
        }
    }
    start = trim_start;
    end = trim_end;

    wstring trigger_word = full_text.substr(start , end - start);
    boost::replace_all(trigger_word, ";", "[;]");
    if (!(trigger_words_offset.count(trigger_word))) {
        vector<string> offsets;
        if (start != (end - 1)) {
            // Set offset for trigger word or phrase.
            offsets.push_back(to_string(start) + "-" + to_string(end - 1));
        } else {
            // Set offset for single character trigger.
            offsets.push_back(to_string(start));
        }
        trigger_words_offset.insert({trigger_word, offsets});
    } else {
        vector<string> &offsets = trigger_words_offset.at(trigger_word);
        string offset;
        if (start != (end - 1)) {
            // Set offset for trigger word or phrase.
            offset = to_string(start) + "-" + to_string(end - 1);
        } else {
            // Set offset for single character trigger.
            offset = to_string(start);
        }
        if (std::find(offsets.begin(), offsets.end(), offset) == offsets.end()) {
            offsets.push_back(offset);
        }
    }

}

/*
 * Check if detection string contains regstr pattern.
 */
bool TesseractOCRTextDetection::comp_regex(const MPFImageJob &job, const wstring &full_text,
                                           const wstring &regstr, map<wstring, vector<string>> &trigger_words_offset,
                                           const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                           bool case_sensitive, MPFDetectionError &job_status) {

    bool found = false;
    try {

        boost::wregex reg_matcher;
        if (case_sensitive) {
            reg_matcher = boost::wregex(regstr, boost::regex_constants::extended);
        } else {
            reg_matcher = boost::wregex(regstr, boost::regex_constants::extended | boost::regex::icase);
        }

        boost::wsmatch m;

        if (ocr_fset.full_regex_search) {
            boost::wsregex_iterator iter(full_text.begin(), full_text.end(), reg_matcher);
            boost::wsregex_iterator end;

            for( iter; iter != end; ++iter ) {
                process_regex_match(*iter, full_text, trigger_words_offset);
                found = true;
            }
        }
        else if (boost::regex_search(full_text, m, reg_matcher)) {
            process_regex_match(m, full_text, trigger_words_offset);
            found = true;
        }
    } catch (const boost::regex_error &e) {
        stringstream ss;
        ss << "[" + job.job_name + "] regex_error caught: " << parse_regex_code(e.code()) << ": " << e.what() << '\n';
        LOG4CXX_ERROR(hw_logger_, ss.str());
        job_status = MPF_COULD_NOT_READ_DATAFILE;
    }
    return found;
}

/*
 * Regex error handling.
 */
string TesseractOCRTextDetection::parse_regex_code(boost::regex_constants::error_type etype) {
    switch (etype) {
        case boost::regex_constants::error_collate:
            return "error_collate: invalid collating element request";
        case boost::regex_constants::error_ctype:
            return "error_ctype: invalid character class";
        case boost::regex_constants::error_escape:
            return "error_escape: invalid escape character or trailing escape";
        case boost::regex_constants::error_backref:
            return "error_backref: invalid back reference";
        case boost::regex_constants::error_brack:
            return "error_brack: mismatched bracket([ or ])";
        case boost::regex_constants::error_paren:
            return "error_paren: mismatched parentheses(( or ))";
        case boost::regex_constants::error_brace:
            return "error_brace: mismatched brace({ or })";
        case boost::regex_constants::error_badbrace:
            return "error_badbrace: invalid range inside a { }";
        case boost::regex_constants::error_range:
            return "erro_range: invalid character range(e.g., [z-a])";
        case boost::regex_constants::error_space:
            return "error_space: insufficient memory to handle this regular expression";
        case boost::regex_constants::error_badrepeat:
            return "error_badrepeat: a repetition character (*, ?, +, or {) was not preceded by a valid regular expression";
        case boost::regex_constants::error_complexity:
            return "error_complexity: the requested match is too complex";
        case boost::regex_constants::error_stack:
            return "error_stack: insufficient memory to evaluate a match";
        default:
            return "";
    }
}

/*
 * Generate a random alphanumeric string of specified length.
 * Used to generate image placeholders.
 */
string random_string(size_t length) {
    static auto &chrs = "0123456789"
                        "abcdefghijklmnopqrstuvwxyz"
                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    thread_local static mt19937 rg{random_device{}()};
    thread_local static uniform_int_distribution<string::size_type> pick(0, sizeof(chrs) - 2);

    string s;

    s.reserve(length);

    while (length--) {
        s += chrs[pick(rg)];
    }
    ostringstream stp;
    stp << ::getpid();
    return s + stp.str();
}

/*
 * Preprocess image before running OSD and OCR.
 */
bool TesseractOCRTextDetection::preprocess_image(const MPFImageJob &job, cv::Mat &image_data,
                                                 const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                 MPFDetectionError &job_status) {
    if (image_data.empty()) {
        LOG4CXX_ERROR(hw_logger_,
                      "[" + job.job_name + "] Could not open transformed image and will not return detections");
        job_status = MPF_IMAGE_READ_ERROR;
        return false;
    } else {
        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Transformed image opened.");
    }
    // Image preprocessing to improve text extraction results.

    // Image histogram equalization
    if (ocr_fset.enable_adaptive_hist_equalization) {
        cv::cvtColor(image_data, image_data, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->setTilesGridSize(cv::Size(ocr_fset.adaptive_hist_tile_size, ocr_fset.adaptive_hist_tile_size));
        clahe->setClipLimit(ocr_fset.adaptive_hist_clip_limit);
        clahe->apply(image_data, image_data);
    } else if (ocr_fset.enable_hist_equalization) {
        cv::cvtColor(image_data, image_data, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(image_data, image_data);
    }

    // Image thresholding.
    if (ocr_fset.enable_otsu_thrs || ocr_fset.enable_adaptive_thrs) {
        cv::cvtColor(image_data, image_data, cv::COLOR_BGR2GRAY);
    }
    if (ocr_fset.enable_adaptive_thrs) {
        // Pixel blocksize ranges 5-51 worked for adaptive threshold.
        cv::adaptiveThreshold(image_data, image_data, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,
                              ocr_fset.adaptive_thrs_pixel, ocr_fset.adaptive_thrs_c);
    } else if (ocr_fset.enable_otsu_thrs) {
        double thresh_val = cv::threshold(image_data, image_data, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    }

    // Rescale and sharpen image (larger images improve detection results).
    if (ocr_fset.scale > 0) {
        cv::resize(image_data, image_data, cv::Size(), ocr_fset.scale, ocr_fset.scale);
    }

    if (ocr_fset.min_height > 0 && image_data.rows < ocr_fset.min_height) {
        double ratio = (double)ocr_fset.min_height / (double)image_data.rows;
        cv::resize(image_data, image_data, cv::Size(), ratio, ratio);
    }

    if (ocr_fset.sharpen > 0) {
        sharpen(image_data, ocr_fset.sharpen);
    }

    // Image inversion.
    if (ocr_fset.invert) {
        double min, max;
        cv::Mat tmp_imb(image_data.size(), image_data.type());
        cv::minMaxLoc(image_data, &min, &max);
        tmp_imb.setTo(cv::Scalar::all(max));
        cv::subtract(tmp_imb, image_data, image_data);
    }

    return true;
}

void TesseractOCRTextDetection::process_tesseract_lang_model(const string &lang, const cv::Mat &imi, const string &job_name,
                                                            MPFDetectionError &job_status, const string &tessdata_script_dir,
                                                            const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                            string &text_result, double &confidence, bool &success, bool parallel_processing) {
    // Process language specified by user or script detected by OSD processing.
    success = false;
    pair<int, string> tess_api_key = make_pair(ocr_fset.oem, lang);
    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Running Tesseract with specified language: " + lang);
    tesseract::TessBaseAPI *tess_api;

    // When parallel processing is enabled, new APIs must be started up to avoid deadlocking issues rather than
    // using globally initialized APIs.
    if (tess_api_map.find(tess_api_key) == tess_api_map.end() || parallel_processing) {

        // Confirm each language model is present in tessdata or shared model directory.
        // Language models that run together must be present in the same directory.
        string tessdata_dir = "";
        if (tessdata_script_dir == "") {
            // Left blank when OSD is not run or scripts not found and reverted to default language.
            // Check default language model is present.
            tessdata_dir = return_valid_tessdir(job_name, lang, ocr_fset.model_dir);
        } else {
            tessdata_dir = tessdata_script_dir;
        }

        if (tessdata_dir == "") {
            LOG4CXX_ERROR(hw_logger_, "[" + job_name + "] Tesseract language models not found. Please add the " +
                                      "associated *.traineddata files to your tessdata directory " +
                                      "($MPF_HOME/plugins/TesseractOCRTextDetection/tessdata) " +
                                      "or shared models directory " +
                                      "($MPF_HOME/share/models/TesseractOCRTextDetection/tessdata).");
            job_status = MPF_COULD_NOT_OPEN_DATAFILE;
            return;
        }

        tesseract::TessBaseAPI *new_tess_api = new tesseract::TessBaseAPI();
        int init_rc = new_tess_api->Init(tessdata_dir.c_str(), lang.c_str(), (tesseract::OcrEngineMode) ocr_fset.oem);

        if (init_rc != 0) {
            LOG4CXX_ERROR(hw_logger_, "[" + job_name + "] Failed to initialize Tesseract! Error code: " +
                                      to_string(init_rc));
            job_status = MPF_DETECTION_NOT_INITIALIZED;
            return;
        }

        if (!parallel_processing) {
            tess_api_map[tess_api_key] = new_tess_api;
        }
        tess_api = new_tess_api;
    }


    if (!parallel_processing) {
        tess_api = tess_api_map[tess_api_key];
    }

    tess_api->SetPageSegMode((tesseract::PageSegMode) ocr_fset.psm);
    tess_api->SetImage(imi.data, imi.cols, imi.rows, imi.channels(), static_cast<int>(imi.step));
    unique_ptr<char[]> text{tess_api->GetUTF8Text()};
    confidence = tess_api->MeanTextConf();

    // Free up recognition results and any stored image data.
    tess_api->Clear();

    if (parallel_processing) {
        // Delete API if specified.
        tess_api->End();
    }
    text_result = text.get();
    success = true;
}


/*
 * Run tesseract ocr on refined image.
 */
void TesseractOCRTextDetection::get_tesseract_detections(const string &job_name,
                                                         vector<TesseractOCRTextDetection::OCR_output> &detections_by_lang,
                                                         cv::Mat &imi,
                                                         const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                         const string &lang,
                                                         MPFDetectionError &job_status,
                                                         const string &tessdata_script_dir,
                                                         bool &process_status,
                                                         bool process_pdf) {
    vector<string> results;
    boost::algorithm::split(results, job_name, boost::algorithm::is_any_of(" :"));

    set<string> ocr_lang_inputs = generate_lang_set(lang);
    process_status = false;

    // Enable multithreading for multiple tracks.
    if (ocr_fset.max_parallel_ocr_threads > 1 && ocr_lang_inputs.size() > 1 && !process_pdf) {
        tuple<string, double, bool, MPFDetectionError, string> ocr_results[ocr_lang_inputs.size()];
        thread ocr_threads[ocr_lang_inputs.size()];
        int index = 0, threads_joined = 0, wait_counter = 0;
        // Initialize a new track for each language specified.

        for (const string &lang: ocr_lang_inputs) {

            ocr_results[index] = make_tuple("", 0.0, false, job_status, lang);
            ocr_threads[index] = thread(&TesseractOCRTextDetection::process_tesseract_lang_model, this, std::cref(get<4>(ocr_results[index])),
                                        std::cref(imi), std::cref(job_name), std::ref(get<3>(ocr_results[index])), std::cref(tessdata_script_dir),
                                        std::cref(ocr_fset), std::ref(get<0>(ocr_results[index])), std::ref(get<1>(ocr_results[index])),
                                        std::ref(get<2>(ocr_results[index])), true);
            index++;
            wait_counter++;
            if (wait_counter == ocr_fset.max_parallel_ocr_threads) {
                wait_counter = 0;
                threads_joined = index;
                // Freeze generation of new threads until processing is complete with current batch.
                int p_index = 1;
                for (p_index; p_index <= ocr_fset.max_parallel_ocr_threads; p_index ++) {
                    ocr_threads[index - p_index].join();
                }
            }
        }


        // Wait for all remaining threads to finish.
        for (int i = threads_joined; i < ocr_lang_inputs.size(); i++){
            ocr_threads[i].join();
        }

        for (const tuple<string, double, bool, MPFDetectionError, string> &ocr_result: ocr_results) {
            string lang = get<4>(ocr_result);
            string text_result = get<0>(ocr_result);
            double confidence = get<1>(ocr_result);
            bool success = get<2>(ocr_result);
            job_status = get<3>(ocr_result);

            if (!success) {
                process_status = false;
                return;
            }
            LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Tesseract run successful.");
            wstring t_detection = boost::locale::conv::utf_to_utf<wchar_t>(text_result);

            TesseractOCRTextDetection::OCR_output output_ocr = {confidence, lang, t_detection, "", false};
            detections_by_lang.push_back(output_ocr);
        }
    } else {

        // Serial ocr or single track processing.
        for (const string &lang: ocr_lang_inputs) {
            double confidence;
            string text_result;
            bool success;
            if (ocr_fset.max_parallel_pdf_threads > 1 && process_pdf) {
                process_tesseract_lang_model(lang, imi, job_name, job_status, tessdata_script_dir, ocr_fset,
                                             text_result, confidence, success, true);
            } else {
                process_tesseract_lang_model(lang, imi, job_name, job_status, tessdata_script_dir, ocr_fset,
                                        text_result, confidence, success, false);
            }
            if (!success) {
                process_status = false;
                return;
            }

            LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Tesseract run successful.");
            wstring t_detection = boost::locale::conv::utf_to_utf<wchar_t>(text_result);

            TesseractOCRTextDetection::OCR_output output_ocr = {confidence, lang, t_detection, "", false};
            detections_by_lang.push_back(output_ocr);
        }
    }
    process_status = true;
    return;
}

string TesseractOCRTextDetection::return_valid_tessdir(const string &job_name, const string &lang_str,
                                                       const string &directory) {

    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Checking tessdata models in " + directory + ".");

    if (check_tess_model_directory(job_name, lang_str, directory)) {
        return directory;
    }

    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }

    string local_plugin_directory = run_dir + "/TesseractOCRTextDetection/tessdata";
    LOG4CXX_DEBUG(hw_logger_,
                  "[" + job_name + "] Not all models found in " + directory + ". Checking local plugin directory "
                  + local_plugin_directory + ".");

    if (check_tess_model_directory(job_name, lang_str, local_plugin_directory)) {
        return local_plugin_directory;
    }

    LOG4CXX_WARN(hw_logger_, "[" + job_name + "] Some Tessdata models were not found, or are not co-located." +
                             " Please ensure that all of the following models exist in the same directory, either " +
                             directory + " or " +
                             local_plugin_directory + ": " + lang_str + " .");
    return "";
}

bool TesseractOCRTextDetection::check_tess_model_directory(const string &job_name, const string &lang_str,
                                                           const string &directory) {

    vector<string> langs;
    boost::split(langs, lang_str, boost::is_any_of(",+"));

    for (string lang : langs) {
        boost::trim(lang);
        // Don't search for invalid NULL language.
        if (lang == "NULL" || lang == "script/NULL") {
            continue;
        }
        if (!boost::filesystem::exists(directory + "/" + lang + ".traineddata")) {
            LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Tessdata file " + lang + ".traineddata does not exist in "
                                      + directory + ".");
            return false;
        }
        LOG4CXX_DEBUG(hw_logger_,
                      "[" + job_name + "] Tessdata file " + lang + ".traineddata found in  " + directory + ".");
    }
    LOG4CXX_INFO(hw_logger_,
                 "[" + job_name + "] All tessdata models found in " + directory + ": " + lang_str + ".");
    return true;
}

void TesseractOCRTextDetection::get_OSD(OSResults &results, cv::Mat &imi, const MPFImageJob &job,
                                        TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                        Properties &detection_properties, MPFDetectionError &job_status,
                                        string &tessdata_script_dir) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Running Tesseract OSD.");

    int oem = 3;
    pair<int, string> tess_api_key = make_pair(oem, "osd");

    if (tess_api_map.find(tess_api_key) == tess_api_map.end()) {
        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Loading OSD model.");

        string tessdata_dir = return_valid_tessdir(job.job_name, "osd", ocr_fset.model_dir);
        if (tessdata_dir == "") {
            LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] OSD model not found!");
            job_status = MPF_COULD_NOT_OPEN_DATAFILE;
            return;
        }

        tesseract::TessBaseAPI *tess_api = new tesseract::TessBaseAPI();
        int init_rc = tess_api->Init(tessdata_dir.c_str(), "osd", (tesseract::OcrEngineMode) oem);

        if (init_rc != 0) {
            LOG4CXX_ERROR(hw_logger_,
                          "[" + job.job_name + "] Failed to initialize Tesseract! Error code: " + to_string(init_rc));
            job_status = MPF_DETECTION_NOT_INITIALIZED;
            return;
        }

        tess_api_map[tess_api_key] = tess_api;

        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] OSD model ready.");
    }

    tesseract::TessBaseAPI *tess_api = tess_api_map[tess_api_key];
    tess_api->SetPageSegMode(tesseract::PSM_AUTO_OSD);
    tess_api->SetImage(imi.data, imi.cols, imi.rows, imi.channels(), static_cast<int>(imi.step));
    tess_api->DetectOS(&results);

    // Free up recognition results and any stored image data.
    tess_api->Clear();

    string initial_script = results.unicharset->get_script_from_script_id(results.best_result.script_id);
    string osd_fallback_occurred = "false";
    if (ocr_fset.enable_osd_fallback && (ocr_fset.min_orientation_confidence > results.best_result.oconfidence ||
        initial_script == "NULL")) {
        osd_fallback_occurred = "true";

        // Replicate the image, four times across.
        cv::Mat amplified_im = cv::repeat(imi, 1, 4);
        tess_api->SetPageSegMode(tesseract::PSM_AUTO_OSD);
        tess_api->SetImage(amplified_im.data, amplified_im.cols, amplified_im.rows, amplified_im.channels(),
                           static_cast<int>(amplified_im.step));
        tess_api->DetectOS(&results);

        // Free up recognition results and any stored image data.
        tess_api->Clear();
    }

    int best_ori = results.best_result.orientation_id;
    int best_id = results.best_result.script_id;
    int candidates = 0;
    double best_score = results.scripts_na[best_ori][best_id];
    int rotation = 0;

    if (ocr_fset.min_orientation_confidence <= results.best_result.oconfidence) {
        switch (results.best_result.orientation_id) {
            case 0:
                // Do not rotate.
                break;
            case 1:
                // Text is rotated 270 degrees counterclockwise.
                // Image will need to be rotated 90 degrees counterclockwise to fix this.
                cv::rotate(imi, imi, cv::ROTATE_90_COUNTERCLOCKWISE);

                // Report current orientation of image (uncorrected, counterclockwise).
                rotation = 270;
                break;
            case 2:
                // 180 degree rotation.
                cv::rotate(imi, imi, cv::ROTATE_180);
                rotation = 180;
                break;
            case 3:
                // Text is rotated 90 degrees counterclockwise.
                // Image will need to be rotated 270 degrees counterclockwise to fix this.
                cv::rotate(imi, imi, cv::ROTATE_90_CLOCKWISE);

                // Report current orientation of image (uncorrected, counterclockwise).
                rotation = 90;
                break;
            default:
                break;
        }
    }

    // Store OSD results.
    detection_properties["OSD_PRIMARY_SCRIPT"] = results.unicharset->get_script_from_script_id(
            results.best_result.script_id);
    detection_properties["OSD_PRIMARY_SCRIPT_CONFIDENCE"] = to_string(results.best_result.sconfidence);
    detection_properties["OSD_PRIMARY_SCRIPT_SCORE"] = to_string(best_score);
    detection_properties["ROTATION"] = to_string(rotation);
    detection_properties["OSD_TEXT_ORIENTATION_CONFIDENCE"] = to_string(results.best_result.oconfidence);
    detection_properties["OSD_FALLBACK_OCCURRED"] = osd_fallback_occurred;

    if (detection_properties["OSD_PRIMARY_SCRIPT"] == "NULL") {
        //If OSD failed to detect any scripts, automatically set confidence scores to -1 (not enough text).
        detection_properties["OSD_PRIMARY_SCRIPT_CONFIDENCE"] = "-1";
        detection_properties["OSD_TEXT_ORIENTATION_CONFIDENCE"] = "-1";
        detection_properties["OSD_PRIMARY_SCRIPT_SCORE"]  = "-1";
    }

    int max_scripts = ocr_fset.max_scripts;
    if (detection_properties["OSD_PRIMARY_SCRIPT"] == "Common") {
        // When "Common" is detected, swap over to secondary scripts for consideration.
        // Common script is automatically excluded from model selection, but still reported in the results as the top
        // script.
        if (max_scripts > 0) {
            max_scripts = max_scripts + 1;
        }
    }

    if (ocr_fset.min_script_confidence <= results.best_result.sconfidence && ocr_fset.min_script_score <= best_score) {

        TesseractOCRTextDetection::OSD_script best_script = {best_id, best_score};
        vector<TesseractOCRTextDetection::OSD_script> script_list;

        // If primary script is not NULL, check if secondary scripts are also valid.
        if (max_scripts != 1 && detection_properties["PRIMARY_SCRIPT"] != "NULL") {
            // Max number of scripts in ICU + "NULL" + Japanese and Korean + Fraktur
            const int kMaxNumberOfScripts = 116 + 1 + 2 + 1;
            double score_cutoff = best_score * ocr_fset.min_secondary_script_thrs;
            if (score_cutoff < ocr_fset.min_script_score) {
                score_cutoff = ocr_fset.min_script_score;
            }
            // Skip index 0 to ignore the "Common" script.
            for (int i = 1; i < kMaxNumberOfScripts; ++i) {
                double score = results.scripts_na[best_ori][i];
                if (i != best_id && score >= score_cutoff) {
                    TesseractOCRTextDetection::OSD_script script_out;
                    script_out.id = i;
                    script_out.score = score;
                    script_list.push_back(script_out);
                    candidates += 1;
                }

            }
            sort(script_list.begin(), script_list.end(), greater<TesseractOCRTextDetection::OSD_script>());
            // Limit number of accepted scripts if user set max_scripts to 2 or greater.
            // Unlimited number when users sets max_scripts to 0 or below.
            if (max_scripts > 1 && candidates > max_scripts - 1) {
                script_list.resize(max_scripts - 1);
            }
        }

        // Store OSD results.
        if (script_list.size() > 0) {

            vector<string> scripts;
            vector<string> scores;

            for (const TesseractOCRTextDetection::OSD_script &script : script_list) {
                scripts.push_back(results.unicharset->get_script_from_script_id(script.id));
                scores.push_back(to_string(script.score));
            }

            detection_properties["OSD_SECONDARY_SCRIPTS"] = boost::algorithm::join(scripts, ", ");
            detection_properties["OSD_SECONDARY_SCRIPT_SCORES"] = boost::algorithm::join(scores, ", ");
        }

        // Include best script result and move it to the front.
        script_list.push_back(best_script);
        rotate(script_list.rbegin(), script_list.rbegin() + 1, script_list.rend());

        bool added_japanese = false;
        bool added_korean = false;
        vector<string> script_results;
        for (const TesseractOCRTextDetection::OSD_script &script : script_list) {
            string script_type = results.unicharset->get_script_from_script_id(script.id);
            // For scripts that support vertical text, run both horizontal and vertical language models.
            if (script_type == "Han") {

                if (ocr_fset.combine_detected_scripts) {
                    script_results.push_back("script/HanS+script/HanT+script/HanS_vert+script/HanT_vert");
                } else {
                    script_results.push_back("script/HanS+script/HanT,script/HanS_vert+script/HanT_vert");
                }

            } else if (script_type == "Korean" || script_type == "Hangul") {
                if (added_korean) {
                    continue;
                }
                // Don't add the same language multiple times.
                added_korean = true;
                if (ocr_fset.combine_detected_scripts) {
                    script_results.push_back("script/Hangul+script/Hangul_vert");
                } else {
                    script_results.push_back("script/Hangul,script/Hangul_vert");
                }

            } else if ((script_type == "Japanese" || script_type == "Hiragana" || script_type == "Katakana")) {
                if (added_japanese) {
                    continue;
                }
                // Don't add the same language multiple times.
                added_japanese = true;
                if (ocr_fset.combine_detected_scripts) {
                    script_results.push_back("script/Japanese+script/Japanese_vert");
                } else {
                    script_results.push_back("script/Japanese,script/Japanese_vert");
                }
            } else if (script_type == "Common" || script_type == "NULL") {
                continue;
            } else {
                script_results.push_back("script/" + script_type);
            }
        }

        string lang_str;
        if (ocr_fset.combine_detected_scripts) {
            lang_str = boost::algorithm::join(script_results, "+");
        } else {
            lang_str = boost::algorithm::join(script_results, ",");
        }

        if (lang_str.empty() || lang_str == "script/NULL") {
            LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] OSD did not detect any valid scripts,"
                                     + " reverting to default language setting: " + ocr_fset.tesseract_lang);
        } else {
            // Check if selected models are present in either models or tessdata directory.
            // All language models must be present in one directory.
            // If scripts are not found, revert to default.
            // If scripts are found, set return_valid_tessdir so that get_tesseract_detections will skip searching for models.
            // Also, for script detection results, skip searching for NULL if that gets detected alongside other languages.
            tessdata_script_dir = TesseractOCRTextDetection::return_valid_tessdir(job.job_name, lang_str, ocr_fset.model_dir);
            if (tessdata_script_dir == "") {
                LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] Script models not found in model and tessdata directories,"
                                         + " reverting to default language setting: " + ocr_fset.tesseract_lang);
            } else {
                ocr_fset.tesseract_lang = lang_str;
            }
        }
    }

}

bool is_only_ascii_whitespace(const wstring &str) {
    auto it = str.begin();
    do {
        if (it == str.end()) {
            return true;
        }
    } while (*it >= 0 && *it <= 0x7f && iswspace(*(it++)));
    // One of these conditions will be optimized away by the compiler,
    // which one depends on whether char is signed or not
    return false;
}


/*
 * Performs regex-tagging of ocr text detection.
 */
set<wstring> TesseractOCRTextDetection::search_regex(const MPFImageJob &job, const wstring &full_text,
                                                     const map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex,
                                                     map<wstring, vector<string>> &trigger_words_offset,
                                                     const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                     MPFDetectionError &job_status) {
    wstring found_tags_regex = L"";
    set<wstring> found_keys_regex;

    if (json_kvs_regex.size() == 0) {
        return found_keys_regex;
    }

    for (const auto &kv : json_kvs_regex) {
        auto key = kv.first;
        auto values = kv.second;
        for (const pair<wstring, bool> &value : values) {
            wstring regex_pattern = value.first;
            bool case_sens = value.second;

            if (comp_regex(job, full_text, regex_pattern, trigger_words_offset,
                ocr_fset, case_sens, job_status)) {
                found_keys_regex.insert(key);
                // Discontinue searching unless full regex search is enabled.
                if (!ocr_fset.full_regex_search) {
                    break;
                }
            }
        }
    }
    int num_found = found_keys_regex.size();
    found_tags_regex = boost::algorithm::join(found_keys_regex, L", ");
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Done searching for regex tags, found: " + to_string(num_found));
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Found regex tags are: " +
                              boost::locale::conv::utf_to_utf<char>(found_tags_regex));

    return found_keys_regex;
}


void TesseractOCRTextDetection::load_tags_json(const MPFJob &job, MPFDetectionError &job_status,
                                               map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex) {

    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/TesseractOCRTextDetection";
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Running from directory " + plugin_path);

    string jsonfile_path = DetectionComponentUtils::GetProperty<string>(job.job_properties, "TAGGING_FILE",
                                                                        "text-tags.json");
    if (jsonfile_path.find('$') != string::npos || jsonfile_path.find('/') != string::npos) {
        string new_jsonfile_name = "";
        Utils::expandFileName(jsonfile_path, new_jsonfile_name);
        jsonfile_path = new_jsonfile_name;
    } else {
        jsonfile_path = plugin_path + "/config/" + jsonfile_path;
    }

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] About to read JSON from: " + jsonfile_path);
    json_kvs_regex = parse_json(job, jsonfile_path, job_status);
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Read JSON");
}

void
TesseractOCRTextDetection::load_settings(const MPFJob &job, TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                         const Text_type &text_type) {
    // Load in settings specified from job_properties and default configuration.

    // Image preprocessing
    bool default_processing_wild = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "UNSTRUCTURED_TEXT_ENABLE_PREPROCESSING", default_ocr_fset.processing_wild_text);

    if ((text_type == Unstructured) || (text_type == Unknown && default_processing_wild)) {
        ocr_fset.enable_adaptive_hist_equalization = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "UNSTRUCTURED_TEXT_ENABLE_ADAPTIVE_HIST_EQUALIZATION", default_ocr_fset.enable_adaptive_hist_equalization);
        ocr_fset.enable_hist_equalization = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "UNSTRUCTURED_TEXT_ENABLE_HIST_EQUALIZATION",  default_ocr_fset.enable_hist_equalization);
        ocr_fset.scale = DetectionComponentUtils::GetProperty<double>(job.job_properties,"UNSTRUCTURED_TEXT_SCALE", default_ocr_fset.scale);
        ocr_fset.enable_adaptive_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"UNSTRUCTURED_TEXT_ENABLE_ADAPTIVE_THRS", default_ocr_fset.enable_adaptive_thrs);
        ocr_fset.enable_otsu_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"UNSTRUCTURED_TEXT_ENABLE_OTSU_THRS", default_ocr_fset.enable_otsu_thrs);
        ocr_fset.sharpen = DetectionComponentUtils::GetProperty<double>(job.job_properties,"UNSTRUCTURED_TEXT_SHARPEN", default_ocr_fset.sharpen);
    } else {
        ocr_fset.enable_adaptive_hist_equalization = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "STRUCTURED_TEXT_ENABLE_ADAPTIVE_HIST_EQUALIZATION", default_ocr_fset.enable_adaptive_hist_equalization);
        ocr_fset.enable_hist_equalization = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "STRUCTURED_TEXT_ENABLE_HIST_EQUALIZATION",  default_ocr_fset.enable_hist_equalization);
        ocr_fset.scale = DetectionComponentUtils::GetProperty<double>(job.job_properties,"STRUCTURED_TEXT_SCALE", default_ocr_fset.scale);
        ocr_fset.enable_adaptive_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"STRUCTURED_TEXT_ENABLE_ADAPTIVE_THRS", default_ocr_fset.enable_adaptive_thrs);
        ocr_fset.enable_otsu_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"STRUCTURED_TEXT_ENABLE_OTSU_THRS", default_ocr_fset.enable_otsu_thrs);
        ocr_fset.sharpen = DetectionComponentUtils::GetProperty<double>(job.job_properties,"STRUCTURED_TEXT_SHARPEN", default_ocr_fset.sharpen);
    }

    ocr_fset.invert = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"INVERT", default_ocr_fset.invert);
    ocr_fset.min_height = DetectionComponentUtils::GetProperty<int>(job.job_properties, "MIN_HEIGHT", default_ocr_fset.min_height);
    ocr_fset.adaptive_hist_tile_size = DetectionComponentUtils::GetProperty<int>(job.job_properties, "ADAPTIVE_HIST_TILE_SIZE", default_ocr_fset.adaptive_hist_tile_size);
    ocr_fset.adaptive_hist_clip_limit = DetectionComponentUtils::GetProperty<int>(job.job_properties, "ADAPTIVE_HIST_CLIP_LIMIT", default_ocr_fset.adaptive_hist_clip_limit);
    ocr_fset.adaptive_thrs_c = DetectionComponentUtils::GetProperty<double>(job.job_properties,"ADAPTIVE_THRS_CONSTANT", default_ocr_fset.adaptive_thrs_c);
    ocr_fset.adaptive_thrs_pixel = DetectionComponentUtils::GetProperty<int>(job.job_properties,"ADAPTIVE_THRS_BLOCKSIZE", default_ocr_fset.adaptive_thrs_pixel);

    // OCR and OSD Engine Settings.
    ocr_fset.tesseract_lang  = DetectionComponentUtils::GetProperty<std::string>(job.job_properties,"TESSERACT_LANGUAGE", default_ocr_fset.tesseract_lang);
    ocr_fset.psm = DetectionComponentUtils::GetProperty<int>(job.job_properties,"TESSERACT_PSM", default_ocr_fset.psm);
    ocr_fset.oem = DetectionComponentUtils::GetProperty<int>(job.job_properties,"TESSERACT_OEM", default_ocr_fset.oem);

    // OSD Settings
    ocr_fset.enable_osd = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_OSD_AUTOMATION", default_ocr_fset.enable_osd);
    ocr_fset.combine_detected_scripts = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"COMBINE_OSD_SCRIPTS", default_ocr_fset.combine_detected_scripts);
    ocr_fset.min_orientation_confidence = DetectionComponentUtils::GetProperty<double>(job.job_properties,"MIN_OSD_TEXT_ORIENTATION_CONFIDENCE", default_ocr_fset.min_orientation_confidence);
    ocr_fset.min_script_confidence = DetectionComponentUtils::GetProperty<double>(job.job_properties,"MIN_OSD_PRIMARY_SCRIPT_CONFIDENCE", default_ocr_fset.min_script_confidence);
    ocr_fset.min_script_score = DetectionComponentUtils::GetProperty<double>(job.job_properties,"MIN_OSD_SCRIPT_SCORE", default_ocr_fset.min_script_score);
    ocr_fset.max_scripts = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MAX_OSD_SCRIPTS", default_ocr_fset.max_scripts);
    ocr_fset.max_text_tracks = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MAX_TEXT_TRACKS", default_ocr_fset.max_text_tracks);
    ocr_fset.min_secondary_script_thrs = DetectionComponentUtils::GetProperty<double>(job.job_properties,"MIN_OSD_SECONDARY_SCRIPT_THRESHOLD", default_ocr_fset.min_secondary_script_thrs);
    ocr_fset.rotate_and_detect = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ROTATE_AND_DETECT", default_ocr_fset.rotate_and_detect);
    ocr_fset.rotate_and_detect_min_confidence = DetectionComponentUtils::GetProperty<double>(job.job_properties, "ROTATE_AND_DETECT_MIN_OCR_CONFIDENCE", default_ocr_fset.rotate_and_detect_min_confidence);
    ocr_fset.full_regex_search = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"FULL_REGEX_SEARCH", default_ocr_fset.full_regex_search);
    ocr_fset.enable_osd_fallback = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_OSD_FALLBACK", default_ocr_fset.enable_osd_fallback);
    ocr_fset.max_parallel_ocr_threads = DetectionComponentUtils::GetProperty<int>(job.job_properties, "MAX_PARALLEL_THREADS_PER_OCR_SCRIPT", default_ocr_fset.max_parallel_ocr_threads);
    ocr_fset.max_parallel_pdf_threads = DetectionComponentUtils::GetProperty<int>(job.job_properties, "MAX_PARALLEL_THREADS_PER_PDF_RUN", default_ocr_fset.max_parallel_pdf_threads);

    // Tessdata setup
    ocr_fset.model_dir =  DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "MODELS_DIR_PATH", default_ocr_fset.model_dir);
    if (ocr_fset.model_dir != "") {
        ocr_fset.model_dir = ocr_fset.model_dir + "/TesseractOCRTextDetection/tessdata";
    } else {
        string run_dir = GetRunDirectory();
        if (run_dir.empty()) {
            run_dir = ".";
        }
        string plugin_path = run_dir + "/TesseractOCRTextDetection";
        // If not specified, set model dir to local plugin dir.
        ocr_fset.model_dir = plugin_path + "/tessdata";
    }
    Utils::expandFileName(ocr_fset.model_dir, ocr_fset.model_dir);
}

// Tag results and store into track detection properties.
bool TesseractOCRTextDetection::process_text_tagging(Properties &detection_properties, const MPFImageJob &job,
                                                     const TesseractOCRTextDetection::OCR_output &ocr_out,
                                                     MPFDetectionError &job_status,
                                                     const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                     const map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex,
                                                     int page_num) {

    string ocr_lang = ocr_out.language;
    wstring full_text = ocr_out.text;
    full_text = clean_whitespace(full_text);

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing tags for Tesseract OCR output: ");
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Tesseract OCR output was: " +
                              boost::locale::conv::utf_to_utf<char>(full_text));

    if (is_only_ascii_whitespace(full_text)) {
        LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] No text found in image!");
        return false;
    }

    set<wstring> trigger_words;
    map<wstring, vector<string>> trigger_words_offset;
    set<wstring> found_tags_regex = search_regex(job, full_text, json_kvs_regex, trigger_words_offset,
                                         ocr_fset, job_status);

    wstring tag_string = boost::algorithm::join(found_tags_regex, L"; ");
    vector<string> offsets_list;
    vector<wstring> triggers_list;
    wstring tag_trigger = boost::algorithm::join(trigger_words, L"; ");
    for (auto const& word_offset : trigger_words_offset )
    {
        triggers_list.push_back(word_offset.first);
        offsets_list.push_back(boost::algorithm::join(word_offset.second, ", "));
    }
    string tag_offset = boost::algorithm::join(offsets_list, "; ");
    tag_trigger = tag_trigger + boost::algorithm::join(triggers_list, L"; ");

    detection_properties["TEXT_LANGUAGE"] = ocr_lang;
    detection_properties["TAGS"] = boost::locale::conv::utf_to_utf<char>(tag_string);
    detection_properties["TRIGGER_WORDS"] = boost::locale::conv::utf_to_utf<char>(tag_trigger);
    detection_properties["TRIGGER_WORDS_OFFSET"] = tag_offset;
    detection_properties["TEXT"] = boost::locale::conv::utf_to_utf<char>(full_text);

    if (page_num >= 0) {
        detection_properties["PAGE_NUM"] = to_string(page_num + 1);
    }
    return true;

}

inline void set_coordinates(int &xLeftUpper, int &yLeftUpper, int &width, int &height, const cv::Size &input_size,
                            const int &orientation_id) {
    switch (orientation_id) {
        case 0:
            // Do not rotate.
            xLeftUpper = 0;
            yLeftUpper = 0;
            width = input_size.width;
            height = input_size.height;
            break;
        case 1:
            // Text is rotated 270 degrees counterclockwise.
            // Image will need to be rotated 90 degrees counterclockwise to fix this.
            xLeftUpper = input_size.width - 1;
            yLeftUpper = 0;
            width = input_size.height;
            height = input_size.width;
            break;
        case 2:
            // 180 degree rotation.
            xLeftUpper = input_size.width - 1;
            yLeftUpper = input_size.height - 1;
            width = input_size.width;
            height = input_size.height;
            break;
        case 3:
            // Text is rotated 90 degrees counterclockwise.
            // Image will be rotated 270 degrees counterclockwise.
            yLeftUpper = input_size.height - 1;
            xLeftUpper = 0;
            width = input_size.height;
            height = input_size.width;
            break;
        default:
            break;
    }
}

MPFDetectionError
TesseractOCRTextDetection::GetDetections(const MPFImageJob &job, vector<MPFImageLocation> &locations) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");
    try{
        TesseractOCRTextDetection::OCR_filter_settings ocr_fset;
        Text_type text_type = Unknown;

        if (job.has_feed_forward_location && job.feed_forward_location.detection_properties.count("TEXT_TYPE")) {
            if (job.feed_forward_location.detection_properties.at("TEXT_TYPE") == "UNSTRUCTURED") {
                text_type = Unstructured;
            } else if (job.feed_forward_location.detection_properties.at("TEXT_TYPE") == "STRUCTURED") {
                text_type = Structured;
            }

            LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Identified text type:  \"" +
                                      job.feed_forward_location.detection_properties.at("TEXT_TYPE") + "\".");
        }
        load_settings(job, ocr_fset, text_type);

        MPFDetectionError job_status = MPF_DETECTION_SUCCESS;
        map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
        load_tags_json(job, job_status, json_kvs_regex);

        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] About to run tesseract");
        vector<TesseractOCRTextDetection::OCR_output> ocr_outputs;

        MPFImageReader image_reader(job);
        cv::Mat image_data = image_reader.GetImage();
        cv::Mat image_data_rotated;
        cv::Size input_size = image_data.size();

        if (!preprocess_image(job, image_data, ocr_fset, job_status)) {
            return job_status;
        }

        Properties osd_detection_properties;
        string tessdata_script_dir = "";
        int xLeftUpper = 0;
        int yLeftUpper = 0;
        int width = input_size.width;
        int height = input_size.height;
        int orientation_result = 0;

        if (ocr_fset.psm == 0 || ocr_fset.enable_osd) {

            OSResults os_results;
            get_OSD(os_results, image_data, job, ocr_fset, osd_detection_properties, job_status,
                    tessdata_script_dir);

            // Rotate upper left coordinates based on OSD results.
            if (ocr_fset.min_orientation_confidence <= os_results.best_result.oconfidence) {
                orientation_result = os_results.best_result.orientation_id;
                set_coordinates(xLeftUpper, yLeftUpper, width, height, input_size, os_results.best_result.orientation_id);
            }

            // When PSM is set to 0, there is no need to process any further.
            if (ocr_fset.psm == 0) {
                LOG4CXX_INFO(hw_logger_,
                             "[" + job.job_name + "] Processing complete. Found " + to_string(locations.size()) +
                             " tracks.");
                MPFImageLocation osd_detection(xLeftUpper, yLeftUpper, width, height, -1, osd_detection_properties);
                locations.push_back(osd_detection);
                return job_status;
            }
        }

        set<string> missing_languages;
        string first_pass_rotation, second_pass_rotation;
        double min_ocr_conf = ocr_fset.rotate_and_detect_min_confidence;
        int corrected_orientation;
        int corrected_X, corrected_Y, corrected_width, corrected_height;

        if (ocr_fset.rotate_and_detect) {
            cv::rotate(image_data, image_data_rotated, cv::ROTATE_180);
            missing_languages = generate_lang_set(ocr_fset.tesseract_lang);
            double rotation_val = 0.0;
            if (osd_detection_properties.count("ROTATION")) {
                rotation_val = boost::lexical_cast<double>(osd_detection_properties["ROTATION"]);
            }
            first_pass_rotation = to_string(rotation_val);
            second_pass_rotation = to_string(180.0 + rotation_val);
            corrected_orientation = (orientation_result + 2) % 4;
            set_coordinates(corrected_X, corrected_Y, corrected_width, corrected_height, input_size, corrected_orientation);
        }

        // Run initial get_tesseract_detections. When autorotate is set, for any languages that fall below initial pass
        // create a second round of extractions with a 180 degree rotation applied on top of the original setting.
        // Second rotation only triggers if ROTATE_AND_DETECT is set.
        bool tesseract_success;
        get_tesseract_detections(job.job_name, ocr_outputs, image_data, ocr_fset, ocr_fset.tesseract_lang, job_status,
                                 tessdata_script_dir, tesseract_success);

        if (!tesseract_success) {
            LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Could not run tesseract!");
            return job_status;
        }

        vector<TesseractOCRTextDetection::OCR_output> all_results;

        for (const TesseractOCRTextDetection::OCR_output &ocr_out: ocr_outputs) {
            TesseractOCRTextDetection::OCR_output final_out = ocr_out;
            if (ocr_fset.rotate_and_detect) {
                missing_languages.erase(ocr_out.language);
                final_out.two_pass_rotation = first_pass_rotation;
                final_out.two_pass_correction = false;

                // Perform second pass OCR if min threshold is disabled (negative) or first pass confidence too low.
                if (min_ocr_conf <= 0 || ocr_out.confidence < min_ocr_conf) {
                    // Perform second pass OCR and provide best result to output.
                    vector<TesseractOCRTextDetection::OCR_output> ocr_outputs_rotated;
                    ocr_fset.tesseract_lang = ocr_out.language;
                    get_tesseract_detections(job.job_name, ocr_outputs_rotated, image_data_rotated, ocr_fset, ocr_fset.tesseract_lang,
                                             job_status, tessdata_script_dir, tesseract_success);
                    if (!tesseract_success) {
                            LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Could not run tesseract!");
                            return job_status;
                    }

                    TesseractOCRTextDetection::OCR_output  ocr_out_rotated = ocr_outputs_rotated.front();
                    if (ocr_out_rotated.confidence > ocr_out.confidence) {
                        final_out = ocr_out_rotated;
                        final_out.two_pass_rotation = second_pass_rotation;
                        final_out.two_pass_correction = true;
                    }
                }
            }
            all_results.push_back(final_out);
        }

        // If two stage OCR is enabled, run the second pass of OCR on any remaining languages where the first pass failed
        // to generate an output.
        for (const string &miss_lang: missing_languages) {
            // Perform second pass OCR and provide best result to output.
            vector<TesseractOCRTextDetection::OCR_output> ocr_outputs_rotated;
            ocr_fset.tesseract_lang = miss_lang;
            get_tesseract_detections(job.job_name, ocr_outputs_rotated, image_data_rotated, ocr_fset, ocr_fset.tesseract_lang,
                                     job_status, tessdata_script_dir, tesseract_success);
            if (!tesseract_success) {
                    LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Could not run tesseract!");
                    return job_status;
            }
            TesseractOCRTextDetection::OCR_output ocr_out_rotated = ocr_outputs_rotated.front();
            ocr_out_rotated.two_pass_rotation = second_pass_rotation;
            ocr_out_rotated.two_pass_correction = true;
            all_results.push_back(ocr_out_rotated);
        }

        // If max_text_tracks is set, filter out to return only the top specified tracks.
        if (ocr_fset.max_text_tracks > 0) {
            sort(all_results.begin(), all_results.end(), greater<TesseractOCRTextDetection::OCR_output>());
            all_results.resize(ocr_fset.max_text_tracks);
        }

        for (const TesseractOCRTextDetection::OCR_output &final_out : all_results) {
            MPFImageLocation image_location(xLeftUpper, yLeftUpper, width, height, final_out.confidence);
            // Copy over OSD detection results into OCR output.
            image_location.detection_properties = osd_detection_properties;

            // Mark two-pass OCR final selected rotation.
            if (ocr_fset.rotate_and_detect) {
                image_location.detection_properties["ROTATION"] = final_out.two_pass_rotation;
                if (final_out.two_pass_correction) {
                    image_location.detection_properties["ROTATE_AND_DETECT_PASS"] = "180";
                    // Also correct top left corner designation:
                    image_location.x_left_upper = corrected_X;
                    image_location.y_left_upper = corrected_Y;
                    image_location.width = corrected_width;
                    image_location.height = corrected_height;
                } else {
                    image_location.detection_properties["ROTATE_AND_DETECT_PASS"] = "0";
                }

            }
            bool process_text = process_text_tagging(image_location.detection_properties, job, final_out, job_status,
                                                     ocr_fset,
                                                     json_kvs_regex);
            if (process_text) {
                locations.push_back(image_location);
            }
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        LOG4CXX_INFO(hw_logger_,
                     "[" + job.job_name + "] Processing complete. Found " + to_string(locations.size()) + " tracks.");
        return job_status;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, hw_logger_);
    }
}

MPFDetectionError TesseractOCRTextDetection::GetDetections(const MPFGenericJob &job, vector<MPFGenericTrack> &tracks) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");

    TesseractOCRTextDetection::OCR_filter_settings ocr_fset;
    load_settings(job, ocr_fset);

    MPFDetectionError job_status = MPF_DETECTION_SUCCESS;
    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
    load_tags_json(job, job_status, json_kvs_regex);

    string temp_im_directory;
    vector<string> job_names;
    boost::split(job_names, job.job_name, boost::is_any_of(":"));
    string job_name = boost::ireplace_all_copy(job_names[0], "job", "");
    boost::trim(job_name);

    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/TesseractOCRTextDetection";
    temp_im_directory = plugin_path + "/tmp-" + job_name + "-" + random_string(20);

    if (boost::filesystem::exists(temp_im_directory)) {
        LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Unable to write temporary directory (already exists): " +
                                  temp_im_directory);
        return MPF_FILE_WRITE_ERROR;
    }

    vector<string> filelist;
    // Attempts to process generic document.
    // If read successfully, convert to images and store in a temporary directory.
    try {
        boost::filesystem::create_directory(temp_im_directory);
        Magick::ReadOptions options;
        options.density(Magick::Geometry(300, 300));
        options.depth(8);
        list<Magick::Image> images;
        readImages(&images, job.data_uri, options);
        size_t i;
        list<Magick::Image>::iterator image;
        for (image = images.begin(), i = 0; image != images.end(); image++, i++) {
            image->matte(false);
            image->backgroundColor(Magick::Color("WHITE"));
            string tiff_file = temp_im_directory + "/results_extracted" + to_string(i) + ".tiff";
            image->write(tiff_file);
            filelist.push_back(tiff_file);
        }

    } catch (Magick::Exception &error) {
        LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Error processing file " + job.data_uri + " .");
        LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Error: " + error.what());
        if (boost::filesystem::exists(temp_im_directory)) {
            boost::filesystem::remove_all(temp_im_directory);
        }
        return MPF_UNSUPPORTED_DATA_TYPE;
    }

    int page_num = 0;
    string default_lang = ocr_fset.tesseract_lang;

    if (ocr_fset.max_parallel_ocr_threads > 1 && filelist.size() > 1 && ocr_fset.psm != 0) {


        PDF_thread_result thread_results[filelist.size()];
        thread pdf_threads[filelist.size()];
        int index = 0, thread_counter = 0, threads_joined = 0;

        for (const string &filename : filelist) {
            thread_results[index].job_status = job_status;
            MPFImageJob c_job(job.job_name, filename, job.job_properties, job.media_properties);

            try {
                MPFImageReader image_reader(c_job);
                thread_results[index].image = image_reader.GetImage();
            } catch (...) {
                return Utils::HandleDetectionException(c_job, hw_logger_);
            }
            if (!preprocess_image(c_job, thread_results[index].image, ocr_fset,  thread_results[index].job_status)) {
                return  thread_results[index].job_status;
            }


            thread_results[index].osd_track_results = MPFGenericTrack(-1.0);
            thread_results[index].tessdata_script_dir = "";
            if (ocr_fset.enable_osd) {
                OSResults os_results;
                // Reset to original specified language before processing OSD.
                ocr_fset.tesseract_lang = default_lang;
                get_OSD(os_results, thread_results[index].image, c_job, ocr_fset, thread_results[index].osd_track_results .detection_properties,
                        thread_results[index].job_status, thread_results[index].tessdata_script_dir);
            }

            vector<TesseractOCRTextDetection::OCR_output> ocr_outputs;

            pdf_threads[index] = thread(&TesseractOCRTextDetection::get_tesseract_detections, this, std::cref(c_job.job_name),
                                 std::ref(thread_results[index].ocr_outputs), std::ref(thread_results[index].image), std::cref(ocr_fset),
                                 std::cref(ocr_fset.tesseract_lang), std::ref(thread_results[index].job_status),
                                 std::cref(thread_results[index].tessdata_script_dir),  std::ref(thread_results[index].tesseract_success),
                                 true);

            thread_counter++;
            index ++;

            if (thread_counter == ocr_fset.max_parallel_pdf_threads) {
                thread_counter = 0;
                threads_joined = index;
                // Freeze generation of new threads until processing is complete with current batch.
                for (int p_index = 1; p_index <= ocr_fset.max_parallel_pdf_threads; p_index ++) {
                    pdf_threads[index - p_index].join();
                }
            }
        }

        for (index = threads_joined; index < filelist.size(); index++ ) {
            pdf_threads[index].join();
        }

        for (index = 0; index < filelist.size(); index++ ) {
            MPFImageJob c_job(job.job_name, "", job.job_properties, job.media_properties);
            if (!thread_results[index].tesseract_success) {
                LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Could not run tesseract!");
                boost::filesystem::remove_all(temp_im_directory);
                return thread_results[index].job_status;
            }

            // If max_text_tracks is set, filter out to return only the top specified tracks.
            if (ocr_fset.max_text_tracks > 0) {
                sort(thread_results[index].ocr_outputs.begin(), thread_results[index].ocr_outputs.end(), greater<TesseractOCRTextDetection::OCR_output>());
                thread_results[index].ocr_outputs.resize(ocr_fset.max_text_tracks);
            }

            for (const auto &ocr_out: thread_results[index].ocr_outputs) {
                MPFGenericTrack generic_track(ocr_out.confidence);
                // Copy over OSD results into OCR tracks.
                generic_track.detection_properties = thread_results[index].osd_track_results.detection_properties;


                bool process_text = process_text_tagging(generic_track.detection_properties, c_job, ocr_out, thread_results[index].job_status,
                                                         ocr_fset, json_kvs_regex, index);
                if (process_text) {
                    tracks.push_back(generic_track);
                }
            }
        }
    } else {
        for (const string &filename : filelist) {
            MPFImageJob c_job(job.job_name, filename, job.job_properties, job.media_properties);
            cv::Mat image_data;
            try {
                MPFImageReader image_reader(c_job);
                image_data = image_reader.GetImage();
            } catch (...) {
                return Utils::HandleDetectionException(c_job, hw_logger_);
            }
            if (!preprocess_image(c_job, image_data, ocr_fset, job_status)) {
                return job_status;
            }

            MPFGenericTrack osd_track_results(-1);
            string tessdata_script_dir = "";
            if (ocr_fset.psm == 0 || ocr_fset.enable_osd) {
                OSResults os_results;
                // Reset to original specified language before processing OSD.
                ocr_fset.tesseract_lang = default_lang;
                get_OSD(os_results, image_data, c_job, ocr_fset, osd_track_results.detection_properties, job_status,
                        tessdata_script_dir);

                // When PSM is set to 0, there is no need to process any further.
                // Proceed to next page for OSD processing.
                if (ocr_fset.psm == 0) {
                    osd_track_results.detection_properties["PAGE_NUM"] = to_string(page_num + 1);
                    tracks.push_back(osd_track_results);
                    page_num++;
                    continue;
                }
            }

            vector<TesseractOCRTextDetection::OCR_output> ocr_outputs;

            bool tesseract_success;


            get_tesseract_detections(c_job.job_name, ocr_outputs, image_data, ocr_fset, ocr_fset.tesseract_lang,
                                     job_status, tessdata_script_dir, tesseract_success, true);


            if (!tesseract_success) {
                LOG4CXX_ERROR(hw_logger_, "[" + job.job_name + "] Could not run tesseract!");
                boost::filesystem::remove_all(temp_im_directory);
                return job_status;
            }

            // If max_text_tracks is set, filter out to return only the top specified tracks.
            if (ocr_fset.max_text_tracks > 0) {
                sort(ocr_outputs.begin(), ocr_outputs.end(), greater<TesseractOCRTextDetection::OCR_output>());
                ocr_outputs.resize(ocr_fset.max_text_tracks);
            }

            for (const auto &ocr_out: ocr_outputs) {
                MPFGenericTrack generic_track(ocr_out.confidence);
                // Copy over OSD results into OCR tracks.
                generic_track.detection_properties = osd_track_results.detection_properties;

                bool process_text = process_text_tagging(generic_track.detection_properties, c_job, ocr_out, job_status,
                                                         ocr_fset, json_kvs_regex, page_num);
                if (process_text) {
                    tracks.push_back(generic_track);
                }
            }
            page_num++;
        }
    }

    boost::filesystem::remove_all(temp_im_directory);
    LOG4CXX_INFO(hw_logger_,
                 "[" + job.job_name + "] Processing complete. Found " + to_string(tracks.size()) + " tracks.");
    return job_status;
}

MPF_COMPONENT_CREATOR(TesseractOCRTextDetection);
MPF_COMPONENT_DELETER();
