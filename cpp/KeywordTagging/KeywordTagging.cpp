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

#include "KeywordTagging.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/locale.hpp>
#include <boost/regex.hpp>
#include <boost/algorithm/string.hpp>
#include "detectionComponentUtils.h"

#include <log4cxx/logmanager.h>

#include <log4cxx/xml/domconfigurator.h>
#include <Utils.h>
#include <MPFDetectionException.h>
#include "JSON.h"

using namespace MPF;
using namespace COMPONENT;

using log4cxx::Logger;
using log4cxx::xml::DOMConfigurator;

using namespace std;

map<wstring, vector<pair<wstring, bool>>>
KeywordTagging::parse_json(const MPFJob &job, const string &jsonfile_path, MPFDetectionError &job_status) {

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

void KeywordTagging::process_regex_match(const boost::wsmatch &match, const wstring &full_text,
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

string KeywordTagging::parse_regex_code(boost::regex_constants::error_type etype) {
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

bool KeywordTagging::comp_regex(const MPFJob &job, const wstring &full_text,
                                           const wstring &regstr, map<wstring, vector<string>> &trigger_words_offset,
                                           bool full_regex,
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

        if (full_regex) {
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

set<wstring> KeywordTagging::search_regex(const MPFJob &job, const wstring &full_text,
                                    const map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex,
                                    map<wstring, vector<string>>  &trigger_words_offset,
                                    bool full_regex, MPFDetectionError &job_status) {
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
                           full_regex, case_sens, job_status)) {
                found_keys_regex.insert(key);
                // Discontinue searching unless full regex search is enabled.
                if (!full_regex) {
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

void KeywordTagging::load_tags_json(const MPFJob &job, MPFDetectionError &job_status,
                                               map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex) {

    string run_dir = GetRunDirectory();

    if (run_dir.empty()) {
        run_dir = ".";
    }

    string plugin_path = run_dir + "/KeywordTagging";

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Running from directory " + plugin_path)

    string jsonfile_path = DetectionComponentUtils::GetProperty<string>(job.job_properties, "TAGGING_FILE",
                                                                        "text-tags.json");

    if (jsonfile_path.find('$') != string::npos || jsonfile_path.find('/') != string::npos) {
        string new_jsonfile_name = "";
        Utils::expandFileName(jsonfile_path, new_jsonfile_name);
        jsonfile_path = new_jsonfile_name;
    } else {
        jsonfile_path = plugin_path + "/config/" + jsonfile_path;
    }

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] About to read JSON from: " + jsonfile_path)
    json_kvs_regex = parse_json(job, jsonfile_path, job_status);
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Read JSON")
}

wstring clean_whitespace(const wstring &input) {
    boost::wregex re(L"\n(\n|[[:space:]])+");
    boost::wregex re2(L"\\\\n(\\\\n|[[:space:]])+");

    wstring result = boost::regex_replace(input, re, L"\n");
    wstring result2 = boost::regex_replace(result, re2, L"\\\\n");

    result2 = boost::trim_copy(result2);

    return result2;
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

bool KeywordTagging::Init() {
    boost::locale::generator gen;
    locale loc = gen("");
    locale::global(loc);

    string run_dir = GetRunDirectory();

    if (run_dir.empty()) {
        run_dir = ".";
    }

    string plugin_path = run_dir + "/KeywordTagging";
    string config_path = plugin_path + "/config";

    log4cxx::xml::DOMConfigurator::configure(plugin_path + "/config/Log4cxxConfig.xml");
    hw_logger_ = log4cxx::Logger::getLogger("KeywordTagging");

    LOG4CXX_DEBUG(hw_logger_, "Plugin path: " << plugin_path);
    LOG4CXX_INFO(hw_logger_, "Initializing keyword tagging");

    return true;
}

bool KeywordTagging::Close() {
    return true;
}

string KeywordTagging::GetDetectionType() {
    return "KEYWORD";
}

vector<MPFGenericTrack> KeywordTagging::GetDetections(const MPFGenericJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");
    MPFDetectionError job_status = MPF_DETECTION_SUCCESS;
    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;

    load_tags_json(job, job_status, json_kvs_regex);

    MPFGenericTrack text_tags;

    wstring text;
    vector<MPFGenericTrack> tags;

    if (job.has_feed_forward_track) {
        string temp = "";
        LOG4CXX_INFO(hw_logger_, "Generic job is feed forward");
        Properties properties = job.feed_forward_track.detection_properties;
        if (properties.count("TEXT")) {
            LOG4CXX_INFO(hw_logger_, "Performing tagging on TEXT property.");
            temp = properties.at("TEXT");
        } else if (properties.count("TRANSCRIPT")) {
            LOG4CXX_INFO(hw_logger_, "Performing tagging on TRANSCRIPT property.");
            temp = properties.at("TRANSCRIPT");
        } else {
            LOG4CXX_DEBUG(hw_logger_, "Feed forward track missing TEXT or TRANSCRIPT property");
            throw MPFDetectionException(MPF_MISSING_PROPERTY, "Feed forward track missing TEXT or TRANSCRIPT property");
        }

        text = boost::locale::conv::utf_to_utf<wchar_t>(temp);
        LOG4CXX_DEBUG(hw_logger_, "Copying properties over from feed forward track");
        text_tags.detection_properties = properties;
    } else {
        LOG4CXX_INFO(hw_logger_, "Generic job is not feed forward. Performing tagging on text file.");
        wifstream file {job.data_uri};
        wstring file_contents((istreambuf_iterator<wchar_t>(file)),
                              (istreambuf_iterator<wchar_t>()));
        text = file_contents;
    }

    bool process_text = process_text_tagging(text_tags.detection_properties, job, text, job_status,
                                             json_kvs_regex);

    if (process_text) {
        tags.push_back(text_tags);
    }

    return tags;
}

vector<MPFAudioTrack> KeywordTagging::GetDetections(const MPFAudioJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");
    MPFDetectionError job_status = MPF_DETECTION_SUCCESS;
    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;

    load_tags_json(job, job_status, json_kvs_regex);

    MPFAudioTrack text_tags;
    vector<MPFAudioTrack> tags;

    wstring text;

    LOG4CXX_INFO(hw_logger_, "Checking audio job is feed forward and the detection has a TRANSCRIPT property");
    if (job.has_feed_forward_track) {
        if (job.feed_forward_track.detection_properties.count("TRANSCRIPT")) {
            string temp = job.feed_forward_track.detection_properties.at("TRANSCRIPT");
            text = boost::locale::conv::utf_to_utf<wchar_t>(temp);
            LOG4CXX_DEBUG(hw_logger_, "Copying properties over from feed forward track");
            text_tags.detection_properties = job.feed_forward_track.detection_properties;
        } else {
            LOG4CXX_DEBUG(hw_logger_, "Feed forward track missing TRANSCRIPT property");
            throw MPFDetectionException(MPF_MISSING_PROPERTY, "Feed forward track missing TRANSCRIPT property");
        }
    } else {
        LOG4CXX_DEBUG(hw_logger_, "Job is not feed forward");
        throw MPFDetectionException(MPF_MISSING_PROPERTY, "Job is not feed forward");
    }

    bool process_text = process_text_tagging(text_tags.detection_properties, job, text, job_status,
                                             json_kvs_regex);

    if (process_text) {
        tags.push_back(text_tags);
    }

    return tags;
}

vector<MPFVideoTrack> KeywordTagging::GetDetections(const MPFVideoJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");
    MPFDetectionError job_status = MPF_DETECTION_SUCCESS;
    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;

    load_tags_json(job, job_status, json_kvs_regex);

    MPFVideoTrack text_tags;
    vector<MPFVideoTrack> tags;

    wstring text;

    LOG4CXX_INFO(hw_logger_, "Checking video job is feed forward and the detection has a TEXT or TRANSCRIPT property");
    if (job.has_feed_forward_track) {
        if (job.feed_forward_track.detection_properties.count("TEXT")) {
            string temp = job.feed_forward_track.detection_properties.at("TEXT");
            text = boost::locale::conv::utf_to_utf<wchar_t>(temp);
            LOG4CXX_DEBUG(hw_logger_, "Copying properties over from feed forward track");
            text_tags.detection_properties = job.feed_forward_track.detection_properties;
        } else if (job.feed_forward_track.detection_properties.count("TRANSCRIPT")) {
            string temp = job.feed_forward_track.detection_properties.at("TRANSCRIPT");
            text = boost::locale::conv::utf_to_utf<wchar_t>(temp);
            LOG4CXX_DEBUG(hw_logger_, "Copying properties over from feed forward track");
            text_tags.detection_properties = job.feed_forward_track.detection_properties;
        } else {
            LOG4CXX_DEBUG(hw_logger_, "Feed forward track missing a TEXT or TRANSCRIPT property");
            throw MPFDetectionException(MPF_MISSING_PROPERTY, "Feed forward track missing TEXT or TRANSCRIPT property");
        }
    } else {
        LOG4CXX_DEBUG(hw_logger_, "Job is not feed forward");
        throw MPFDetectionException(MPF_MISSING_PROPERTY, "Job is not feed forward");
    }

    bool process_text = process_text_tagging(text_tags.detection_properties, job, text, job_status,
                                             json_kvs_regex);

    if (process_text) {
        tags.push_back(text_tags);
    }

    return tags;
}

vector<MPFImageLocation> KeywordTagging::GetDetections(const MPFImageJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");


    MPFDetectionError job_status = MPF_DETECTION_SUCCESS;
    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;

    load_tags_json(job, job_status, json_kvs_regex);

    MPFImageLocation text_tags;
    vector<MPFImageLocation> tags;

    wstring text;

    LOG4CXX_DEBUG(hw_logger_, "Checking image job is feed forward and the detection has a TEXT property");
    if (job.has_feed_forward_location) {
        if (job.feed_forward_location.detection_properties.count("TEXT")) {
            string temp = job.feed_forward_location.detection_properties.at("TEXT");
            text = boost::locale::conv::utf_to_utf<wchar_t>(temp);
            LOG4CXX_DEBUG(hw_logger_, "Copying properties over from feed forward detection");
            text_tags.detection_properties = job.feed_forward_location.detection_properties;

        } else {
            LOG4CXX_DEBUG(hw_logger_, "Feed forward detection missing TEXT property");
            throw MPFDetectionException(
                    MPF_MISSING_PROPERTY, "Feed forward detection missing TEXT property");
        }
    } else {
        LOG4CXX_DEBUG(hw_logger_, "Job is not feed forward.");
        throw MPFDetectionException(MPF_DETECTION_FAILED, "Job is not feed forward");
    }

    bool process_text = process_text_tagging(text_tags.detection_properties, job, text, job_status,
                                             json_kvs_regex);

    if (process_text) {
        tags.push_back(text_tags);
    }

    return tags;
}

bool KeywordTagging::Supports(MPFDetectionDataType data_type) {
    return data_type == MPFDetectionDataType::IMAGE || data_type == MPFDetectionDataType::UNKNOWN
        || data_type == MPFDetectionDataType::AUDIO || data_type == MPFDetectionDataType::VIDEO;
}

bool KeywordTagging::process_text_tagging(Properties &detection_properties, const MPFJob &job,
                                         wstring text,
                                         MPFDetectionError &job_status,
                                         const map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex) {

    text = clean_whitespace(text);

    bool full_regex = false;

    if (DetectionComponentUtils::GetProperty<string>(job.job_properties, "FULL_REGEX_SEARCH",
                                                     "true") == "true") {
        full_regex = true;
    }

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing tags: ");
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Text was: " +
                              boost::locale::conv::utf_to_utf<char>(text));

    if (is_only_ascii_whitespace(text)) {
        LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] No text in file!");
        return false;
    }

    set<wstring> trigger_words;
    map<wstring, vector<string>> trigger_words_offset;
    set<wstring> found_tags_regex = search_regex(job, text, json_kvs_regex, trigger_words_offset, full_regex, job_status);

    wstring tag_string = boost::algorithm::join(found_tags_regex, L"; ");

    vector<string> offsets_list;
    vector<wstring> triggers_list;

    wstring tag_trigger = boost::algorithm::join(trigger_words, L"; ");

    for (auto const& word_offset : trigger_words_offset ) {
        triggers_list.push_back(word_offset.first);
        offsets_list.push_back(boost::algorithm::join(word_offset.second, ", "));
    }

    string tag_offset = boost::algorithm::join(offsets_list, "; ");

    tag_trigger = tag_trigger + boost::algorithm::join(triggers_list, L"; ");

    detection_properties["TAGS"] = boost::locale::conv::utf_to_utf<char>(tag_string);
    detection_properties["TRIGGER_WORDS"] = boost::locale::conv::utf_to_utf<char>(tag_trigger);
    detection_properties["TRIGGER_WORDS_OFFSET"] = tag_offset;
    detection_properties["TEXT"] = boost::locale::conv::utf_to_utf<char>(text);

    return true;
}

MPF_COMPONENT_CREATOR(KeywordTagging);
MPF_COMPONENT_DELETER();