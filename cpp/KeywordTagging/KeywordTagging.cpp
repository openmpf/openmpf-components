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
KeywordTagging::parse_json(const MPFJob &job, const string &jsonfile_path) {

    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
    ifstream ifs(jsonfile_path);

    if (!ifs.is_open()) {
        throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE,
                                    "Could not open tagging file: " + jsonfile_path);
    }

    string j;
    stringstream buffer2;
    wstring x;

    buffer2 << ifs.rdbuf();
    j = buffer2.str();

    x = boost::locale::conv::utf_to_utf<wchar_t>(j);

    JSONValue *value = JSON::Parse(x.c_str());

    if (value == NULL) {
        throw MPFDetectionException(MPF_COULD_NOT_READ_DATAFILE,
                                    "Could not parse tagging file: " + jsonfile_path);
    }

    JSONObject root;
    root = value->AsObject();

    if (root.find(L"TAGS_BY_REGEX") != root.end() && root[L"TAGS_BY_REGEX"]->IsObject()) {
        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Regex tags found.");

        JSONObject key_tags = root[L"TAGS_BY_REGEX"]->AsObject();
        vector<wstring> keys = root[L"TAGS_BY_REGEX"]->ObjectKeys();
        vector<wstring>::iterator iter = keys.begin();

        while (iter != keys.end()) {
            auto term = *iter;
            wstring term_temp(term);

            if (!key_tags[term]->IsArray()) {
                throw MPFDetectionException(MPF_COULD_NOT_READ_DATAFILE,
                                            "Could not parse tagging file: " + jsonfile_path +
                                            ". In TAGS_BY_REGEX the entry for \"" +
                                            boost::locale::conv::utf_to_utf<char>(term) +
                                            "\" is not a valid JSON array.");
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
        throw MPFDetectionException(MPF_COULD_NOT_READ_DATAFILE,
                                    "Could not parse tagging file: " + jsonfile_path +
                                    ". TAGS_BY_REGEX not found.");
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

    if (trigger_words_offset.find(trigger_word) == trigger_words_offset.end()) {
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

string KeywordTagging::parse_regex_code(const boost::regex_constants::error_type &etype) {
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
                                bool case_sensitive) {
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

            for(iter; iter != end; ++iter ) {
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
        ss << "regex_error caught: " << parse_regex_code(e.code()) << ": " << e.what() << '\n';
        throw MPFDetectionException(MPF_COULD_NOT_READ_DATAFILE, ss.str());
    }

    return found;
}

set<wstring> KeywordTagging::search_regex(const MPFJob &job, const wstring &full_text,
                                          const map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex,
                                          map<wstring, vector<string>>  &trigger_words_offset,
                                          bool full_regex) {
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

            if (comp_regex(job, full_text, regex_pattern, trigger_words_offset, full_regex, case_sens)) {
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

void KeywordTagging::load_tags_json(const MPFJob &job, map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex) {

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
    json_kvs_regex = parse_json(job, jsonfile_path);
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
    return "TEXT";
}

vector<MPFGenericTrack> KeywordTagging::GetDetections(const MPFGenericJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");

    MPFGenericTrack track;
    bool has_prop = true;
    wstring text;

    if (job.has_feed_forward_track) {
        track = job.feed_forward_track;
        has_prop = get_text_to_process(job, track.detection_properties, text);
    } else {
        LOG4CXX_INFO(hw_logger_, "Generic job is not feed forward. Performing tagging on text file.");
        wifstream file {job.data_uri};
        if (!file.is_open()) {
            throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, "Cannot open: " + job.data_uri);
        }
        wstring file_contents((istreambuf_iterator<wchar_t>(file)), (istreambuf_iterator<wchar_t>()));
        text = move(file_contents);
        track.detection_properties["TEXT"] = boost::locale::conv::utf_to_utf<char>(text);
        file.close();
    }

    if (has_prop) {
        map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
        load_tags_json(job, json_kvs_regex);
        process_text_tagging(track.detection_properties, job, text, json_kvs_regex);
    }

    return {track};
}

vector<MPFAudioTrack> KeywordTagging::GetDetections(const MPFAudioJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");

    if (!job.has_feed_forward_track) {
        LOG4CXX_DEBUG(hw_logger_, "Can only process audio files in feed forward jobs.");
        throw MPFDetectionException(MPF_UNSUPPORTED_DATA_TYPE, "Can only process audio files in feed forward jobs.");
    }

    MPFAudioTrack track = job.feed_forward_track;
    wstring text;

    if (get_text_to_process(job, track.detection_properties, text)) {
        map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
        load_tags_json(job, json_kvs_regex);
        process_text_tagging(track.detection_properties, job, text, json_kvs_regex);
    }

    return {track};
}

vector<MPFImageLocation> KeywordTagging::GetDetections(const MPFImageJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");

    if (!job.has_feed_forward_location) {
        LOG4CXX_DEBUG(hw_logger_, "Can only process image files in feed forward jobs.");
        throw MPFDetectionException(MPF_UNSUPPORTED_DATA_TYPE, "Can only process image files in feed forward jobs.");
    }

    MPFImageLocation location = job.feed_forward_location;
    wstring text;

    if (get_text_to_process(job, location.detection_properties, text)) {
        map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
        load_tags_json(job, json_kvs_regex);
        process_text_tagging(location.detection_properties, job, text, json_kvs_regex);
    }

    return {location};
}

vector<MPFVideoTrack> KeywordTagging::GetDetections(const MPFVideoJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");

    if (!job.has_feed_forward_track) {
        LOG4CXX_DEBUG(hw_logger_, "Can only process video files in feed forward jobs.");
        throw MPFDetectionException(MPF_UNSUPPORTED_DATA_TYPE, "Can only process video files in feed forward jobs.");
    }

    map<wstring, vector<pair<wstring, bool>>> json_kvs_regex;
    load_tags_json(job, json_kvs_regex);

    MPFVideoTrack track = job.feed_forward_track;
    wstring text;

    // process track-level property
    if (get_text_to_process(job, track.detection_properties, text)) {
        process_text_tagging(track.detection_properties, job, text, json_kvs_regex);
    }

    // process each detection-level property
    for (auto &pair : track.frame_locations) {
        if (get_text_to_process(job, pair.second.detection_properties, text)) {
            process_text_tagging(pair.second.detection_properties, job, text, json_kvs_regex);
        }
    }

    return {track};
}

bool KeywordTagging::Supports(MPFDetectionDataType data_type) {
    return data_type == MPFDetectionDataType::IMAGE || data_type == MPFDetectionDataType::UNKNOWN
        || data_type == MPFDetectionDataType::AUDIO || data_type == MPFDetectionDataType::VIDEO;
}

bool KeywordTagging::get_text_to_process(const MPFJob &job, const Properties &detection_properties, wstring& text) {
    string props_to_process = DetectionComponentUtils::GetProperty<string>(job.job_properties,
                                                                           "FEED_FORWARD_PROP_TO_PROCESS",
                                                                           "TEXT,TRANSCRIPT");
    vector<string> split_props_to_process;
    boost::split(split_props_to_process, props_to_process, boost::is_any_of(","));

    for (string prop_to_process : split_props_to_process) {
        boost::trim(prop_to_process);
        if (detection_properties.find(prop_to_process) != detection_properties.end()) {
            LOG4CXX_INFO(hw_logger_, "Performing tagging on " + prop_to_process + " property.");
            text = boost::locale::conv::utf_to_utf<wchar_t>(detection_properties.at(prop_to_process));
            return true;
        }
    }

    LOG4CXX_WARN(hw_logger_, "Feed forward element missing one of the following properties: " + props_to_process);
    return false;
}

void KeywordTagging::process_text_tagging(Properties &detection_properties, const MPFJob &job, const wstring &full_text,
                                          const map<wstring, vector<pair<wstring, bool>>> &json_kvs_regex) {
    wstring text = clean_whitespace(full_text);

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing tags");
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Text is: " +
                              boost::locale::conv::utf_to_utf<char>(text));

    if (is_only_ascii_whitespace(text)) {
        LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] No text to process!");
        return;
    }

    bool full_regex = DetectionComponentUtils::GetProperty(job.job_properties, "FULL_REGEX_SEARCH", true);

    set<wstring> trigger_words;
    map<wstring, vector<string>> trigger_words_offset;
    set<wstring> found_tags_regex = search_regex(job, text, json_kvs_regex, trigger_words_offset, full_regex);

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
}

MPF_COMPONENT_CREATOR(KeywordTagging);
MPF_COMPONENT_DELETER();