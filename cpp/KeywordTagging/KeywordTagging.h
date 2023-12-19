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

#ifndef OPENMPF_COMPONENTS_KEYWORDTAGGING_H
#define OPENMPF_COMPONENTS_KEYWORDTAGGING_H

#include <set>
#include "adapters/MPFGenericDetectionComponentAdapter.h"
#include <MPFDetectionComponent.h>
#include <boost/regex.hpp>
#include <log4cxx/logger.h>

using namespace MPF;
using namespace COMPONENT;

class KeywordTagging : public MPFDetectionComponent {
public:
    bool Init() override;

    bool Close() override;

    std::vector<MPFGenericTrack> GetDetections(const MPFGenericJob &job) override;

    std::vector<MPFImageLocation> GetDetections(const MPFImageJob &job) override;

    std::vector<MPFAudioTrack> GetDetections(const MPFAudioJob &job) override;

    std::vector<MPFVideoTrack> GetDetections(const MPFVideoJob &job) override;

    bool Supports(MPFDetectionDataType data_type) override;

    std::string GetDetectionType() override;

private:
    log4cxx::LoggerPtr hw_logger_;

    std::set<std::wstring> search_regex(const MPFJob &job, const std::wstring &full_text,
                                        const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex,
                                        std::map<std::wstring, std::map<std::wstring, std::vector<std::string>>> &trigger_tags_words_offset,
                                        bool full_regex);

    void process_regex_match(const boost::wsmatch &match, const std::wstring &full_text,
                             std::map<std::wstring, std::vector<std::string>> &trigger_words_offset);

    void process_text_tagging(Properties &detection_properties, const MPFJob &job, const std::map<std::string, std::wstring>& prop_texts,
                              const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex);

    void load_tags_json(const MPFJob &job,
                        std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex);

    std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> parse_json(const MPFJob &job,
                                                                                  const std::string &jsonfile_path);

    bool comp_regex(const MPFJob &job, const std::wstring &full_text, const std::wstring &regstr,
                    std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,
                    bool full_regex, bool case_sensitive);

    std::string parse_regex_code(const boost::regex_constants::error_type &etype);

    bool get_text_to_process(const MPFJob &job, const Properties &detection_properties, std::map<std::string, std::wstring>& prop_texts);
};

#endif //OPENMPF_COMPONENTS_KEYWORDTAGGING_H
