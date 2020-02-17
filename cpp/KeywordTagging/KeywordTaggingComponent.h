//
// Created by mpf on 1/22/20.
//

#ifndef OPENMPF_COMPONENTS_KEYWORDTAGGINGCOMPONENT_H
#define OPENMPF_COMPONENTS_KEYWORDTAGGINGCOMPONENT_H

#include <set>
#include "adapters/MPFGenericDetectionComponentAdapter.h"
#include <boost/regex.hpp>
#include <log4cxx/logger.h>

using namespace MPF;
using namespace COMPONENT;

class KeywordTagger : public MPFGenericDetectionComponentAdapter {
public:
    bool Init() override;

    bool Close() override;

    MPFDetectionError GetDetections(const MPFGenericJob &job,
            std::vector<MPFGenericTrack> &locations) override;

    bool Supports(MPFDetectionDataType data_type) override;

    std::string GetDetectionType() override;

private:

    log4cxx::LoggerPtr hw_logger_;
    bool full_regex;

    std::set<std::wstring> search_regex(const MPFGenericJob &job, const std::wstring &full_text,

                                        const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex,

                                        std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,

                                        bool full_regex, MPFDetectionError &job_status);


    void process_regex_match(const boost::wsmatch &match, const std::wstring &full_text,

                             std::map<std::wstring, std::vector<std::string>> &trigger_words_offset);

    bool process_text_tagging(Properties &detection_properties, const MPFGenericJob &job,
            std::wstring text,
            MPFDetectionError &job_status,
            const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex);


    void load_tags_json(const MPFJob &job, MPFDetectionError &job_status,

                        std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex);

    std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> parse_json(const MPFJob &job,
            const std::string &jsonfile_path,
            MPFDetectionError &job_status);

    bool comp_regex(const MPFGenericJob &job, const std::wstring &full_text, const std::wstring &regstr,

                    std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,

                    bool full_regex, bool case_sensitive,

                    MPFDetectionError &job_status);


    std::string parse_regex_code(boost::regex_constants::error_type etype);
};

#endif //OPENMPF_COMPONENTS_KEYWORDTAGGINGCOMPONENT_H
