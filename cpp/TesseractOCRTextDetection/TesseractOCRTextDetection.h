/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

#ifndef OPENMPF_COMPONENTS_TESSERACTOCRTEXTDETECTION_H
#define OPENMPF_COMPONENTS_TESSERACTOCRTEXTDETECTION_H

#include "adapters/MPFImageDetectionComponentAdapter.h"
#include <MPFImageReader.h>
#include <MPFDetectionComponent.h>

#include <opencv2/opencv.hpp>
#include <log4cxx/logger.h>
#include <boost/regex.hpp>

#include <QHash>
#include <QString>

namespace MPF {

namespace COMPONENT {

class TesseractOCRTextDetection : public MPFImageDetectionComponentAdapter {

  public:
    TesseractOCRTextDetection() = default;
    ~TesseractOCRTextDetection() = default;

    std::string GetDetectionType();

    bool Init();
    bool Close();

    MPFDetectionError GetDetections(const MPFImageJob &job,
                                    std::vector<MPFImageLocation> &locations);

  private:
    struct OCR_char_stats{
        int alphabet_count;
        int num_count;
        int whspace_count;
        int punct_count;
        int non_eng_count;
        int char_list[26];
    };

    struct OCR_filter_settings{
        bool num_only_ok;
        bool threshold_check;
        bool hist_check;
        int min_word_len;
        int hist_min_char;
        double scale;
        double sharpen;
        double excess_eng_symbols;
        double excess_non_eng_symbols;
        double vowel_min;
        double vowel_max;
        double correl_limit;
    };

    QHash<QString, QString> parameters;
    OCR_filter_settings ocr_fset;
    std::string job_name;

    log4cxx::LoggerPtr hw_logger_;
    std::map<std::string,std::string> regTable;
    std::string fix_regex(std::string inreg);
    std::map<std::string,std::map<std::string,std::vector<std::string>>> parse_json(const std::string &jsonfile_name);
    bool get_tesseract_detections(const MPFImageJob &job, std::string &detection, cv::Mat &original, double weight, int psm, std::string lang);

    void SetDefaultParameters();
    void SetReadConfigParameters();
    OCR_char_stats char_count(const std::string &s, const std::string &white_space, const std::string &eng_symbol, const std::string &eng_num );
    std::string check_string(const std::string &s, const OCR_filter_settings &ocrset);

    bool comp_strcmp(const std::string &strHaystack, const std::string &strNeedle);
    bool comp_regex(const std::string &detection, const std::string &regstr);
    bool Supports(MPFDetectionDataType data_type);
    void Sharpen(cv::Mat &image, double weight);
    std::string log_print_str(const std::string &text);
    std::string log_print_str(const std::wstring &text);
    std::string log_print_str(const char* text);
    std::string log_print_str(const wchar_t* text);
    std::vector<std::string> get_tokens(const std::string &str);

    std::string parseRegexCode(boost::regex_constants::error_type etype);

    std::set<std::string> search_regex(const std::string &ocr_detections, const std::map<std::string,std::vector<std::string>> &json_kvs_regex);
    std::set<std::string> search_string(const std::string &ocr_detections, const std::map<std::string,std::vector<std::string>> &json_kvs_string);
    std::set<std::string> search_string_split(const std::vector<std::string> &tokenized, const std::map<std::string,std::vector<std::string>> &json_kvs_string);
};

}
}

#endif
