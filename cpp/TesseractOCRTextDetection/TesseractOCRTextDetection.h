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
#include <boost/locale.hpp>

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
        bool invert;
        bool enable_adaptive_thrs;
        bool enable_otsu_thrs;
        bool enable_rescale;
        bool enable_sharpen;
        int adaptive_thrs_pixel;
        int min_word_len;
        int hist_min_char;
        double adaptive_thrs_c;
        double scale;
        double sharpen;
        double excess_eng_symbols;
        double excess_non_eng_symbols;
        double vowel_min;
        double vowel_max;
        double correl_limit;
        std::string tesseract_lang;

    };

    QHash<QString, QString> parameters;
    OCR_filter_settings ocr_fset;

    log4cxx::LoggerPtr hw_logger_;
    std::map<std::wstring,std::wstring> regTable;
    std::wstring fix_regex(std::wstring inreg);
    std::map<std::wstring,std::map<std::wstring,std::vector<std::wstring>>> parse_json(const MPFImageJob &job, const std::string &jsonfile_name, MPFDetectionError &job_status);
    bool get_tesseract_detections(const MPFImageJob &job, std::vector<std::pair<std::string, std::wstring>> &detection, cv::Mat &original, double weight, int psm, std::string lang, MPFDetectionError &job_status);

    void SetDefaultParameters();
    void SetReadConfigParameters();
    OCR_char_stats char_count(const std::wstring &s, const std::wstring &white_space, const std::wstring &eng_symbol, const std::wstring &eng_num );
    std::wstring check_string(const std::wstring &s, const OCR_filter_settings &ocrset);

    bool comp_strcmp(const std::wstring &strHaystack, const std::wstring &strNeedle);
    bool comp_regex(const MPFImageJob &job, const std::wstring &detection, const std::wstring &regstr, MPFDetectionError &job_status);
    bool Supports(MPFDetectionDataType data_type);
    bool token_comparison(const std::wstring &token, const std::wstring &value);
    void Sharpen(cv::Mat &image, double weight);
    std::string log_print_str(const std::string &text);
    std::string log_print_str(const std::wstring &text);
    std::string log_print_str(const char* text);
    std::string log_print_str(const wchar_t* text);
    std::vector<std::wstring> get_tokens(const std::wstring &str);

    std::string parseRegexCode(boost::regex_constants::error_type etype);

    std::set<std::wstring> search_regex(const MPFImageJob &job, const std::wstring &ocr_detections, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_regex, MPFDetectionError &job_status);
    std::set<std::wstring> search_string(const MPFImageJob &job, const std::wstring &ocr_detections, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string);
    std::set<std::wstring> search_string_split(const MPFImageJob &job, const std::vector<std::wstring> &tokenized, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string);
};

}
}

#endif
