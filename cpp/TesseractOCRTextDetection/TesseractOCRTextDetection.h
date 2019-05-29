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

#ifndef OPENMPF_COMPONENTS_TESSERACTOCRTEXTDETECTION_H
#define OPENMPF_COMPONENTS_TESSERACTOCRTEXTDETECTION_H

#include "adapters/MPFImageDetectionComponentAdapter.h"
#include <MPFImageReader.h>
#include <MPFDetectionComponent.h>

#include <opencv2/opencv.hpp>
#include <log4cxx/logger.h>
#include <boost/regex.hpp>
#include <boost/locale.hpp>

#include <tesseract/osdetect.h>
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

    MPFDetectionError GetDetections(const MPFGenericJob &job,
                                    std::vector<MPFGenericTrack> &tracks);

  private:
    struct OCR_char_stats {
        int alphabet_count;
        int num_count;
        int whspace_count;
        int punct_count;
        int non_eng_count;
        int char_list[26];
    };

    struct OCR_filter_settings {
        bool num_only_ok;
        bool threshold_check;
        bool hist_check;
        bool invert;
        bool enable_adaptive_thrs;
        bool enable_otsu_thrs;
        bool enable_rescale;
        bool enable_sharpen;
        bool enable_osd;
        bool combine_detected_scripts;
        int adaptive_thrs_pixel;
        int min_word_len;
        int hist_min_char;
        int psm;
        int oem;
        int max_scripts;
        int max_text_tracks;
        double adaptive_thrs_c;
        double scale;
        double sharpen;
        double excess_eng_symbols;
        double excess_non_eng_symbols;
        double vowel_min;
        double vowel_max;
        double correl_limit;
        double min_orientation_confidence;
        double min_script_confidence;
        double min_script_score;
        double min_secondary_script_thrs;
        std::string tesseract_lang;
        std::string model_dir;
    };

    struct OCR_output {
        double confidence;
        std::string language;
        std::wstring text;

        bool operator < (const OCR_output& ocr_out) const
        {
            return (confidence < ocr_out.confidence);
        }

        bool operator > (const OCR_output& ocr_out) const
        {
            return (confidence > ocr_out.confidence);
        }
    };

    struct OSD_script {
        int id;
        double score;
        bool operator < (const OSD_script& script) const
        {
            return (score < script.score);
        }

        bool operator > (const OSD_script& script) const
        {
            return (score > script.score);
        }
    };

    QHash<QString, QString> parameters;
    OCR_filter_settings default_ocr_fset;

    log4cxx::LoggerPtr hw_logger_;
    std::map<std::wstring,std::wstring> regTable;
    std::wstring fix_regex(std::wstring inreg);
    std::map<std::wstring,std::map<std::wstring,std::vector<std::wstring>>> parse_json(const MPFJob &job, const std::string &jsonfile_path, MPFDetectionError &job_status);
    bool get_tesseract_detections(const MPFImageJob &job, std::vector<OCR_output> &detection, cv::Mat &imi, const OCR_filter_settings &ocr_fset, MPFDetectionError &job_status, const std::string &tessdata_script_dir);
    bool preprocess_image(const MPFImageJob &job, cv::Mat &input_image, const OCR_filter_settings &ocr_fset,  MPFDetectionError &job_status);

    void set_default_parameters();
    void set_read_config_parameters();
    void load_settings(const MPFJob &job, OCR_filter_settings &ocr_fset);
    void load_tags_json(const MPFJob &job, MPFDetectionError &job_status,
        std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string,
        std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string_split,
        std::map<std::wstring,std::vector<std::wstring>> &json_kvs_regex);

    void get_path_files(const std::string root, const std::string& ext, std::vector<std::string> &ret);
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

    std::string parse_regex_code(boost::regex_constants::error_type etype);

    std::set<std::wstring> search_regex(const MPFImageJob &job, const std::wstring &ocr_detections, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_regex, MPFDetectionError &job_status);
    std::set<std::wstring> search_string(const MPFImageJob &job, const std::wstring &ocr_detections, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string);
    std::set<std::wstring> search_string_split(const MPFImageJob &job, const std::vector<std::wstring> &tokenized, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string);

    bool process_text_tagging(Properties &detection_properties, const MPFImageJob &job, OCR_output &ocr_out, MPFDetectionError  &job_status,
        const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
        const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_regex,
        const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string_split,
        const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string, int page_num = -1);

    void get_OSD(OSResults &results, cv::Mat &imi, const MPFImageJob &job, OCR_filter_settings &ocr_fset, Properties &detection_properties, MPFDetectionError &job_status, std::string &tessdata_script_dir);
    std::string return_valid_tessdir(const MPFImageJob &job, const std::string &lang_str, const std::string &directory);
    bool check_tess_model_directory(const MPFImageJob &job, const std::string &lang_str, const std::string &directory);
};

}
}

#endif
