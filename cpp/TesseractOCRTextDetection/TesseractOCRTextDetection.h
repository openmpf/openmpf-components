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

#include <tesseract/baseapi.h>
#include <tesseract/osdetect.h>
#include <QHash>
#include <QString>

namespace MPF {

    namespace COMPONENT {

        enum Text_type{Unknown, Structured, Unstructured};

        class TesseractOCRTextDetection : public MPFImageDetectionComponentAdapter {

        public:
            TesseractOCRTextDetection() = default;

            ~TesseractOCRTextDetection();

            bool Init() override;

            bool Close() override;

            MPFDetectionError GetDetections(const MPFImageJob &job,
                                            std::vector<MPFImageLocation> &locations) override;

            MPFDetectionError GetDetections(const MPFGenericJob &job,
                                            std::vector<MPFGenericTrack> &tracks) override;

            std::string GetDetectionType() override;

            bool Supports(MPFDetectionDataType data_type) override;

        private:

            struct OCR_filter_settings {
                bool invert;
                bool enable_hist_equalization;
                bool enable_adaptive_hist_equalization;
                bool enable_adaptive_thrs;
                bool enable_otsu_thrs;
                bool enable_osd;
                bool combine_detected_scripts;
                bool processing_wild_text;
                bool rotate_and_detect;
                bool full_regex_search;
                bool enable_osd_fallback;
                bool enable_parallel_processing;
                int adaptive_thrs_pixel;
                int psm;
                int oem;
                int max_scripts;
                int max_text_tracks;
                int min_height;
                int adaptive_hist_tile_size;
                int max_parallel_threads;
                double adaptive_hist_clip_limit;
                double adaptive_thrs_c;
                double scale;
                double sharpen;
                double min_orientation_confidence;
                double min_script_confidence;
                double min_script_score;
                double min_secondary_script_thrs;
                double rotate_and_detect_min_confidence;
                std::string tesseract_lang;
                std::string model_dir;
            };

            struct OCR_output {
                double confidence;
                std::string language;
                std::wstring text;
                std::string two_pass_rotation;
                bool two_pass_correction;

                bool operator<(const OCR_output &ocr_out) const {
                    return (confidence < ocr_out.confidence);
                }

                bool operator>(const OCR_output &ocr_out) const {
                    return (confidence > ocr_out.confidence);
                }
            };

            struct OSD_script {
                int id;
                double score;

                bool operator<(const OSD_script &script) const {
                    return (score < script.score);
                }

                bool operator>(const OSD_script &script) const {
                    return (score > script.score);
                }
            };

            log4cxx::LoggerPtr hw_logger_;
            QHash<QString, QString> parameters;
            OCR_filter_settings default_ocr_fset;

            // Map of {OCR engine, language} pairs to Tesseract API pointers
            std::map<std::pair<int, std::string>, tesseract::TessBaseAPI *> tess_api_map;

            std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> parse_json(const MPFJob &job,
                                                                               const std::string &jsonfile_path,
                                                                               MPFDetectionError &job_status);

            bool get_tesseract_detections(const MPFImageJob &job, std::vector<OCR_output> &detections_by_lang, cv::Mat &imi,
                                          const OCR_filter_settings &ocr_fset, MPFDetectionError &job_status,
                                          const std::string &tessdata_script_dir);

            bool preprocess_image(const MPFImageJob &job, cv::Mat &input_image, const OCR_filter_settings &ocr_fset,
                                  MPFDetectionError &job_status);

            void process_tesseract_lang_model(const std::string &lang, const cv::Mat &imi, const MPFImageJob &job,
                                              MPFDetectionError &job_status, const std::string &tessdata_script_dir,
                                              const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                              std::string &text_result, double &confidence, bool &success, bool parallel);

            void set_default_parameters();

            void set_read_config_parameters();

            void load_settings(const MPFJob &job, OCR_filter_settings &ocr_fset, const Text_type &text_type = Unknown);

            void load_tags_json(const MPFJob &job, MPFDetectionError &job_status,
                                std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex);

            bool comp_regex(const MPFImageJob &job, const std::wstring &full_text, const std::wstring &regstr,
                            std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,
                            const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset, bool case_sensitive,
                            MPFDetectionError &job_status);

            void process_regex_match(const boost::wsmatch &match, const std::wstring &full_text,
                                     std::map<std::wstring, std::vector<std::string>> &trigger_words_offset);

            void sharpen(cv::Mat &image, double weight);

            std::string parse_regex_code(boost::regex_constants::error_type etype);

            std::set<std::wstring> search_regex(const MPFImageJob &job, const std::wstring &full_text,
                                                const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex,
                                                std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,
                                                const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                MPFDetectionError &job_status);

            bool process_text_tagging(Properties &detection_properties, const MPFImageJob &job, OCR_output &ocr_out,
                                      MPFDetectionError &job_status,
                                      const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                      const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex,
                                      int page_num = -1);

            void get_OSD(OSResults &results, cv::Mat &imi, const MPFImageJob &job, OCR_filter_settings &ocr_fset,
                         Properties &detection_properties, MPFDetectionError &job_status,
                         std::string &tessdata_script_dir);

            std::string
            return_valid_tessdir(const MPFImageJob &job, const std::string &lang_str, const std::string &directory);

            bool check_tess_model_directory(const MPFImageJob &job, const std::string &lang_str,
                                            const std::string &directory);
        };

    }
}

#endif
