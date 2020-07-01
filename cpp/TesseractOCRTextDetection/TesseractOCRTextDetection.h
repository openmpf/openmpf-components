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

#ifndef OPENMPF_COMPONENTS_TESSERACTOCRTEXTDETECTION_H
#define OPENMPF_COMPONENTS_TESSERACTOCRTEXTDETECTION_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <boost/regex.hpp>
#include <boost/locale.hpp>

#include <log4cxx/logger.h>

#include <opencv2/opencv.hpp>

#include <QHash>
#include <QString>

#include <tesseract/baseapi.h>
#include <tesseract/osdetect.h>

#include <adapters/MPFImageDetectionComponentAdapter.h>
#include <MPFImageReader.h>
#include <MPFDetectionComponent.h>


namespace MPF {

    namespace COMPONENT {

        enum Text_type{Unknown, Structured, Unstructured};

        class TessApiWrapper;

        class TesseractOCRTextDetection : public MPFImageDetectionComponentAdapter {

        public:
            bool Init() override;

            bool Close() override;

            std::vector<MPFImageLocation> GetDetections(const MPFImageJob &job) override;

            std::vector<MPFGenericTrack> GetDetections(const MPFGenericJob &job) override;

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
                int adaptive_thrs_pixel;
                int psm;
                int oem;
                int max_pixels;
                int max_scripts;
                int max_text_tracks;
                int min_height;
                int invalid_min_image_size;
                int adaptive_hist_tile_size;
                int max_parallel_ocr_threads;
                int max_parallel_pdf_threads;
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

            struct OCR_job_inputs {
                const std::string *job_name;
                const std::string *lang;
                const std::string *tessdata_script_dir;
                const std::string *run_dir;
                const cv::Mat *imi;
                const OCR_filter_settings *ocr_fset;
                bool process_pdf;
                bool parallel_processing;

                std::set<std::string> ocr_lang_inputs;
                log4cxx::LoggerPtr hw_logger_;
                std::map<std::pair<int, std::string>, TessApiWrapper> *tess_api_map;
            };

            struct Image_results{
                std::vector<OCR_output> detections_by_lang;
                MPFDetectionError job_status;
            };

            struct OCR_results {
                std::string text_result;
                std::string lang;
                MPFDetectionError job_status;
                double confidence;
            };

            struct PDF_page_inputs {
                std::string run_dir;
                std::vector<std::string> filelist;
                const MPFGenericJob *job;
                std::string default_lang;
                OCR_filter_settings ocr_fset;
                log4cxx::LoggerPtr hw_logger_;
                std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> json_kvs_regex;
            };

            struct PDF_page_results {
                std::set<std::string> all_missing_languages;
                MPFDetectionError job_status;
                std::vector<MPFGenericTrack> *tracks;
            };

            struct PDF_thread_variables {
                cv::Mat image;
                std::string lang;
                std::string tessdata_script_dir;
                MPFGenericTrack osd_track_results;

                OCR_job_inputs ocr_input;
                Image_results page_thread_res;

                PDF_thread_variables()
                {
                    ocr_input.lang = &lang;
                    ocr_input.tessdata_script_dir = &tessdata_script_dir;
                    ocr_input.imi = &image;
                }
            };

            void process_parallel_pdf_pages(PDF_page_inputs &page_inputs, PDF_page_results &page_results);
            void process_serial_pdf_pages(PDF_page_inputs &page_inputs, PDF_page_results &page_results);

            log4cxx::LoggerPtr hw_logger_;
            QHash<QString, QString> parameters;
            OCR_filter_settings default_ocr_fset;

            // Map of {OCR engine, language} pairs to TessApiWrapper
            std::map<std::pair<int, std::string>, TessApiWrapper> tess_api_map;
            std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> parse_json(const MPFJob &job,
                                                                                          const std::string &jsonfile_path,
                                                                                          MPFDetectionError &job_status);


            static void get_tesseract_detections(OCR_job_inputs &input, Image_results &result);
            
            static void process_parallel_image_runs(OCR_job_inputs &inputs, Image_results &results);
            static void process_serial_image_runs(OCR_job_inputs &inputs, Image_results &results);

            void preprocess_image(const MPFImageJob &job, cv::Mat &input_image, const OCR_filter_settings &ocr_fset);
            void rescale_image(const MPFImageJob &job, cv::Mat &input_image, const OCR_filter_settings &ocr_fset);

            static void process_tesseract_lang_model(OCR_job_inputs &input, OCR_results  &result);

            void set_default_parameters();

            void set_read_config_parameters();

            void load_settings(const MPFJob &job, OCR_filter_settings &ocr_fset, const Text_type &text_type = Unknown);

            void load_tags_json(const MPFJob &job, MPFDetectionError &job_status,
                                std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex);

            bool comp_regex(const MPFImageJob &job, const std::wstring &full_text, const std::wstring &regstr,
                            std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,
                            const OCR_filter_settings &ocr_fset, bool case_sensitive,
                            MPFDetectionError &job_status);

            void process_regex_match(const boost::wsmatch &match, const std::wstring &full_text,
                                     std::map<std::wstring, std::vector<std::string>> &trigger_words_offset);

            void sharpen(cv::Mat &image, double weight);

            std::string parse_regex_code(boost::regex_constants::error_type etype);

            std::set<std::wstring> search_regex(const MPFImageJob &job, const std::wstring &full_text,
                                                const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex,
                                                std::map<std::wstring, std::vector<std::string>> &trigger_words_offset,
                                                const OCR_filter_settings &ocr_fset,
                                                MPFDetectionError &job_status);

            bool process_text_tagging(Properties &detection_properties, const MPFImageJob &job,
                                      const OCR_output &ocr_out, MPFDetectionError &job_status,
                                      const OCR_filter_settings &ocr_fset,
                                      const std::map<std::wstring, std::vector<std::pair<std::wstring, bool>>> &json_kvs_regex,
                                      int page_num = -1);

            static std::string process_osd_lang(const std::string &script_type,
                                                const OCR_filter_settings &ocr_fset);

            void get_OSD(OSResults &results, cv::Mat &imi, const MPFImageJob &job,
                         OCR_filter_settings &ocr_fset,
                         Properties &detection_properties,
                         std::string &tessdata_script_dir, std::set<std::string> &missing_languages);

            static std::string return_valid_tessdir(const std::string &job_name,
                                                    const std::string &lang_str,
                                                    const std::string &directory,
                                                    const std::string &run_directory,
                                                    log4cxx::LoggerPtr &logger,
                                                    const bool &secondary_pass,
                                                    std::set<std::string> &missing_languages,
                                                    std::set<std::string> &found_languages);

            static bool check_tess_model_directory(const std::string &job_name,
                                                   const std::string &lang_str,
                                                   const std::string &directory,
                                                   log4cxx::LoggerPtr &logger,
                                                   const bool &secondary_pass,
                                                   std::set<std::string> &missing_languages,
                                                   std::set<std::string> &found_languages);

            void check_default_languages(const OCR_filter_settings &ocr_fset,
                                         const std::string &job_name,
                                         const std::string &run_dir,
                                         MPFDetectionError &job_status);
        };

        // The primary reason this class exists is that tesseract::TessBaseAPI segfaults when copying or moving.
        // It is very easy to unintentionally invoke a copy constructor (e.g. forgetting a &),
        // so we disable copying and moving in this class.
        class TessApiWrapper {
        public:
            TessApiWrapper(const std::string& data_path, const std::string& language, tesseract::OcrEngineMode oem);

            TessApiWrapper(const TessApiWrapper&) = delete;
            TessApiWrapper& operator=(const TessApiWrapper&) = delete;
            TessApiWrapper(TessApiWrapper&&) = delete;
            TessApiWrapper& operator=(TessApiWrapper&&) = delete;

            void SetPageSegMode(tesseract::PageSegMode mode);

            void SetImage(const cv::Mat& image);

            std::string GetUTF8Text();

            int MeanTextConf();

            void Clear();

            bool DetectOS(OSResults* results);

        private:
            tesseract::TessBaseAPI tess_api_;
        };

    }
}

#endif
