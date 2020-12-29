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

#include "TesseractOCRTextDetection.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>

#include <boost/regex.hpp>
#include <boost/locale.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/sequenced_index.hpp>
#include <boost/multi_index/ordered_index.hpp>

#include <Magick++.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <tesseract/baseapi.h>
#include <tesseract/osdetect.h>
#include <tesseract/unicharset.h>

#include <detectionComponentUtils.h>
#include <MPFSimpleConfigLoader.h>
#include <Utils.h>


using namespace MPF;
using namespace COMPONENT;
using namespace std;
using log4cxx::Logger;
typedef boost::multi_index::multi_index_container<string,
        boost::multi_index::indexed_by<
                boost::multi_index::sequenced<>,
                boost::multi_index::ordered_unique<boost::multi_index::identity<string>>
        >> unique_vector;


class TempDirectory {
public:
    const std::string path;

    explicit TempDirectory(const std::string &path)
            : path(path) {
        bool created = boost::filesystem::create_directories(path);
        if (!created) {
            throw MPFDetectionException(
                    MPF_FILE_WRITE_ERROR,
                    "Unable to write temporary directory (may already exist): " + path);
        }
    }

    ~TempDirectory() {
        boost::filesystem::remove_all(path);
    }

    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
};


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


bool TesseractOCRTextDetection::Close() {
    return true;
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
    default_ocr_fset.max_pixels = 10000000;
    default_ocr_fset.max_scripts = 1;
    default_ocr_fset.max_text_tracks = 0;
    default_ocr_fset.min_secondary_script_thrs = 0.80;
    default_ocr_fset.rotate_and_detect = false;
    default_ocr_fset.rotate_and_detect_min_confidence = 95.0;

    default_ocr_fset.enable_hist_equalization = false;
    default_ocr_fset.enable_adaptive_hist_equalization = false;
    default_ocr_fset.invalid_min_image_size = 4;
    default_ocr_fset.min_height = -1;
    default_ocr_fset.adaptive_hist_tile_size = 5;
    default_ocr_fset.adaptive_hist_clip_limit = 2.0;

    default_ocr_fset.processing_wild_text = false;
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
    if (parameters.contains("INVALID_MIN_IMAGE_SIZE")) {
        default_ocr_fset.invalid_min_image_size = parameters["INVALID_MIN_IMAGE_SIZE"].toInt();
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
    if (parameters.contains("MAX_PIXELS")){
        default_ocr_fset.max_pixels = parameters["MAX_PIXELS"].toInt();
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
    if (parameters.contains("ENABLE_OSD_FALLBACK")) {
        default_ocr_fset.enable_osd_fallback = parameters["ENABLE_OSD_FALLBACK"].toInt() > 0;
    }
    if (parameters.contains("MAX_PARALLEL_SCRIPT_THREADS")) {
        default_ocr_fset.max_parallel_ocr_threads = parameters["MAX_PARALLEL_SCRIPT_THREADS"].toInt();
    }
    if (parameters.contains("MAX_PARALLEL_PAGE_THREADS")) {
        default_ocr_fset.max_parallel_pdf_threads = parameters["MAX_PARALLEL_PAGE_THREADS"].toInt();
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
set<string> generate_lang_set(const string& input) {
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
void TesseractOCRTextDetection::preprocess_image(const MPFImageJob &job, cv::Mat &image_data,
                                                 const OCR_filter_settings &ocr_fset) {
    if (image_data.empty()) {
        throw MPFDetectionException(MPF_IMAGE_READ_ERROR,
                                    "Could not open transformed image and will not return detections.");
    }
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Transformed image opened.");

    // Image preprocessing to improve text extraction results.

    try {
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

        // Image inversion.
        if (ocr_fset.invert) {
            double min, max;
            cv::Mat tmp_imb(image_data.size(), image_data.type());
            cv::minMaxLoc(image_data, &min, &max);
            tmp_imb.setTo(cv::Scalar::all(max));
            cv::subtract(tmp_imb, image_data, image_data);
        }
    }
    catch (const std::exception& ex) {
        throw MPFDetectionException(
                MPF_OTHER_DETECTION_ERROR_TYPE,
                std::string("Error during image preprocessing: ") + ex.what());
    }
}

/*
 * Rescales image before running OSD and OCR.
 */
void TesseractOCRTextDetection::rescale_image(const MPFImageJob &job, cv::Mat &image_data,
                                              const OCR_filter_settings &ocr_fset) {
    int im_width  = image_data.size().width;
    int im_height = image_data.size().height;

    double default_rescale = ocr_fset.scale;
    bool need_rescale = true;

    if (default_rescale < 0) {
        need_rescale = false;
        default_rescale = 1.0;
    }

    // Maximum acceptable image dimensions under Tesseract.
    int max_size = 0x7fff;

    if (im_height < ocr_fset.invalid_min_image_size) {
        throw MPFDetectionException(MPF_BAD_FRAME_SIZE,
                                    "Invalid image height, image too short: " + std::to_string(im_height));
    }

    if (im_width < ocr_fset.invalid_min_image_size) {
        throw MPFDetectionException(MPF_BAD_FRAME_SIZE,
                                    "Invalid image width, image too narrow: " + std::to_string(im_width));
    }

    int min_dim = im_width;
    int max_dim = im_height;

    if (im_width > im_height) {
        min_dim = im_height;
        max_dim = im_width;
    }

    // Update scaling if min_height check recommends upsampling.
    if (im_height * default_rescale < ocr_fset.min_height) {
        default_rescale = (double)ocr_fset.min_height / (double)im_height;
        need_rescale = true;
        LOG4CXX_INFO(hw_logger_, "[" + job.job_name + "] Attempting to increase image scaling to meet desired minimum image height.");

    }

    bool max_dim_exceeded = max_dim * default_rescale > max_size;
    bool max_pix_exceeded = ocr_fset.max_pixels > 0 &&
        (default_rescale * im_width) * (im_height * default_rescale) > ocr_fset.max_pixels;
    if (max_dim_exceeded || max_pix_exceeded) {
        // Users will most likely to request image upsampling.
        // If image size reaches tesseract limits or user sets pixel limits,
        // cap image upsampling to maximum allowed dimensions and pixels.

        // Warn user that down-sampling is occurring to meet Tesseract requirements.
        string warning_msg = "[" + job.job_name + "] Warning, resampling (" + to_string(im_width) + ", "
            + to_string(im_height) + ") sized image by recommended scaling factor would put";
        if (max_dim_exceeded) {
            warning_msg += " an image dimension above Tesseract limit of " + to_string(max_size) + " pixels.";
        } else if (max_pix_exceeded) {
            warning_msg += " image above limit of " + to_string(ocr_fset.max_pixels) + " max pixels.";
        }
        warning_msg += " Capping upsampling to meet Tesseract limits.";
        LOG4CXX_WARN(hw_logger_, warning_msg);
        need_rescale = true;

        default_rescale = (double)max_size / (double)max_dim;

        // If rescale still exceeds pixel limits, decrease further.
        if (ocr_fset.max_pixels > 0 &&
            (default_rescale * im_width) * (im_height * default_rescale) > ocr_fset.max_pixels) {
            default_rescale = std::sqrt((double)ocr_fset.max_pixels / (double)(im_height * im_width));
        }

        if (min_dim * default_rescale < ocr_fset.invalid_min_image_size) {
            string error_msg =  "[" + job.job_name + "] Unable to rescale image as one image dimension ({"
                + to_string(max_dim) + "}) would exceed maximum tesseract limits while the other dimension ({"
                + to_string(min_dim) + "}) would fall below minimum OCR-readable limits if rescaled to fit.";
            throw MPFDetectionException(MPF_BAD_FRAME_SIZE, error_msg);
        }
    } else if (min_dim * default_rescale < ocr_fset.invalid_min_image_size) {
        // Although users are unlikely to request image downsampling,
        // notify if downsampling would fall below minimum image limits.

        string warning_msg =  "[" + job.job_name + "] Warning, downsampling (" + to_string(im_width) + ", "
            + to_string(im_height) + ") sized image by requested scaling factor would put an image"
            + " dimension below minimum limit of " + to_string(ocr_fset.invalid_min_image_size)
            + " pixels. Rescaling to fit minimum OCR-readable limit.";
        LOG4CXX_WARN(hw_logger_, warning_msg);
        need_rescale = true;

        default_rescale = (double)ocr_fset.invalid_min_image_size / (double)min_dim;

        bool max_dim_exceeded = max_dim * default_rescale > max_size;
        bool max_pix_exceeded = ocr_fset.max_pixels > 0 &&
            (default_rescale * im_width) * (im_height * default_rescale) > ocr_fset.max_pixels;
        if (max_dim_exceeded || max_pix_exceeded) {
            string error_msg;
            if (max_dim_exceeded) {
                error_msg =  "[" + job.job_name + "] Unable to rescale image as one image dimension ({"
                    + to_string(max_dim) + "}) would exceed maximum tesseract limits while the other dimension ({"
                    + to_string(min_dim) + "}) would fall below minimum OCR-readable limits if rescaled to fit.";
            } else if (max_pix_exceeded) {
                error_msg =  "[" + job.job_name + "] Unable to rescale image within limit of "
                    + to_string(ocr_fset.max_pixels) + " max pixels.";
            }
            throw MPFDetectionException(MPF_BAD_FRAME_SIZE, error_msg);
        }
    }

    try {
        // Rescale and sharpen image (larger images improve detection results).
        if (need_rescale) {
            cv::resize(image_data, image_data, cv::Size(),  default_rescale,  default_rescale);
        }
        if (ocr_fset.sharpen > 0) {
            sharpen(image_data, ocr_fset.sharpen);
        }
    } catch (const std::exception& ex) {
        throw MPFDetectionException(MPF_OTHER_DETECTION_ERROR_TYPE,
                                    std::string("Error during image rescaling: ") + ex.what());
    }
}

void TesseractOCRTextDetection::process_tesseract_lang_model(OCR_job_inputs &inputs,
                                                             OCR_results  &results) {

    // Process language specified by user or script detected by OSD processing.
    pair<int, string> tess_api_key = make_pair(inputs.ocr_fset->oem, inputs.ocr_fset->model_dir + "/" + results.lang);
    LOG4CXX_DEBUG(inputs.hw_logger_, "[" + *inputs.job_name + "] Running Tesseract with specified language: " +
                                     results.lang);


    bool tess_api_in_map = inputs.tess_api_map->find(tess_api_key) != inputs.tess_api_map->end();
    string tessdata_dir;
    std::unique_ptr<TessApiWrapper> tess_api_for_parallel;
    set<string> languages_found, missing_languages;
    // If running OSD scripts, set tessdata_dir to the location found during OSD processing.
    // Otherwise, for each individual language setting, locate the appropriate tessdata directory.
    if (inputs.tessdata_script_dir->empty()) {
        // Left blank when OSD is not run or scripts not found and reverted to default language.
        // Check default language models are present.
        tessdata_dir = return_valid_tessdir(
                *inputs.job_name, results.lang, inputs.ocr_fset->model_dir, *inputs.run_dir, inputs.hw_logger_,
                true, missing_languages, languages_found);
    }
    else {
        tessdata_dir = *inputs.tessdata_script_dir;
    }
    if (!tess_api_in_map || inputs.parallel_processing) {
        // Fail if user specified language is missing.
        // This error should not occur as both OSD and user specified languages have been already checked.
        if (!missing_languages.empty()) {
            results.job_status = MPF_COULD_NOT_OPEN_DATAFILE;
            throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, "Tesseract language models not found.");
        }

        if (inputs.parallel_processing) {
            // When parallel processing is enabled, new APIs must be started up to avoid deadlocking issues rather than
            // using globally initialized APIs.
            tess_api_for_parallel.reset(
                    new TessApiWrapper(tessdata_dir, results.lang, (tesseract::OcrEngineMode) inputs.ocr_fset->oem));
        }
        else if (!tess_api_in_map) {
            inputs.tess_api_map->emplace(
                    std::piecewise_construct,
                    std::forward_as_tuple(tess_api_key),
                    std::forward_as_tuple(tessdata_dir, results.lang, (tesseract::OcrEngineMode) inputs.ocr_fset->oem));
        }
    } else if (!missing_languages.empty()) {
        LOG4CXX_WARN(inputs.hw_logger_, "[" + *inputs.job_name + "] Tesseract language models no longer found in " +
                                        "tessdata directory. " +
                                        "Reverting to cached models.");
    }

    TessApiWrapper &tess_api = inputs.parallel_processing
                               ? *tess_api_for_parallel
                               : inputs.tess_api_map->at(tess_api_key);

    tess_api.SetPageSegMode((tesseract::PageSegMode) inputs.ocr_fset->psm);
    tess_api.SetImage(*inputs.imi);
    std::string text = tess_api.GetUTF8Text();
    results.confidence = tess_api.MeanTextConf();

    // Free up recognition results and any stored image data.
    tess_api.Clear();

    if (!text.empty()) {
        results.text_result = std::move(text);
    } else {
        LOG4CXX_WARN(inputs.hw_logger_, "[" + *inputs.job_name + "] OCR processing unsuccessful, no outputs.");
    }
}

void TesseractOCRTextDetection::process_parallel_image_runs(OCR_job_inputs &inputs,
                                                            Image_results &results) {

    OCR_results ocr_results[inputs.ocr_lang_inputs.size()];
    std::future<void> ocr_threads[inputs.ocr_lang_inputs.size()];
    int index = 0;
    std::set<int> active_threads;

    // Initialize a new track for each language specified.
    for (const string &lang: inputs.ocr_lang_inputs) {
        ocr_results[index].job_status = results.job_status;
        ocr_results[index].lang = lang;
        ocr_threads[index] = std::async(launch::async,
                                        &process_tesseract_lang_model,
                                        std::ref(inputs),
                                        std::ref(ocr_results[index]));
        active_threads.insert(index);
        index++;
        while (active_threads.size() >= inputs.ocr_fset->max_parallel_ocr_threads) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            for (auto active_thread_iter = active_threads.begin(); active_thread_iter != active_threads.end();
                /* Intentionally no ++ to support erasing. */) {
                int i = *active_thread_iter;
                if (ocr_threads[i].wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    // Will re-throw exception from thread.
                    ocr_threads[i].get();
                    active_thread_iter = active_threads.erase(active_thread_iter);
                }
                else {
                    ++active_thread_iter;
                }
            }
        }
    }

    // Wait for all remaining threads to finish.
    for (const int &i: active_threads) {
        // Will re-throw exception from thread.
        ocr_threads[i].get();
    }

    for (OCR_results &ocr_res: ocr_results) {
        LOG4CXX_DEBUG(inputs.hw_logger_, "[" + *inputs.job_name + "] Tesseract run successful.");
        wstring t_detection = boost::locale::conv::utf_to_utf<wchar_t>(ocr_res.text_result);

        OCR_output output_ocr = {ocr_res.confidence, ocr_res.lang, t_detection, "", false};
        results.detections_by_lang.push_back(output_ocr);
    }
}


void TesseractOCRTextDetection::process_serial_image_runs(OCR_job_inputs &inputs,
                                                          Image_results &results) {
    for (const string &lang: inputs.ocr_lang_inputs) {
        OCR_results ocr_results;

        ocr_results.job_status = results.job_status;
        ocr_results.lang = lang;

        process_tesseract_lang_model(inputs, ocr_results);
        LOG4CXX_DEBUG(inputs.hw_logger_, "[" + *inputs.job_name + "] Tesseract run successful.");
        wstring t_detection = boost::locale::conv::utf_to_utf<wchar_t>(ocr_results.text_result);

        OCR_output output_ocr = {ocr_results.confidence, ocr_results.lang, t_detection, "", false};
        results.detections_by_lang.push_back(output_ocr);
    }
}

bool TesseractOCRTextDetection::process_ocr_text(Properties &detection_properties, const MPFImageJob &job,
                                                 const TesseractOCRTextDetection::OCR_output &ocr_out,
                                                 const TesseractOCRTextDetection::OCR_filter_settings &ocr_fset,
                                                 int page_num) {

    string ocr_lang = ocr_out.language;
    wstring full_text = ocr_out.text;
    full_text = clean_whitespace(full_text);

    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing text for Tesseract OCR output: ");
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Tesseract OCR output was: " +
                              boost::locale::conv::utf_to_utf<char>(full_text));

    if (is_only_ascii_whitespace(full_text)) {
        LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] No text found in image!");
        return false;
    }

    detection_properties["TEXT_LANGUAGE"] = ocr_lang;
    detection_properties["TEXT"] = boost::locale::conv::utf_to_utf<char>(full_text);

    if (page_num >= 0) {
        detection_properties["PAGE_NUM"] = to_string(page_num + 1);
    }
    return true;
}

void TesseractOCRTextDetection::get_tesseract_detections(OCR_job_inputs &inputs, Image_results &results) {
    inputs.ocr_lang_inputs = generate_lang_set(*inputs.lang);
    // Check multithreading for multiple tracks.
    if (inputs.ocr_fset->max_parallel_ocr_threads > 1 && inputs.ocr_lang_inputs.size() > 1 && !inputs.process_pdf) {
        // Parallel ocr or multiple parallel track processing.
        // Disable usage of global tesseract APIs during parallel processing.
        inputs.parallel_processing = true;
        process_parallel_image_runs(inputs, results);
    } else {
        // Serial ocr or single track processing.
        if (inputs.ocr_fset->max_parallel_pdf_threads > 1 && inputs.process_pdf) {
            // Disable usage of global tesseract APIs during parallel processing of PDF images.
            inputs.parallel_processing = true;
        } else {
            inputs.parallel_processing = false;
        }
        process_serial_image_runs(inputs, results);
    }
}

string TesseractOCRTextDetection::return_valid_tessdir(const string &job_name, const string &lang_str,
                                                       const string &directory, const string &run_directory,
                                                       log4cxx::LoggerPtr &hw_logger_, const bool &secondary_pass,
                                                       set<string> &missing_languages, set<string> &found_languages) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Checking tessdata models in " + directory + ".");
    set<string> missing_first, missing_second;
    set<string> partial_first, partial_second;

    if (check_tess_model_directory(job_name, lang_str, directory, hw_logger_, secondary_pass, missing_first, partial_first)) {
        found_languages.insert(partial_first.begin(), partial_first.end());
        return directory;
    }

    string run_dir = run_directory;
    if (run_dir.empty()) {
        run_dir = ".";
    }

    // If user specified tessdata directory fails, revert to default plugin directory and tessdata models.
    string local_plugin_directory = run_dir + "/TesseractOCRTextDetection/tessdata";
    LOG4CXX_DEBUG(hw_logger_,
                  "[" + job_name + "] Not all models found in " + directory + ". Checking local plugin directory "
                  + local_plugin_directory + ".");

    if (check_tess_model_directory(job_name, lang_str, local_plugin_directory, hw_logger_, secondary_pass,
                                   missing_second, partial_second)) {
        found_languages.insert(partial_second.begin(), partial_second.end());
        return local_plugin_directory;
    }


    LOG4CXX_WARN(hw_logger_, "[" + job_name + "] Tessdata models were not found, or are not co-located." +
                             " Please ensure that the following models exist in the same directory, either " +
                             directory + " or " +
                             local_plugin_directory + ": " + lang_str + ".");

    // Return directory with greater number of valid scripts.
    if (partial_first.size() > 0 || partial_second.size() > 0) {
        if  (partial_first.size() > partial_second.size()) {
            found_languages.insert(partial_first.begin(), partial_first.end());
            missing_languages.insert(missing_first.begin(), missing_first.end());
            return directory;
        } else {
            found_languages.insert(partial_second.begin(), partial_second.end());
            missing_languages.insert(missing_second.begin(), missing_second.end());
            return local_plugin_directory;
        }
    }
    missing_languages.insert(missing_first.begin(), missing_first.end());
    return "";
}

bool TesseractOCRTextDetection::check_tess_model_directory(const string &job_name, const string &lang_str,
                                                           const string &directory, log4cxx::LoggerPtr &hw_logger_,
                                                           const bool &secondary_pass,
                                                           set<string> &missing, set<string> &partial_match) {

    set<string> langs;
    boost::split(langs, lang_str, boost::is_any_of(",+"));
    bool status = true;

    for (string lang : langs) {
        boost::trim(lang);
        // Don't search for invalid NULL language.
        if (lang == "NULL" || lang == "script/NULL") {
            continue;
        }
        if (!boost::filesystem::exists(directory + "/" + lang + ".traineddata")) {
            LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Tessdata file " + lang + ".traineddata does not exist in "
                                      + directory + ".");
            missing.insert(lang);
            status = false;
        } else {
            partial_match.insert(lang);
            LOG4CXX_DEBUG(hw_logger_,
                          "[" + job_name + "] Tessdata file " + lang + ".traineddata found in  " + directory + ".");
        }
    }
    if (status && !secondary_pass) {
        LOG4CXX_INFO(hw_logger_,
                     "[" + job_name + "] Tessdata models found in " + directory + ": " + lang_str + ".");
    }

    return status;
}

string TesseractOCRTextDetection::process_osd_lang(const string &script_type, const OCR_filter_settings &ocr_fset) {
    if (script_type == "Han") {
        if (ocr_fset.combine_detected_scripts) {
            return "script/HanS+script/HanT+script/HanS_vert+script/HanT_vert";
        } else {
            return "script/HanS+script/HanT,script/HanS_vert+script/HanT_vert";
        }
    } else if (script_type == "Korean" || script_type == "Hangul") {
        if (ocr_fset.combine_detected_scripts) {
            return "script/Hangul+script/Hangul_vert";
        } else {
            return "script/Hangul,script/Hangul_vert";
        }

    } else if ((script_type == "Japanese" || script_type == "Hiragana" || script_type == "Katakana")) {
        if (ocr_fset.combine_detected_scripts) {
            return "script/Japanese+script/Japanese_vert";
        } else {
            return "script/Japanese,script/Japanese_vert";
        }
    } else if (script_type == "Common" || script_type == "NULL") {
        return "";
    } else {
        return "script/" + script_type;
    }
}

// Get OSD orientation scores and convert them into ROTATION values.
// Both the scaled imi image and unscaled imi_original image are also needed since vertical rotations may result in
// changes to the original image rescaling. The final rescaled image will be stored in imi_scaled.
bool TesseractOCRTextDetection::get_OSD_rotation(OSResults *results, cv::Mat &imi_scaled, cv::Mat &imi_original,
                                                 int &rotation, const MPFImageJob &job, OCR_filter_settings &ocr_fset) {

    switch (results->best_result.orientation_id) {
        case 0:
            // Do not rotate. Use the scaled image provided.
            break;
        case 1:
            // Text is rotated 270 degrees counterclockwise.
            // Image will need to be rotated 90 degrees counterclockwise to fix this.
            // Reapply rescaling as rotation has changed image dimensions.
            cv::rotate(imi_original, imi_scaled, cv::ROTATE_90_COUNTERCLOCKWISE);
            try {
                rescale_image(job, imi_scaled, ocr_fset);
            }
            catch (const MPFDetectionException &ex) {
                return false;
            }
            // Report current orientation of image (uncorrected, counterclockwise).
            rotation = 270;
            break;
        case 2:
            // 180 degree rotation.
            // Since width and height are unchanged, no rescaling is needed.
            cv::rotate(imi_scaled, imi_scaled, cv::ROTATE_180);
            rotation = 180;
            break;
        case 3:
            // Text is rotated 90 degrees counterclockwise.
            // Image will need to be rotated 270 degrees counterclockwise to fix this.
            // Reapply rescaling as rotation has changed image dimensions.
            cv::rotate(imi_original, imi_scaled, cv::ROTATE_90_CLOCKWISE);
            try {
                rescale_image(job, imi_scaled, ocr_fset);
            }
            catch (const MPFDetectionException &ex) {
                return false;
            }

            // Report current orientation of image (uncorrected, counterclockwise).
            rotation = 90;
            break;
        default:
            break;
        }
    return true;
}

void TesseractOCRTextDetection::get_OSD(OSBestResult &best_result, cv::Mat &imi, const MPFImageJob &job,
                                        OCR_filter_settings &ocr_fset,
                                        Properties &detection_properties,
                                        string &tessdata_script_dir, set<string> &missing_languages) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Running Tesseract OSD.");

    int oem = 3;
    pair<int, string> tess_api_key = make_pair(oem, ocr_fset.model_dir + "/" + "osd");
    set<string> found_languages;
    string run_dir = GetRunDirectory();
    OSResults original_results, fallback_results;
    OSResults *results;

    // Preserve a copy for images that may swap width and height. Rescaling will be performed based on new dimensions.
    cv::Mat imi_copy = imi.clone();

    // Default values for empty OSD result.
    // If OSD failed to detect any scripts, automatically set confidence scores to -1 (not enough text).
    detection_properties["OSD_PRIMARY_SCRIPT"] = "NULL";
    detection_properties["OSD_PRIMARY_SCRIPT_CONFIDENCE"] = "-1";
    detection_properties["OSD_TEXT_ORIENTATION_CONFIDENCE"] = "-1";
    detection_properties["OSD_PRIMARY_SCRIPT_SCORE"]  = "-1";
    detection_properties["ROTATION"] = "0";
    detection_properties["OSD_FALLBACK_OCCURRED"] = "false";

    // Rescale original image and process through OSD.
    rescale_image(job, imi, ocr_fset);

    if (tess_api_map.find(tess_api_key) == tess_api_map.end()) {
        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Loading OSD model.");

        string tessdata_dir = return_valid_tessdir(job.job_name, "osd", ocr_fset.model_dir,
                                                   run_dir, hw_logger_, false, missing_languages,
                                                   found_languages);
        if (tessdata_dir.empty()) {
            throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE, "OSD model not found!");
        }

        tess_api_map.emplace(
                std::piecewise_construct,
                std::forward_as_tuple(tess_api_key),
                std::forward_as_tuple(tessdata_dir, "osd", (tesseract::OcrEngineMode) oem));

        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] OSD model ready.");
    }

    found_languages.clear();

    auto& tess_api = tess_api_map.at(tess_api_key);
    tess_api.SetPageSegMode(tesseract::PSM_AUTO_OSD);
    tess_api.SetImage(imi);
    if (!tess_api.DetectOS(&original_results)) {
        LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] OSD processing returned no outputs.");
    }
    // Free up recognition results and any stored image data.
    tess_api.Clear();

    // Store initial script result and set rotation if orientation confidence threshold is met.
    results = &original_results;
    string initial_script = results->unicharset->get_script_from_script_id(results->best_result.script_id);
    best_result = original_results.best_result;
    int candidates = 0;
    double best_score = original_results.scripts_na[best_result.orientation_id][best_result.script_id];
    int rotation = 0;

    // If OSD fall back is enabled and initial script results are unacceptable, rerun OSD on amplified image.
    if (ocr_fset.enable_osd_fallback && (initial_script == "NULL" ||
        !(ocr_fset.min_script_confidence <= best_result.sconfidence &&
          ocr_fset.min_script_score <= best_score &&
          ocr_fset.min_orientation_confidence <= best_result.oconfidence))) {

        LOG4CXX_INFO(hw_logger_, "[" + job.job_name + "] Attempting OSD processing with image amplification.");

        // Replicate the image, four times across.
        // Do not pass back MPF frame errors if amplified image cannot fit.
        cv::Mat amplified_im = cv::repeat(imi, 1, 4);
        // Double check that amplified image still fits OCR dimension limits.
        // Restore scaling to original settings after check is complete.
        double original_scale = ocr_fset.scale;
        ocr_fset.scale = -1;
        try {
            rescale_image(job, amplified_im, ocr_fset);
        }
        catch (const MPFDetectionException &ex) {
            LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] OSD processing with image amplification not possible "
                                      "within image size limits. Aborting second pass of OSD.");
            ocr_fset.scale = original_scale;
            return;
        }

        ocr_fset.scale = original_scale;
        tess_api.SetPageSegMode(tesseract::PSM_AUTO_OSD);
        tess_api.SetImage(amplified_im);
        tess_api.DetectOS(&fallback_results);

        // Free up recognition results and any stored image data.
        tess_api.Clear();
        double best_fallback_score = fallback_results.scripts_na[fallback_results.best_result.orientation_id]
                                                                [fallback_results.best_result.script_id];

        // Update best results after performing OSD fallback if all results are acceptable.
        if (ocr_fset.min_script_confidence <= fallback_results.best_result.sconfidence &&
            ocr_fset.min_script_score <= best_fallback_score &&
            ocr_fset.min_orientation_confidence <= fallback_results.best_result.oconfidence) {

            detection_properties["OSD_FALLBACK_OCCURRED"] = "true";
            results = &fallback_results;
            best_result = fallback_results.best_result;
            best_score = best_fallback_score;
        }
    }

    // Set rotation and rescale image if necessary.
    if (ocr_fset.min_orientation_confidence <= best_result.oconfidence) {
        if (!get_OSD_rotation(results, imi, imi_copy, rotation, job, ocr_fset))
        {
            return;
        }
    }

    // Store OSD results.
    detection_properties["OSD_PRIMARY_SCRIPT"] = results->unicharset->get_script_from_script_id(best_result.script_id);
    if (detection_properties["OSD_PRIMARY_SCRIPT"] != "NULL") {
        detection_properties["OSD_PRIMARY_SCRIPT_CONFIDENCE"] = to_string(best_result.sconfidence);
        detection_properties["OSD_PRIMARY_SCRIPT_SCORE"] = to_string(best_score);
        detection_properties["ROTATION"] = to_string(rotation);
        detection_properties["OSD_TEXT_ORIENTATION_CONFIDENCE"] = to_string(best_result.oconfidence);
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

    if (ocr_fset.min_script_confidence <= best_result.sconfidence && ocr_fset.min_script_score <= best_score) {

        OSD_script best_script = {best_result.script_id, best_score};
        vector<OSD_script> script_list, backup_scripts, secondary_scripts;

        // If primary script is not NULL, check if secondary scripts are also valid.
        if (detection_properties["OSD_PRIMARY_SCRIPT"] != "NULL") {
            // Max number of scripts in ICU + "NULL" + Japanese and Korean + Fraktur
            const int kMaxNumberOfScripts = 116 + 1 + 2 + 1;
            double score_cutoff = best_score * ocr_fset.min_secondary_script_thrs;
            if (score_cutoff < ocr_fset.min_script_score) {
                score_cutoff = ocr_fset.min_script_score;
            }
            // Skip index 0 to ignore the "Common" script.
            for (int i = 1; i < kMaxNumberOfScripts; ++i) {
                double score = results->scripts_na[best_result.orientation_id][i];
                if (i != best_result.script_id && score >= score_cutoff) {
                    OSD_script script_out;
                    script_out.id = i;
                    script_out.score = score;
                    script_list.push_back(script_out);
                    candidates += 1;
                }

            }
            // Sort out list of acceptable scripts.
            sort(script_list.begin(), script_list.end(), greater<OSD_script>());
        }

        // Limit number of accepted scripts if user set max_scripts to 2 or greater.
        // Unlimited number when users sets max_scripts to 0 or below.
        if (max_scripts > 0 && script_list.size() > max_scripts - 1) {
            secondary_scripts.insert(secondary_scripts.begin(),
                                     script_list.begin(), script_list.begin() + max_scripts - 1);
            backup_scripts.insert(backup_scripts.begin(),
                                  script_list.begin() + max_scripts - 1, script_list.end());
            script_list.resize(max_scripts - 1);
        } else {
            secondary_scripts = script_list;
        }

        // Include best script result and move it to the front.
        script_list.push_back(best_script);
        rotate(script_list.rbegin(), script_list.rbegin() + 1, script_list.rend());

        unique_vector script_results;
        int missing_count = 0;
        for (const OSD_script &script : script_list) {
            string lang = process_osd_lang(results->unicharset->get_script_from_script_id(script.id), ocr_fset);
            if (lang == "") {
                continue;
            }

            tessdata_script_dir = return_valid_tessdir(job.job_name, lang,
                                                       ocr_fset.model_dir,
                                                       run_dir,
                                                       hw_logger_,
                                                       false,
                                                       missing_languages,
                                                       found_languages);
            if (tessdata_script_dir == "") {
                missing_count += 1;
                continue;
            } else {
                script_results.push_back(lang);
            }

        }

        while (backup_scripts.size() > 0 && missing_count > 0) {
            OSD_script next_best = backup_scripts.front();
            backup_scripts.erase(backup_scripts.begin());

            string lang = process_osd_lang(results->unicharset->get_script_from_script_id(next_best.id), ocr_fset);
            if (lang == "") {
                continue;
            }
            tessdata_script_dir = return_valid_tessdir(job.job_name, lang,
                                                       ocr_fset.model_dir,
                                                       run_dir,
                                                       hw_logger_,
                                                       false,
                                                       missing_languages,
                                                       found_languages);
            if (tessdata_script_dir == "") {
                continue;
            } else {
                missing_count -= 1;
                script_results.push_back(lang);
                secondary_scripts.push_back(next_best);
            }
        }

        // Store OSD results.
        if (secondary_scripts.size() > 0) {

            vector<string> scripts;
            vector<string> scores;

            for (const OSD_script &script : secondary_scripts) {
                scripts.push_back(results->unicharset->get_script_from_script_id(script.id));
                scores.push_back(to_string(script.score));
            }

            detection_properties["OSD_SECONDARY_SCRIPTS"] = boost::algorithm::join(scripts, ", ");
            detection_properties["OSD_SECONDARY_SCRIPT_SCORES"] = boost::algorithm::join(scores, ", ");
        }

        string lang_str;
        if (ocr_fset.combine_detected_scripts) {
            lang_str = boost::algorithm::join(script_results, "+");
        } else {
            lang_str = boost::algorithm::join(script_results, ",");
        }

        found_languages.clear();
        if (lang_str.empty() || lang_str == "script/NULL") {
            LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] OSD could not find all required script models,"
                                     + " reverting to default language setting: " + ocr_fset.tesseract_lang);
            tessdata_script_dir = "";
        } else {
            // Check if selected models are present in either models or tessdata directory.
            // All language models must be present in one directory.
            // If scripts are not found, revert to default.
            // If scripts are found, set return_valid_tessdir so that get_tesseract_detections will skip searching for models.
            // Also, for script detection results, skip searching for NULL if that gets detected alongside other languages.
            tessdata_script_dir = return_valid_tessdir(job.job_name, lang_str,
                                                       ocr_fset.model_dir,
                                                       run_dir,
                                                       hw_logger_,
                                                       true,
                                                       missing_languages,
                                                       found_languages);
            if (missing_languages.size() > 0) {
                // If some acceptable OSD scripts were found, use them instead.
                if (found_languages.size() > 0) {
                    string join_type = ",";
                    if (ocr_fset.combine_detected_scripts) {
                        join_type = "+";
                    }
                    ocr_fset.tesseract_lang = boost::algorithm::join(found_languages, join_type);
                } else {
                    LOG4CXX_WARN(hw_logger_, "[" + job.job_name + "] Script models not found in model and tessdata directories,"
                                             + " switching to default language setting: " + ocr_fset.tesseract_lang);
                }
                // Reset script directory, run check again while processing with default languages.
                tessdata_script_dir = "";

            } else {
                ocr_fset.tesseract_lang = lang_str;
            }
        }
    }

}


void
TesseractOCRTextDetection::load_settings(const MPFJob &job, OCR_filter_settings &ocr_fset,
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
    ocr_fset.invalid_min_image_size = DetectionComponentUtils::GetProperty<int>(job.job_properties, "INVALID_MIN_IMAGE_SIZE", default_ocr_fset.invalid_min_image_size);
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
    ocr_fset.max_pixels = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MAX_PIXELS", default_ocr_fset.max_pixels);
    ocr_fset.max_text_tracks = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MAX_TEXT_TRACKS", default_ocr_fset.max_text_tracks);
    ocr_fset.min_secondary_script_thrs = DetectionComponentUtils::GetProperty<double>(job.job_properties,"MIN_OSD_SECONDARY_SCRIPT_THRESHOLD", default_ocr_fset.min_secondary_script_thrs);
    ocr_fset.rotate_and_detect = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ROTATE_AND_DETECT", default_ocr_fset.rotate_and_detect);
    ocr_fset.rotate_and_detect_min_confidence = DetectionComponentUtils::GetProperty<double>(job.job_properties, "ROTATE_AND_DETECT_MIN_OCR_CONFIDENCE", default_ocr_fset.rotate_and_detect_min_confidence);
    ocr_fset.enable_osd_fallback = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_OSD_FALLBACK", default_ocr_fset.enable_osd_fallback);
    ocr_fset.max_parallel_ocr_threads = DetectionComponentUtils::GetProperty<int>(job.job_properties, "MAX_PARALLEL_SCRIPT_THREADS", default_ocr_fset.max_parallel_ocr_threads);
    ocr_fset.max_parallel_pdf_threads = DetectionComponentUtils::GetProperty<int>(job.job_properties, "MAX_PARALLEL_PAGE_THREADS", default_ocr_fset.max_parallel_pdf_threads);

    // Tessdata setup
    ocr_fset.model_dir =  DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "MODELS_DIR_PATH", default_ocr_fset.model_dir);
    ocr_fset.tessdata_models_subdir =  DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "TESSDATA_MODELS_SUBDIRECTORY", default_ocr_fset.tessdata_models_subdir);
    if (ocr_fset.model_dir != "") {
        ocr_fset.model_dir = ocr_fset.model_dir + "/" + ocr_fset.tessdata_models_subdir;
    } else {
        string run_dir = GetRunDirectory();
        if (run_dir.empty()) {
            run_dir = ".";
        }
        // If not specified, set model dir to local plugin dir.
        ocr_fset.model_dir = run_dir + "/" + ocr_fset.tessdata_models_subdir;
    }
    Utils::expandFileName(ocr_fset.model_dir, ocr_fset.model_dir);
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

/*
 * Confirm each language model is present in tessdata or shared model directory.
 * Language models that run together must be present in the same directory.
 */
void TesseractOCRTextDetection::check_default_languages(const OCR_filter_settings &ocr_fset,
                                                        const string &job_name,
                                                        const string &run_dir,
                                                        MPFDetectionError &job_status) {

    set<string> languages_found, missing_languages;
    set<string> lang_list = generate_lang_set(ocr_fset.tesseract_lang);
    bool tess_api_in_map = true;

    // Check that all tessdata files are present in models directory or cached in memory.
    for (const string &lang: lang_list) {

        return_valid_tessdir(job_name, lang, ocr_fset.model_dir, run_dir, hw_logger_, false, missing_languages, languages_found);

        pair<int, string> tess_api_key = make_pair(ocr_fset.oem, ocr_fset.model_dir + "/" + lang);
        tess_api_in_map = tess_api_in_map && (tess_api_map.find(tess_api_key) != tess_api_map.end());
    }

    if (!missing_languages.empty()) {
        bool parallel_processing = (ocr_fset.max_parallel_ocr_threads > 1) || (ocr_fset.max_parallel_pdf_threads > 1);

        // Fail if cached models do not exist or cannot be used due to parallel processing.
        if (!tess_api_in_map || parallel_processing) {
            throw MPFDetectionException(MPF_COULD_NOT_OPEN_DATAFILE,
                                        "User-specified Tesseract language models not found.");
        } else {
            LOG4CXX_WARN(hw_logger_, "[" + job_name + "] User-specified Tesseract language models no longer found. " +
                                      "Using cached language models.");
        }
    }
}


vector<MPFImageLocation> TesseractOCRTextDetection::GetDetections(const MPFImageJob &job) {
    LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");
    try{
        OCR_filter_settings ocr_fset;
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

        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] About to run tesseract");
        vector<OCR_output> ocr_outputs;

        MPFImageReader image_reader(job);
        cv::Mat image_data = image_reader.GetImage();
        cv::Mat image_data_rotated;
        cv::Size input_size = image_data.size();

        string run_dir = GetRunDirectory();

        check_default_languages(ocr_fset, job.job_name, run_dir, job_status);
        preprocess_image(job, image_data, ocr_fset);

        Properties osd_detection_properties;
        string tessdata_script_dir = "";
        set<string> missing_languages;
        int xLeftUpper = 0;
        int yLeftUpper = 0;
        int width = input_size.width;
        int height = input_size.height;
        int orientation_result = 0;
        vector<MPFImageLocation> locations;

        if (ocr_fset.psm == 0 || ocr_fset.enable_osd) {

            OSBestResult os_best_result;
            get_OSD(os_best_result, image_data, job, ocr_fset, osd_detection_properties,
                    tessdata_script_dir, missing_languages);

            // Rotate upper left coordinates based on OSD results.
            if (ocr_fset.min_orientation_confidence <= os_best_result.oconfidence) {
                orientation_result = os_best_result.orientation_id;
                set_coordinates(xLeftUpper, yLeftUpper, width, height, input_size, os_best_result.orientation_id);
            }

            // When PSM is set to 0, there is no need to process any further.
            if (ocr_fset.psm == 0) {
                LOG4CXX_INFO(hw_logger_,
                             "[" + job.job_name + "] Processing complete. Found " + to_string(locations.size()) +
                             " tracks.");
                osd_detection_properties["MISSING_LANGUAGE_MODELS"] = boost::algorithm::join(missing_languages, ", ");
                MPFImageLocation osd_detection(xLeftUpper, yLeftUpper, width, height, -1, osd_detection_properties);
                locations.push_back(osd_detection);
                return locations;
            }
        } else {
            // If OSD is not run, the image won't be rescaled yet.
            rescale_image(job, image_data, ocr_fset);
        }
        osd_detection_properties["MISSING_LANGUAGE_MODELS"] = boost::algorithm::join(missing_languages, ", ");

        set<string> remaining_languages;
        string first_pass_rotation, second_pass_rotation;
        double min_ocr_conf = ocr_fset.rotate_and_detect_min_confidence;
        int corrected_orientation;
        int corrected_X, corrected_Y, corrected_width, corrected_height;

        if (ocr_fset.rotate_and_detect) {
            cv::rotate(image_data, image_data_rotated, cv::ROTATE_180);
            remaining_languages = generate_lang_set(ocr_fset.tesseract_lang);
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

        OCR_job_inputs ocr_job_inputs;
        ocr_job_inputs.job_name = &job.job_name;
        ocr_job_inputs.lang = &ocr_fset.tesseract_lang;
        ocr_job_inputs.tessdata_script_dir = &tessdata_script_dir;
        ocr_job_inputs.run_dir = &run_dir;
        ocr_job_inputs.imi = &image_data;
        ocr_job_inputs.ocr_fset = &ocr_fset;
        ocr_job_inputs.process_pdf = false;
        ocr_job_inputs.hw_logger_ = hw_logger_;
        ocr_job_inputs.tess_api_map = &tess_api_map;

        Image_results image_results;
        image_results.job_status = job_status;
        get_tesseract_detections(ocr_job_inputs, image_results);
        ocr_outputs = image_results.detections_by_lang;
        job_status = image_results.job_status;

        vector<OCR_output> all_results;

        for (const OCR_output &ocr_out: ocr_outputs) {
            OCR_output final_out = ocr_out;
            if (ocr_fset.rotate_and_detect) {
                remaining_languages.erase(ocr_out.language);
                final_out.two_pass_rotation = first_pass_rotation;
                final_out.two_pass_correction = false;

                // Perform second pass OCR if min threshold is disabled (negative) or first pass confidence too low.
                if (min_ocr_conf <= 0 || ocr_out.confidence < min_ocr_conf) {
                    // Perform second pass OCR and provide best result to output.
                    vector<OCR_output> ocr_outputs_rotated;
                    ocr_fset.tesseract_lang = ocr_out.language;
                    ocr_job_inputs.lang = &ocr_fset.tesseract_lang;
                    ocr_job_inputs.imi = &image_data_rotated;
                    image_results.detections_by_lang.clear();
                    image_results.job_status = job_status;

                    get_tesseract_detections(ocr_job_inputs, image_results);
                    ocr_outputs_rotated = image_results.detections_by_lang;
                    job_status = image_results.job_status;

                    OCR_output  ocr_out_rotated = ocr_outputs_rotated.front();
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
        for (const string &rem_lang: remaining_languages) {
            // Perform second pass OCR and provide best result to output.
            vector<OCR_output> ocr_outputs_rotated;
            ocr_fset.tesseract_lang = rem_lang;
            ocr_job_inputs.lang = &ocr_fset.tesseract_lang;
            ocr_job_inputs.imi = &image_data_rotated;
            image_results.detections_by_lang.clear();
            image_results.job_status = job_status;

            get_tesseract_detections(ocr_job_inputs, image_results);
            ocr_outputs_rotated = image_results.detections_by_lang;
            job_status = image_results.job_status;

            OCR_output ocr_out_rotated = ocr_outputs_rotated.front();
            ocr_out_rotated.two_pass_rotation = second_pass_rotation;
            ocr_out_rotated.two_pass_correction = true;
            all_results.push_back(ocr_out_rotated);
        }

        // If max_text_tracks is set, filter out to return only the top specified tracks.
        if (ocr_fset.max_text_tracks > 0) {
            sort(all_results.begin(), all_results.end(), greater<OCR_output>());
            all_results.resize(ocr_fset.max_text_tracks);
        }

        for (const OCR_output &final_out : all_results) {
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
            bool process_text = process_ocr_text(image_location.detection_properties, job, final_out,
                                                 ocr_fset);
            if (process_text) {
                locations.push_back(image_location);
            }
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        LOG4CXX_INFO(hw_logger_,
                     "[" + job.job_name + "] Processing complete. Found " + to_string(locations.size()) + " tracks.");
        return locations;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, hw_logger_);
    }
}


void TesseractOCRTextDetection::process_parallel_pdf_pages(PDF_page_inputs &page_inputs,
                                                           PDF_page_results &page_results) {
    PDF_thread_variables thread_var[page_inputs.filelist.size()];
    std::future<void> pdf_threads[page_inputs.filelist.size()];
    int index = 0;
    set<int> active_threads;

    for (const string &filename : page_inputs.filelist) {
        thread_var[index].page_thread_res.job_status = page_results.job_status;
        MPFImageJob c_job((*page_inputs.job).job_name,
                          filename,
                          (*page_inputs.job).job_properties,
                          (*page_inputs.job).media_properties);

        MPFImageReader image_reader(c_job);
        thread_var[index].image = image_reader.GetImage();
        preprocess_image(c_job, thread_var[index].image, page_inputs.ocr_fset);

        thread_var[index].osd_track_results = MPFGenericTrack(-1.0);
        thread_var[index].tessdata_script_dir = "";
        if (page_inputs.ocr_fset.enable_osd) {
            OSBestResult os_best_result;
            // Reset to original specified language before processing OSD.
            page_inputs.ocr_fset.tesseract_lang = page_inputs.default_lang;
            set<string> missing_languages;
            get_OSD(os_best_result, thread_var[index].image, c_job, page_inputs.ocr_fset, thread_var[index].osd_track_results.detection_properties,
                    thread_var[index].tessdata_script_dir, missing_languages);
            page_results.all_missing_languages.insert(missing_languages.begin(), missing_languages.end());
        }
        else {
            // If OSD is not run, the image won't be rescaled yet.
            rescale_image(c_job, thread_var[index].image, page_inputs.ocr_fset);
        }

        thread_var[index].lang = page_inputs.ocr_fset.tesseract_lang;
        thread_var[index].ocr_input.job_name = &(*page_inputs.job).job_name;
        thread_var[index].ocr_input.run_dir = &page_inputs.run_dir;
        thread_var[index].ocr_input.ocr_fset = &page_inputs.ocr_fset;
        thread_var[index].ocr_input.process_pdf = true;
        thread_var[index].ocr_input.hw_logger_ = hw_logger_;
        thread_var[index].ocr_input.tess_api_map = &tess_api_map;
        pdf_threads[index] = std::async(launch::async,
                                        &get_tesseract_detections,
                                        std::ref(thread_var[index].ocr_input),
                                        std::ref(thread_var[index].page_thread_res));
        active_threads.insert(index);
        index ++;
        while (active_threads.size() >= page_inputs.ocr_fset.max_parallel_pdf_threads) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            for (auto active_thread_iter = active_threads.begin(); active_thread_iter != active_threads.end();
                /* Intentionally no ++ to support erasing. */) {
                int i = *active_thread_iter;
                if (pdf_threads[i].wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                    // Will re-throw exception from thread.
                    pdf_threads[i].get();
                    active_thread_iter = active_threads.erase(active_thread_iter);
                }
                else {
                    ++active_thread_iter;
                }
            }
        }
    }

    // Wait for all remaining threads to finish.
    for (const int &i: active_threads) {
        // Will re-throw exception from thread.
        pdf_threads[i].get();
    }

    for (index = 0; index < page_inputs.filelist.size(); index++ ) {
        MPFImageJob c_job((*page_inputs.job).job_name, "", (*page_inputs.job).job_properties, (*page_inputs.job).media_properties);

        // If max_text_tracks is set, filter out to return only the top specified tracks.
        if (page_inputs.ocr_fset.max_text_tracks > 0) {
            sort(thread_var[index].page_thread_res.detections_by_lang.begin(),
                 thread_var[index].page_thread_res.detections_by_lang.end(),
                 greater<OCR_output>());

            thread_var[index].page_thread_res.detections_by_lang.resize(page_inputs.ocr_fset.max_text_tracks);
        }

        for (const auto &ocr_out: thread_var[index].page_thread_res.detections_by_lang) {
            MPFGenericTrack generic_track(ocr_out.confidence);
            // Copy over OSD results into OCR tracks.
            generic_track.detection_properties = thread_var[index].osd_track_results.detection_properties;

            bool process_text = process_ocr_text(generic_track.detection_properties, c_job, ocr_out,
                                                 page_inputs.ocr_fset, index);
            if (process_text) {
                (*page_results.tracks).push_back(generic_track);
            }
        }
    }
}

void TesseractOCRTextDetection::process_serial_pdf_pages(PDF_page_inputs &page_inputs,
                                                         PDF_page_results &page_results) {
    int page_num = 0;
    for (const string &filename : page_inputs.filelist) {
        MPFImageJob c_job((*page_inputs.job).job_name, filename, (*page_inputs.job).job_properties, (*page_inputs.job).media_properties);
        MPFImageReader image_reader(c_job);
        cv::Mat image_data = image_reader.GetImage();
        preprocess_image(c_job, image_data, page_inputs.ocr_fset);

        MPFGenericTrack osd_track_results(-1);
        string tessdata_script_dir = "";
        if (page_inputs.ocr_fset.psm == 0 || page_inputs.ocr_fset.enable_osd) {
            OSBestResult os_best_result;
            // Reset to original specified language before processing OSD.
            page_inputs.ocr_fset.tesseract_lang = page_inputs.default_lang;
            set<string> missing_languages;
            get_OSD(os_best_result, image_data, c_job, page_inputs.ocr_fset, osd_track_results.detection_properties,
                    tessdata_script_dir, missing_languages);
            page_results.all_missing_languages.insert(missing_languages.begin(), missing_languages.end());

            // When PSM is set to 0, there is no need to process any further.
            // Proceed to next page for OSD processing.
            if (page_inputs.ocr_fset.psm == 0) {
                osd_track_results.detection_properties["PAGE_NUM"] = to_string(page_num + 1);
                page_results.tracks->push_back(osd_track_results);
                page_num++;
                continue;
            }
        }
        else {
            // If OSD is not run, the image won't be rescaled yet.
            rescale_image(c_job, image_data, page_inputs.ocr_fset);
        }


        OCR_job_inputs ocr_job_inputs;
        ocr_job_inputs.job_name = &c_job.job_name;
        ocr_job_inputs.lang = &page_inputs.ocr_fset.tesseract_lang;
        ocr_job_inputs.tessdata_script_dir = &tessdata_script_dir;
        ocr_job_inputs.run_dir = &page_inputs.run_dir;
        ocr_job_inputs.imi = &image_data;
        ocr_job_inputs.ocr_fset = &page_inputs.ocr_fset;
        ocr_job_inputs.process_pdf = true;
        ocr_job_inputs.hw_logger_ = hw_logger_;
        ocr_job_inputs.tess_api_map = &tess_api_map;

        Image_results image_results;
        image_results.job_status = page_results.job_status;

        get_tesseract_detections(ocr_job_inputs, image_results);
        page_results.job_status = image_results.job_status;

        // If max_text_tracks is set, filter out to return only the top specified tracks.
        if (page_inputs.ocr_fset.max_text_tracks > 0) {
            sort(image_results.detections_by_lang.begin(),
                 image_results.detections_by_lang.end(),
                 greater<OCR_output>());
            image_results.detections_by_lang.resize(page_inputs.ocr_fset.max_text_tracks);
        }

        for (const auto &ocr_out: image_results.detections_by_lang) {
            MPFGenericTrack generic_track(ocr_out.confidence);
            // Copy over OSD results into OCR tracks.
            generic_track.detection_properties = osd_track_results.detection_properties;

            bool process_text = process_ocr_text(generic_track.detection_properties, c_job, ocr_out,
                                                 page_inputs.ocr_fset, page_num);
            if (process_text) {
                page_results.tracks->push_back(generic_track);
            }
        }
        page_num++;
    }
}

vector<MPFGenericTrack> TesseractOCRTextDetection::GetDetections(const MPFGenericJob &job) {
    try {
        LOG4CXX_DEBUG(hw_logger_, "[" + job.job_name + "] Processing \"" + job.data_uri + "\".");

        PDF_page_inputs page_inputs;
        PDF_page_results page_results;

        std::vector<MPFGenericTrack> tracks;
        page_results.tracks = &tracks;
        page_inputs.job = &job;

        load_settings(job, page_inputs.ocr_fset);

        page_results.job_status = MPF_DETECTION_SUCCESS;

        vector<string> job_names;
        boost::split(job_names, job.job_name, boost::is_any_of(":"));
        string job_name = boost::ireplace_all_copy(job_names[0], "job", "");
        boost::trim(job_name);

        page_inputs.run_dir = GetRunDirectory();
        if (page_inputs.run_dir.empty()) {
            page_inputs.run_dir = ".";
        }

        check_default_languages(page_inputs.ocr_fset, job.job_name, page_inputs.run_dir, page_results.job_status);

        string plugin_path = page_inputs.run_dir + "/TesseractOCRTextDetection";
        TempDirectory temp_im_directory(plugin_path + "/tmp-" + job_name + "-" + random_string(20));

        // Attempts to process generic document.
        // If read successfully, convert to images and store in a temporary directory.
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
            string tiff_file = temp_im_directory.path + "/results_extracted" + to_string(i) + ".tiff";
            image->write(tiff_file);
            page_inputs.filelist.push_back(tiff_file);
        }

        page_inputs.default_lang = page_inputs.ocr_fset.tesseract_lang;

        if (page_inputs.ocr_fset.max_parallel_pdf_threads > 1 && page_inputs.filelist.size() > 1 &&
            page_inputs.ocr_fset.psm != 0) {
            // Process PDF pages in parallel.
            process_parallel_pdf_pages(page_inputs, page_results);
        }
        else {
            // Process PDF pages in serial order.
            process_serial_pdf_pages(page_inputs, page_results);
        }

        for (auto &track: tracks) {
            track.detection_properties["MISSING_LANGUAGE_MODELS"] = boost::algorithm::join(page_results.all_missing_languages,
                                                                                           ", ");
        }
        LOG4CXX_INFO(hw_logger_,
                     "[" + job.job_name + "] Processing complete. Found " + to_string(tracks.size()) + " tracks.");
        return tracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, hw_logger_);
    }
}


TessApiWrapper::TessApiWrapper(const std::string& data_path, const std::string& language, tesseract::OcrEngineMode oem) {
    int rc = tess_api_.Init(data_path.c_str(), language.c_str(), oem);
    if (rc != 0) {
        throw MPFDetectionException(
                MPF_DETECTION_NOT_INITIALIZED,
                "Failed to initialize Tesseract! Error code: " + std::to_string(rc));
    }
}

void TessApiWrapper::SetPageSegMode(tesseract::PageSegMode mode) {
    tess_api_.SetPageSegMode(mode) ;
}

void TessApiWrapper::SetImage(const cv::Mat &image) {
    tess_api_.SetImage(image.data, image.cols, image.rows, image.channels(),
                       static_cast<int>(image.step));
}

std::string TessApiWrapper::GetUTF8Text() {
    std::unique_ptr<char[]> text{tess_api_.GetUTF8Text()};
    if (text == nullptr) {
        return "";
    }
    return std::string(text.get());
}

int TessApiWrapper::MeanTextConf() {
    return tess_api_.MeanTextConf();
}

void TessApiWrapper::Clear() {
    tess_api_.Clear();
}

bool TessApiWrapper::DetectOS(OSResults *results) {
    return tess_api_.DetectOS(results);
}


MPF_COMPONENT_CREATOR(TesseractOCRTextDetection);
MPF_COMPONENT_DELETER();
