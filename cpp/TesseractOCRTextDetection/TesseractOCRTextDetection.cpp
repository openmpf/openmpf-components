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

#include <map>
#include <iostream>
#include <boost/regex.hpp>
#include <boost/locale.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>
#include <algorithm>
#include <cctype>
#include "TesseractOCRTextDetection.h"
#include "detectionComponentUtils.h"
#include <Utils.h>
#include <fstream>
#include "JSON.h"
#include "MPFSimpleConfigLoader.h"
#include <tesseract/baseapi.h>

using namespace MPF;
using namespace COMPONENT;
using namespace std;
using log4cxx::Logger;
using log4cxx::xml::DOMConfigurator;

std::string TesseractOCRTextDetection::GetDetectionType() {
    return "TEXT";
}


/*
 * Called during Init.
 * Initialize default parameters.
 */
 void TesseractOCRTextDetection::SetDefaultParameters() {
    ocr_fset.sharpen = 1.0;
    ocr_fset.scale = 2.4;
    ocr_fset.threshold_check = true;
    ocr_fset.hist_check = true;
    ocr_fset.num_only_ok = true;
    ocr_fset.min_word_len = 3;
    ocr_fset.hist_min_char = 45;
    ocr_fset.excess_eng_symbols = 0.35;
    ocr_fset.excess_non_eng_symbols = 0.10;
    ocr_fset.vowel_min = 0.10;
    ocr_fset.vowel_max = 0.95;
    ocr_fset.correl_limit = 0.52;
    ocr_fset.invert = false;

    ocr_fset.enable_sharpen = false;
    ocr_fset.enable_rescale = true;
    ocr_fset.enable_otsu_thrs = false;
    ocr_fset.enable_adaptive_thrs = false;

    ocr_fset.tesseract_lang = "eng";

    ocr_fset.adaptive_thrs_pixel = 11;
    ocr_fset.adaptive_thrs_c = 0;
}

/*
 * Called during Init.
 * Copy over parameters values from .ini file.
 */
void TesseractOCRTextDetection::SetReadConfigParameters() {

    if (parameters.contains("SHARPEN")) {
        ocr_fset.sharpen = parameters["SHARPEN"].toDouble();
    }
    if (parameters.contains("SCALE")) {
        ocr_fset.scale = parameters["SCALE"].toDouble();
    }
    if (parameters.contains("INVERT")) {
        ocr_fset.invert = (parameters["INVERT"].toInt() > 0);
    }
    if (parameters.contains("THRS_FILTER")) {
        ocr_fset.threshold_check = (parameters["THRS_FILTER"].toInt() > 0);
    }
    if (parameters.contains("HIST_FILTER")) {
        ocr_fset.hist_check = (parameters["HIST_FILTER"].toInt() > 0);
    }
    if (parameters.contains("NUM_ONLY")) {
        ocr_fset.num_only_ok = (parameters["NUM_ONLY"].toInt() > 0);
    }
    if (parameters.contains("MIN_WORD_LEN")) {
        ocr_fset.min_word_len = parameters["MIN_WORD_LEN"].toInt();
    }
    if (parameters.contains("MIN_HIST_SIZE")) {
        ocr_fset.hist_min_char = parameters["MIN_HIST_SIZE"].toInt();
    }
    if (parameters.contains("MIN_HIST_SCORE")) {
        ocr_fset.correl_limit = parameters["MIN_HIST_SCORE"].toDouble();
    }
    if (parameters.contains("MAX_ENG_PNCT")) {
        ocr_fset.excess_eng_symbols = parameters["MAX_ENG_PNCT"].toDouble();
    }
    if (parameters.contains("MAX_FRN_CHAR")) {
        ocr_fset.excess_non_eng_symbols = parameters["MAX_FRN_CHAR"].toDouble();
    }
    if (parameters.contains("VOWEL_MIN")) {
        ocr_fset.vowel_min = parameters["VOWEL_MIN"].toDouble();
    }
    if (parameters.contains("VOWEL_MAX")) {
        ocr_fset.vowel_max = parameters["VOWEL_MAX"].toDouble();
    }
    if (parameters.contains("ENABLE_SHARPEN")) {
        ocr_fset.enable_sharpen = (parameters["ENABLE_SHARPEN"].toInt() > 0);
    }
    if (parameters.contains("ENABLE_RESCALE")) {
        ocr_fset.enable_rescale = (parameters["ENABLE_RESCALE"].toInt() > 0);
    }
    if (parameters.contains("ENABLE_OTSU_THRS")) {
        ocr_fset.enable_otsu_thrs = (parameters["ENABLE_OTSU_THRS"].toInt() > 0);
    }
    if (parameters.contains("ENABLE_ADAPTIVE_THRS")) {
        ocr_fset.enable_adaptive_thrs = (parameters["ENABLE_ADAPTIVE_THRS"].toInt() > 0);
    }
    if (parameters.contains("ADAPTIVE_THRS_CONSTANT")) {
        ocr_fset.adaptive_thrs_c = parameters["ADAPTIVE_THRS_CONSTANT"].toDouble();
    }
    if (parameters.contains("ADAPTIVE_THRS_BLOCKSIZE")) {
        ocr_fset.adaptive_thrs_pixel = parameters["ADAPTIVE_THRS_BLOCKSIZE"].toInt();
    }
    if (parameters.contains("TESSERACT_LANGUAGE")) {
        ocr_fset.tesseract_lang = parameters["TESSERACT_LANGUAGE"].toStdString();
    }
}

/*
 * Counts whitespace, alphanumeric, non-english characters in string.
 */
TesseractOCRTextDetection::OCR_char_stats TesseractOCRTextDetection::char_count(const std::wstring &s,const std::wstring &white_space, const std::wstring &eng_symbol, const std::wstring &eng_num) {
    TesseractOCRTextDetection::OCR_char_stats stats = {
        0,  //alphabet_count
        0,  //num_count
        0,  //whspace_count
        0,  //punct_count
        0,  //non_eng_count
        {0} //char_list
    };

    std::locale locale("en_US");
    for (const wchar_t &c : s) {
        if (std::isspace(c, locale)) {
            stats.whspace_count++;
            continue;
        }

        if (std::ispunct(c, locale)) {
            stats.punct_count++;
            continue;
        }

        if (std::isdigit(c, locale)) {
            stats.num_count++;
            continue;
        }

        int x = std::tolower(c, locale) - L'a';
        if (x >= 0 && x <= 25) {
            stats.alphabet_count++;
            stats.char_list[x]++;
            continue;
        }

        stats.non_eng_count++;
    }


    return stats;
}

/*
 * Conduct filtering of results.
 * Reject/accept text based on char frequency/histogram comparison to english language.
 */
std::wstring TesseractOCRTextDetection::check_string(const std::wstring &s,const TesseractOCRTextDetection::OCR_filter_settings &ocrset) {
    bool num_only_ok = ocrset.num_only_ok;
    bool threshold_check = ocrset.threshold_check;
    bool hist_check = ocrset.hist_check;
    int min_word_len = ocrset.min_word_len;
    float excess_eng_symbols = ocrset.excess_eng_symbols;
    float excess_non_eng_symbols = ocrset.excess_non_eng_symbols;
    float vowel_min = ocrset.vowel_min;
    float vowel_max = ocrset.vowel_max;
    int hist_min_char = ocrset.hist_min_char;
    float correl_limit = ocrset.correl_limit;

    //The following are characters commonly used in the english language.
    //We should not penalize the OCR for detecting these.
    //Only start penalizing when they become excessive.

    //Allow white space to be ignored.
    std::wstring white_space = L" \n\t\f\v\r";

    //Common english characters and punctuation.
    //May need to penalize if these occur too frequently.
    //If a large portion of the sentences are composed of these characters,
    //it's likely gibberish so toss them out.
    std::wstring eng_symbol = L".,?!-()[]{}<>:;/@#$%^&*-+_='\\~\"";

    //Allow numbers by default.
    //Text could be from an academic source, or a phone number.
    std::wstring eng_num = L"0123456789";

    //Histogram of english characters
    float eng_list[] = {8.167,1.492,2.782,4.253,12.702,2.228,2.015,6.094,
            6.966,0.153,0.772,4.025,2.406,6.749,7.507,1.929,0.095,
            5.987,6.327,9.056,2.758,0.978,2.360,0.150,1.974,0.074};

    //Histogram parameters:
    int nbins = 200;
    int histSize[] = {nbins};
    float range[]  = {0,100};
    const float *ranges[] = {range};
    int channels[] = {0};

    cv::Mat   eng_mat = cv::Mat(26, 1, CV_32F, eng_list);
    cv::MatND eng_hist;
    cv::calcHist(&eng_mat, 1, channels, cv::Mat(), // do not use mask
             eng_hist, 1, histSize, ranges);


    TesseractOCRTextDetection::OCR_char_stats results = char_count(s, white_space, eng_symbol, eng_num);
    int alphabet_count = results.alphabet_count;
    int num_count = results.num_count;
    int whspace_count = results.whspace_count;
    int punct_count = results.punct_count;
    int non_eng_count = results.non_eng_count;
    int* char_list = results.char_list;
    float char_f_list[26] = {0};
    cv::Mat   char_mat;
    cv::MatND char_hist;
    if (alphabet_count > 0) {
       for (int i = 0; i < 26; i++) {
           float f = (float)char_list[i] / (float)alphabet_count * 100.0;
           char_f_list[i] = f;
       }
       char_mat= cv::Mat(26, 1, CV_32F, char_f_list);
       cv::calcHist(&char_mat, 1, channels, cv::Mat(), // do not use mask
                 char_hist, 1, histSize, ranges);
    }


    if (threshold_check) {
        int total_eng_char = num_count+alphabet_count+punct_count;
        if ((alphabet_count + num_count < min_word_len)) {
            return L"";
        }

        float eng_symb_fraction = float(punct_count) / float(total_eng_char);
        if (eng_symb_fraction > excess_eng_symbols) {
            return L"";
        }

        float non_eng_fraction = float(non_eng_count)/float(total_eng_char+non_eng_count);
        if (non_eng_fraction > eng_symb_fraction) {
             return L"";
        }

        int max_wsize = 0;
        wstringstream iss(s);

        do
        {
            std::wstring subs;
            iss >> subs;
            if(subs.length()>max_wsize) max_wsize = subs.length();
        } while (iss);

        if (max_wsize < min_word_len) {
            return L"";
        }

        //Calculate vowel percentage and check if threshold is met.
        float vowel_percent = (float) (char_list[0] + char_list[4] + char_list[8]
            + char_list[14] + char_list[20] + char_list[24]) / (float) alphabet_count;
        if (vowel_percent < vowel_min || vowel_percent > vowel_max) {
            return L"";
        }

    }
    if (alphabet_count == 0) {
        if (num_only_ok) {
            return s;
        } else {
            return L"";
        }
    }
    if (alphabet_count >= hist_min_char) {
        double result = abs(cv::compareHist( eng_hist, char_hist, CV_COMP_CORREL));
        if (result < ocr_fset.correl_limit) {
            return L"";
        }
    }
    return s;

}

bool TesseractOCRTextDetection::Init() {
    job_name = "TesseractOCR initialization";

    //Set global locale
    boost::locale::generator gen;
    locale loc = gen("");
    locale::global(loc);

    // Determine where the executable is running
    std::string run_dir = GetRunDirectory();
    if (run_dir == "") {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/TesseractOCR";
    std::string config_path = plugin_path + "/config";
    cout << "looking for logger at " << plugin_path+"/config/Log4cxxConfig.xml" <<endl;
    log4cxx::xml::DOMConfigurator::configure(plugin_path+"/config/Log4cxxConfig.xml");
    hw_logger_ = log4cxx::Logger::getLogger("TesseractOCR");

    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] " + "Running in directory " + plugin_path));

    regTable[L"\\\\d"] = L"[[:digit:]]";
    regTable[L"\\\\l"] = L"[[:lower:]]";
    regTable[L"\\\\s"] = L"[[:space:]]";
    regTable[L"\\\\u"] = L"[[:upper:]]";
    regTable[L"\\\\w"] = L"[[:word:]]";
    regTable[L"\\\\D"] = L"[^[:digit:]]";
    regTable[L"\\\\L"] = L"[^[:lower:]]";
    regTable[L"\\\\S"] = L"[^[:space:]]";
    regTable[L"\\\\U"] = L"[^[:upper:]]";
    regTable[L"\\\\W"] = L"[^[:word:]]";
    regTable[L"\\b"] = L"\\b";
    regTable[L"\\B"] = L"\\B";
    regTable[L"\\p"] = L"\\p";
    regTable[L"\\P"] = L"\\P";


    SetDefaultParameters();
    //Once this is done - parameters will be set and SetReadConfigParameters() can be called again to revert back
    //to the params read at initialization.
    std::string config_params_path = config_path + "/mpfOCR.ini";
    int rc = LoadConfig(config_params_path, parameters);
    if (rc) {
        LOG4CXX_ERROR(hw_logger_,  "[" + job_name + "] Could not parse config file: " << config_params_path);
        return (false);
    }
    SetReadConfigParameters();

    LOG4CXX_INFO(hw_logger_,  "[" + job_name + "] INITIALIZED COMPONENT." );
    return true;
}

/*
 * Image preprocessing to improve text extraction.
 * Sharpens image.
 */
void TesseractOCRTextDetection::Sharpen(cv::Mat &image, double weight = 1.0) {
    cv::Mat blurred, mask;
    cv::blur(image,blurred,cv::Size(2,2));
    cv::threshold(blurred,mask,48,1,cv::THRESH_BINARY);
    cv::multiply(blurred,mask,blurred);
    cv::addWeighted(image,1.0+weight,blurred,-1.0,0,image);
}

bool TesseractOCRTextDetection::Close() {

    return true;
}

/*
 * Helper function for string processing.
 */
inline wstring to_lowercase(const wstring &data)
{
    wstring d2(data);
    d2 = boost::locale::normalize(d2);
    d2 = boost::locale::to_lower(d2);
    return d2;
}

/*
 * Helper function for string processing.
 */
inline wstring trim_punc(const wstring &in)
{
    wstring d2(in);
    boost::trim_if(d2, [](wchar_t c) { return std::iswpunct(c);});
    return d2;
}

/*
 * Helper function for string processing.
 */
std::wstring clean_whitespace(const std::wstring &input)
{

    boost::wregex re(L"\n(\n|[[:space:]])+");
    boost::wregex re2(L"\\\\n(\\\\n|[[:space:]])+");
    wstring result = boost::regex_replace(input,re,L"\n");

    wstring result2 = boost::regex_replace(result,re2,L"\\\\n");
    result2 = boost::trim_copy(result2);
    return result2;
}

/*
 * Split a string into a vector of tokens (for split-search).
 */
std::vector<std::wstring> TesseractOCRTextDetection::get_tokens(const std::wstring &str) {
    vector<wstring> dt;
    wstringstream ss;
    wstring tmp;
    ss << str;
    for (size_t i; !ss.eof(); ++i) {
        ss >> tmp;
        dt.push_back(to_lowercase(trim_punc(tmp)));
    }
    return dt;
}

/*
 * Reads JSON Tag filter file.
 * Setup tags for split-string and regex filters.
 */
std::map<std::wstring,std::map<std::wstring,std::vector<std::wstring>>> TesseractOCRTextDetection::parse_json(const std::string &jsonfile_name)
{
    std::map<std::wstring,std::map<std::wstring,std::vector<std::wstring>>> json_kvs;
    std::ifstream ifs(jsonfile_name);
    if (!ifs.is_open())
        LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] ERROR READING JSON FILE AT " + jsonfile_name));
    std::string j;
    std::stringstream buffer2;
    std::wstring x;
    buffer2 << ifs.rdbuf();
    j = buffer2.str();
    x = boost::locale::conv::utf_to_utf<wchar_t>(j);
    JSONValue *value = JSON::Parse(x.c_str());
    if (value == NULL)
    {
        LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] JSON is corrupted."));
    }
    JSONObject root;
    root = value->AsObject();


    std::map<std::wstring,std::vector<std::wstring>> json_kvs_string;
    std::map<std::wstring,std::vector<std::wstring>> json_kvs_regex;
    std::map<std::wstring,std::vector<std::wstring>> json_kvs_string_split;
    if (root.find(L"TAGS_STRING") != root.end() && root[L"TAGS_STRING"]->IsObject())
    {
        LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] String tags found."));
        JSONValue *root2 = JSON::Parse(root[L"TAGS_STRING"]->Stringify().c_str());
        std::vector<std::wstring> keys = root2->ObjectKeys();
        std::vector<std::wstring>::iterator iter = keys.begin();
        JSONObject root3 = root2->AsObject();
        while (iter != keys.end())
        {
            auto term = *iter;
            wstring term_temp(term);
            string term_str(term_temp.begin(),term_temp.end());
            JSONArray array = root3[term]->AsArray();
            if (!root3[term]->IsArray())
            {
                LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Invalid JSON Arrayin STRING tags!"));
            }
            for (unsigned int i = 0; i < array.size(); i++)
            {
                wstring temp = array[i]->Stringify();
                temp = temp.substr(1, temp.size() - 2);
                temp = to_lowercase(temp);
                json_kvs_string[term].push_back(temp);
            }
            iter++;
        }
    }


    if (root.find(L"TAGS_BY_KEYWORD") != root.end() && root[L"TAGS_BY_KEYWORD"]->IsObject())
    {
        LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Split tags found."));
        JSONValue *root2 = JSON::Parse(root[L"TAGS_BY_KEYWORD"]->Stringify().c_str());
        std::vector<std::wstring> keys = root2->ObjectKeys();
        std::vector<std::wstring>::iterator iter = keys.begin();
        JSONObject root3 = root2->AsObject();
        while (iter != keys.end())
        {
            auto term = *iter;
            wstring term_temp(term);
            string term_str(term_temp.begin(),term_temp.end());
            JSONArray array = root3[term]->AsArray();
            if (!root3[term]->IsArray())
            {
                LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Invalid JSON Array in SPLIT tags!"));
            }
            for (unsigned int i = 0; i < array.size(); i++)
            {
                wstring temp = array[i]->Stringify();
                temp = temp.substr(1, temp.size() - 2);
                temp = to_lowercase(temp);
                json_kvs_string_split[term].push_back(temp);
            }
            iter++;
        }
    }
    else{
        LOG4CXX_WARN(hw_logger_, log_print_str( "[" + job_name + "] TAGS_BY_KEYWORD NOT FOUND."));
    }

    // REGEX STUFF
    if (root.find(L"TAGS_BY_REGEX") != root.end() && root[L"TAGS_BY_REGEX"]->IsObject())
    {
        LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Regex tags found."));
        JSONValue *root2 = JSON::Parse(root[L"TAGS_BY_REGEX"]->Stringify().c_str());
        std::vector<std::wstring> keys = root2->ObjectKeys();
        std::vector<std::wstring>::iterator iter = keys.begin();
        JSONObject root3 = root2->AsObject();
        while (iter != keys.end())
        {
            auto term = *iter;
            wstring term_temp(term);
            string term_str(term_temp.begin(),term_temp.end());
            JSONArray array = root3[term]->AsArray();
            if (!root3[term]->IsArray())
            {
                LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Invalid JSON Array in REGEX tags!"));
            }
            for (unsigned int i = 0; i < array.size(); i++)
            {
                wstring temp = array[i]->Stringify();
                temp = temp.substr(1, temp.size() - 2);
                temp = fix_regex(temp);
                temp = to_lowercase(temp);
                json_kvs_regex[term].push_back(temp);
            }
            iter++;
        }
    }
    else{
        LOG4CXX_WARN(hw_logger_, log_print_str( "[" + job_name + "] TAGS_BY_REGEX NOT FOUND."));
    }


    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] successfully read JSON."));
    json_kvs[L"TAGS_STRING"] = json_kvs_string;
    json_kvs[L"TAGS_STRING_SPLIT"] = json_kvs_string_split;
    json_kvs[L"TAGS_REGEX"] = json_kvs_regex;
    return json_kvs;
}

/*
 * Verify that two strings are identical (ignore letter case).
 */
bool TesseractOCRTextDetection::comp_strcmp(const std::wstring &strHaystack, const std::wstring &strNeedle)
{
    auto it = std::search(
    strHaystack.begin(), strHaystack.end(),
    strNeedle.begin(), strNeedle.end(),
    [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
    );
  return (it != strHaystack.end());
}

/*
 * Check if detection string contains regstr pattern.
 */
bool TesseractOCRTextDetection::comp_regex(const std::wstring &detection, const std::wstring  &regstr)
{
    bool found = false;
     try {
        boost::wregex reg_matcher(regstr, boost::regex_constants::extended);
        boost::wsmatch m;
        if (boost::regex_search(detection,m,reg_matcher)) {
            found = true;
        }
     } catch (const boost::regex_error& e) {
        std::stringstream ss;
        ss << "[" + job_name + "] regex_error caught: " << parseRegexCode(e.code()) << ": " << e.what() << '\n';
        LOG4CXX_ERROR(hw_logger_, log_print_str(ss.str()));
     }
    return found;
}
/*
 * Regex error handling.
 */
std::string TesseractOCRTextDetection::parseRegexCode(boost::regex_constants::error_type etype) {
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


string TesseractOCRTextDetection::log_print_str(const string &text)
{
    return text;
}

string TesseractOCRTextDetection::log_print_str(const char* text)
{
    string t2(text);
    return log_print_str(t2);
}

string TesseractOCRTextDetection::log_print_str(const wchar_t* text)
{
    wstring temp(text);
    return log_print_str(string(temp.begin(),temp.end()));
}

string TesseractOCRTextDetection::log_print_str(const std::wstring &text)
{
    return log_print_str(string(text.begin(),text.end()));
}

/*
 * Generate a random alphanumeric string of specified length.
 * Used to generate image placeholders.
 */
std::string random_string( size_t length )
{
    static auto& chrs = "0123456789"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    thread_local static std::mt19937 rg{std::random_device{}()};
    thread_local static std::uniform_int_distribution<std::string::size_type> pick(0, sizeof(chrs) - 2);

    std::string s;

    s.reserve(length);

    while(length--)
        s += chrs[pick(rg)];
    std::ostringstream stp;
    stp << ::getpid();
    return s+ stp.str();
}

/*
 * Run tesseract ocr on refined image.
 */
bool TesseractOCRTextDetection::get_tesseract_detections(const MPFImageJob &job, std::vector<std::pair<std::string, std::wstring>> &detection, cv::Mat &original, double weight, int psm, std::string lang_input)
{
    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }

    tesseract::TessBaseAPI tess_api;


    MPFImageReader image_reader(job);
    cv::Mat image_data;
    image_data = image_reader.GetImage();
    if( image_data.empty() ) {
        LOG4CXX_WARN(hw_logger_,  "[" + job_name + "] Could not open transformed image and will not return detections" );
        return false;
    }else{
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Transformed image opened." );
    }


    ocr_fset.scale = DetectionComponentUtils::GetProperty<double>(job.job_properties,"SCALE",ocr_fset.scale);
    ocr_fset.adaptive_thrs_c = DetectionComponentUtils::GetProperty<double>(job.job_properties,"ADAPTIVE_THRS_CONSTANT", ocr_fset.adaptive_thrs_c);
    ocr_fset.adaptive_thrs_pixel = DetectionComponentUtils::GetProperty<int>(job.job_properties,"ADAPTIVE_THRS_BLOCKSIZE", ocr_fset.adaptive_thrs_pixel);
    ocr_fset.enable_adaptive_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_ADAPTIVE_THRS", ocr_fset.enable_adaptive_thrs);
    ocr_fset.enable_otsu_thrs = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_OTSU_THRS", ocr_fset.enable_otsu_thrs);
    ocr_fset.enable_sharpen = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_SHARPEN", ocr_fset.enable_sharpen);
    ocr_fset.enable_rescale = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"ENABLE_RESCALE", ocr_fset.enable_rescale);

    // Image preprocessig to improve text extraction results.

    // Image thresholding.
    if (ocr_fset.enable_otsu_thrs || ocr_fset.enable_adaptive_thrs)
        cv::cvtColor(image_data,image_data,cv::COLOR_BGR2GRAY);
    if (ocr_fset.enable_adaptive_thrs) {
        // Pixel blocksize ranges 5-51 worked for adaptive threshold.
        cv::adaptiveThreshold(image_data, image_data, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY,ocr_fset.adaptive_thrs_pixel, ocr_fset.adaptive_thrs_c);
    } else if (ocr_fset.enable_otsu_thrs) {
        double thresh_val = cv::threshold(image_data, image_data, 0, 255, cv::THRESH_BINARY|cv::THRESH_OTSU);
    }

    // Rescale and sharpen image (larger images improve detection results).

    if (ocr_fset.enable_rescale){
        cv::resize(image_data,image_data,cv::Size(),ocr_fset.scale ,ocr_fset.scale);
    }
    if (ocr_fset.enable_sharpen){
        Sharpen(image_data,weight);
    }
    cv::Mat imb,imi;
    imb = image_data;

    // Image inversion.
    if (ocr_fset.invert) {
        double min, max;
        cv::Mat tmp_imb(imb.size(),imb.type());
        cv::minMaxLoc(imb, &min, &max);
        tmp_imb.setTo(cv::Scalar::all(max));
        cv::subtract(tmp_imb,imb,imi);
    } else {
        imi = imb;
    }

    vector<string> results;
    boost::algorithm::split(results, job_name, boost::algorithm::is_any_of(" :"));
    string plugin_path = run_dir + "/TesseractOCR";

    vector<std::string> lang_tracks;
    boost::algorithm::split(lang_tracks, lang_input, boost::algorithm::is_any_of(","));
    std::string tessdata_path =  plugin_path+"/bin/tessdata";
    for(std::string lang: lang_tracks) {
        // Process each language specified by user.
        lang = boost::trim_copy(lang);
        std::array<char, 128> buffer;
        std::string result;
        std::string bin_path = plugin_path+"/bin";

        bool missing_lang_model = false;
        vector<std::string> languages;
        boost::algorithm::split(languages, lang, boost::algorithm::is_any_of("+"));
        for ( std::string & c_lang : languages) {
            c_lang = boost::trim_copy(c_lang);
        }

        lang = boost::algorithm::join(languages,"+");

        // Confirm each language model is supported.
        for (std::string c_lang : languages) {

            std::string language_model = bin_path + "/tessdata/" + c_lang + ".traineddata";
            if (!boost::filesystem::exists( language_model )){
                missing_lang_model = true;
                LOG4CXX_WARN(hw_logger_,  "[" + job_name + "] Tesseract language model (" + c_lang
                + ".traineddata) not found. This language is not supported. To support this language, please add "
                + c_lang + ".traineddata to your tessdata directory (TesseractOCRTextDetection/.../bin/tessdata)." );
            }
        }


        int init_rc = tess_api.Init(tessdata_path.c_str(), lang.c_str());
        if (init_rc != 0) {
            LOG4CXX_ERROR(hw_logger_,  "[" + job_name + "] Failed to initialize Tesseract! Error code: " + std::to_string(init_rc));
        }
        tess_api.SetPageSegMode((tesseract::PageSegMode)psm);
        LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] About to call tesseract");
        tess_api.SetImage(imi.data, imi.cols, imi.rows, imi.channels(), static_cast<int>(imi.step));
        std::unique_ptr<char[]> text { tess_api.GetUTF8Text() };

        // Reset tesseract for the next language process.
        tess_api.End();
        result = text.get();

        std::pair<std::string, std::string> output_ocr (lang, result);
        detection.push_back(output_ocr);
    }

    return true;
}

template <typename T, typename U>
T &replace (
          T &str,
    const U &from,
    const U &to)
{
    size_t pos;
    size_t offset = 0;
    const size_t increment = to.size();

    while ((pos = str.find(from, offset)) != T::npos)
    {
        str.replace(pos, from.size(), to);
        offset = pos + increment;
    }

    return str;
}

std::wstring TesseractOCRTextDetection::fix_regex(std::wstring inreg)
{
    for(auto const& x : regTable)
    {
            replace(inreg, x.first, x.second);
    }
    return inreg;
}

bool is_only_ascii_whitespace( const std::wstring& str )
{
    auto it = str.begin();
    do {
        if (it == str.end()) return true;
    } while (*it >= 0 && *it <= 0x7f && std::iswspace(*(it++)));
             // one of these conditions will be optimized away by the compiler,
             // which one depends on whether char is signed or not
    return false;
}



/*
 * Performs regex-tagging of ocr text detection.
 */
set<wstring> TesseractOCRTextDetection::search_regex(const std::wstring &ocr_detections, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_regex)
{
        wstring found_tags_regex = L"";
        set<wstring> found_keys_regex;
        if (json_kvs_regex.size() == 0) return found_keys_regex;

        for (const auto& kv : json_kvs_regex)
        {
            auto key = kv.first;
            auto values = kv.second;
            bool breaker = false;
            for (auto value : values)
            {
                    if (comp_regex(ocr_detections,value))
                    {
                        found_keys_regex.insert(key);
                        breaker = true;
                        break;
                    }
                if (breaker) break;
            }
        }
        int num_found = 0;
        for (auto tags : found_keys_regex)
        {
            found_tags_regex +=tags + L", ";
            num_found++;
        }
        if (num_found > 0)
        {
            found_tags_regex = found_tags_regex.substr(0,found_tags_regex.size()-2);
        }
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Done searching for regex tags, found: " + to_string(num_found));
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Found regex tags are: "+ boost::locale::conv::utf_to_utf<char>(found_tags_regex));

        return found_keys_regex;
}

/*
 * Performs split-string-tagging of ocr text detection.
 */
set<wstring> TesseractOCRTextDetection::search_string_split(const std::vector<std::wstring> &tokenized, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string)
{
    wstring found_tags_string = L"";
    set<wstring> found_keys_string;
    std::locale loc = std::locale();
    if (json_kvs_string.size() == 0) return found_keys_string;
        boost::wregex rgx(L"\\s+");
        for (const auto& kv : json_kvs_string)
        {
            auto key = kv.first;
            auto values = kv.second;
            bool breaker = false;
            for(auto value : values)
            {
                vector<wstring> tag_tokens;
                boost::split_regex(tag_tokens,value,rgx);
                if (tag_tokens.size() == 1)
                {
                    for (const auto& token : tokenized)
                    {
                        if (std::use_facet<boost::locale::collator<wchar_t>>(loc).compare(boost::locale::collator_base::primary,token,value) == 0)
                        {
                            found_keys_string.insert(key);
                            breaker = true;
                            break;
                        }
                    }
                    if ( breaker) break;
                }
                else
                {
                    int word_id = 0;
                    for (const auto& token : tokenized)
                    {
                        if (word_id == tag_tokens.size())
                        {
                                found_keys_string.insert(key);
                                breaker = true;
                                break;
                        }
                        else if (std::use_facet<boost::locale::collator<wchar_t>>(loc).compare(boost::locale::collator_base::primary,token,tag_tokens[word_id]) == 0)
                        {
                            word_id++;
                        }
                        else if (word_id > 0)
                        {
                            if (std::use_facet<boost::locale::collator<wchar_t>>(loc).compare(boost::locale::collator_base::primary,token,tag_tokens[0]) == 0)
                            {
                                word_id = 1;
                            }
                            else
                            {
                                word_id = 0;
                            }

                        }
                    }
                    if (breaker) break;
                }
            }
        }
    int num_found = 0;
        for (auto tags : found_keys_string)
        {
            found_tags_string += tags + L", ";
            num_found++;
        }
        if (num_found > 0)
        {
            found_tags_string = found_tags_string.substr(0,found_tags_string.size() - 2);
        }
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Done searching for string tags, found: " + to_string(num_found));
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Found string tags are: "+ boost::locale::conv::utf_to_utf<char>(found_tags_string));
        return found_keys_string;
}

/*
 * Performs full-string tagging of ocr text detection.
 */
set<wstring> TesseractOCRTextDetection::search_string(const std::wstring &ocr_detections, const std::map<std::wstring,std::vector<std::wstring>> &json_kvs_string)
{
    wstring found_tags_string = L"";
    set<wstring> found_keys_string;
    if (json_kvs_string.size() == 0) return found_keys_string;

        for (const auto& kv : json_kvs_string)
        {
            auto key = kv.first;
            auto values = kv.second;
            bool breaker = false;

            for (auto value : values)
            {
                    if (comp_strcmp(ocr_detections,value))
                    {
                        found_keys_string.insert(key);
                        breaker = true;
                        break;
                    }
            }
        }
    int num_found = 0;
        for (auto tags : found_keys_string)
        {
            found_tags_string += tags + L", ";
            num_found++;
        }
        if (num_found > 0)
        {
            found_tags_string = found_tags_string.substr(0,found_tags_string.size() - 2);
        }
        LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Done searching for full string tags, found: " + to_string(num_found));
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Found full string tags are: "+ boost::locale::conv::utf_to_utf<char>(found_tags_string));
        return found_keys_string;
}

MPFDetectionError TesseractOCRTextDetection::GetDetections(const MPFImageJob &job, std::vector <MPFImageLocation> &locations)
{
    job_name = job.job_name;
    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Processing \"" + job.data_uri + "\".");
    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/TesseractOCR";
    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Running from directory " + plugin_path));



    string jsonfile_name =  DetectionComponentUtils::GetProperty<std::string>(job.job_properties,"TAGGING_FILE","text-tags.json");
    if ( jsonfile_name.find('$') != std::string::npos || jsonfile_name.find('/') != std::string::npos ) {
        string new_jsonfile_name = "";
        Utils::expandFileName(jsonfile_name,new_jsonfile_name);
        jsonfile_name = new_jsonfile_name;
    } else {
        jsonfile_name = plugin_path + "/config/" + jsonfile_name;
    }

    int psm = DetectionComponentUtils::GetProperty<int>(job.job_properties,"TESSERACT_PSM",3);
    std::string lang = DetectionComponentUtils::GetProperty<std::string>(job.job_properties,"TESSERACT_LANGUAGE", ocr_fset.tesseract_lang);

    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] About to read JSON from: " + jsonfile_name));
    auto json_kvs_full = parse_json(jsonfile_name);
    std::map<std::wstring,std::vector<std::wstring>> json_kvs_string = json_kvs_full[L"TAGS_STRING"];
    std::map<std::wstring,std::vector<std::wstring>> json_kvs_string_split = json_kvs_full[L"TAGS_STRING_SPLIT"];
    std::map<std::wstring,std::vector<std::wstring>> json_kvs_regex = json_kvs_full[L"TAGS_REGEX"];
    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Read JSON"));

    std::vector<std::pair<std::string, std::wstring>> ocr_outputs;
    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] About to run tesseract"));
    cv::Mat image;
    ocr_fset.sharpen = DetectionComponentUtils::GetProperty<double>(job.job_properties,"SHARPEN",ocr_fset.sharpen);
    ocr_fset.invert = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"INVERT",ocr_fset.invert);

    if (!get_tesseract_detections(job,ocr_outputs, image, ocr_fset.sharpen, psm, lang))
    {
        LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Could not read image!"));
        return MPF_IMAGE_READ_ERROR;
    }
    for (auto ocr_out: ocr_outputs) {
        std::string ocr_lang = ocr_out.first;
        std::wstring ocr_detections = ocr_out.second;
        ocr_detections = clean_whitespace(ocr_detections);
            //String Filtering.
            ocr_fset.threshold_check = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"THRS_FILTER",ocr_fset.threshold_check);
            ocr_fset.hist_check = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"HIST_FILTER",ocr_fset.hist_check);
            ocr_fset.num_only_ok = DetectionComponentUtils::GetProperty<bool>(job.job_properties,"NUM_ONLY",ocr_fset.num_only_ok);
            ocr_fset.min_word_len = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MIN_WORD_LEN",ocr_fset.min_word_len);
            ocr_fset.hist_min_char = DetectionComponentUtils::GetProperty<int>(job.job_properties,"MIN_HIST_SIZE",ocr_fset.hist_min_char);
            ocr_fset.excess_eng_symbols = DetectionComponentUtils::GetProperty<float>(job.job_properties,"MAX_ENG_PNCT",ocr_fset.excess_eng_symbols);
            ocr_fset.excess_non_eng_symbols = DetectionComponentUtils::GetProperty<float>(job.job_properties,"MAX_FRN_CHAR",ocr_fset.excess_non_eng_symbols);
            ocr_fset.vowel_min = DetectionComponentUtils::GetProperty<float>(job.job_properties,"VOWEL_MIN",ocr_fset.vowel_min);
            ocr_fset.vowel_max = DetectionComponentUtils::GetProperty<float>(job.job_properties,"VOWEL_MAX",ocr_fset.vowel_max);
            ocr_fset.correl_limit = DetectionComponentUtils::GetProperty<float>(job.job_properties,"MIN_HIST_SCORE",ocr_fset.correl_limit);

        if (ocr_fset.hist_check || ocr_fset.threshold_check) {
            ocr_detections = TesseractOCRTextDetection::check_string(ocr_detections,ocr_fset);
        }

        LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Ran tesseract"));
        LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Tesseract output was: " + boost::locale::conv::utf_to_utf<char>(ocr_detections)));


        MPFImageLocation image_location(0, 0, image.cols, image.rows);


        if(is_only_ascii_whitespace(ocr_detections))
        {
            LOG4CXX_WARN(hw_logger_, log_print_str("[" + job_name + "] No text found in image!"));
        }
        else {
            auto tokenized = get_tokens(ocr_detections);
            std::wstring norm_detections = to_lowercase(ocr_detections);
            auto found_tags_regex = search_regex(norm_detections,json_kvs_regex);
            auto found_tags_string_split = search_string_split(tokenized,json_kvs_string_split);
            auto found_tags_string = search_string(norm_detections,json_kvs_string);

            for ( auto tag: found_tags_string_split) {
                found_tags_string.insert(tag);
            }
            for ( auto tag: found_tags_regex) {
                found_tags_string.insert(tag);
            }
            std::wstring tag_string = L"";
            int num_found = 0;
            for (auto tags : found_tags_string)
            {
                tag_string += tags + L", ";
                num_found++;
            }
             if (num_found > 0)
            {
                tag_string = tag_string.substr(0,tag_string.size() - 2);
            }

            image_location.detection_properties["LANGUAGE"] = ocr_lang;
            image_location.detection_properties["TAGS"] = boost::locale::conv::utf_to_utf<char>(tag_string);
            image_location.detection_properties["TEXT"] = boost::locale::conv::utf_to_utf<char>(ocr_detections);
            locations.push_back(image_location);
        }
    }
    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Processing complete. Found " + std::to_string(locations.size()) + " tracks.");
    ocr_outputs.clear();
    return MPF_DETECTION_SUCCESS;
}

bool TesseractOCRTextDetection::Supports(MPFDetectionDataType data_type) {

    return data_type == MPFDetectionDataType::IMAGE;
}

MPF_COMPONENT_CREATOR(TesseractOCRTextDetection);
MPF_COMPONENT_DELETER();
