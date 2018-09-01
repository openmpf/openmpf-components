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
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/regex.hpp>
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
}

/*
 * Called during Init.
 * Copy over parameters values from .ini file.
 */
void TesseractOCRTextDetection::SetReadConfigParameters() {

    if(parameters.contains("SHARPEN")) {
        ocr_fset.sharpen = parameters["SHARPEN"].toDouble();
    }
    if(parameters.contains("SCALE")) {
        ocr_fset.scale = parameters["SCALE"].toDouble();
    }
    if(parameters.contains("THRS_FILTER")) {
        ocr_fset.threshold_check = (parameters["THRS_FILTER"].toInt() > 0);
    }
    if(parameters.contains("HIST_FILTER")) {
        ocr_fset.hist_check = (parameters["HIST_FILTER"].toInt() > 0);
    }
    if(parameters.contains("NUM_ONLY")) {
        ocr_fset.num_only_ok = (parameters["NUM_ONLY"].toInt() > 0);
    }
    if(parameters.contains("MIN_WORD_LEN")) {
        ocr_fset.min_word_len = parameters["MIN_WORD_LEN"].toInt();
    }
    if(parameters.contains("MIN_HIST_SIZE")) {
        ocr_fset.hist_min_char = parameters["MIN_HIST_SIZE"].toInt();
    }
    if(parameters.contains("MIN_HIST_SCORE")) {
        ocr_fset.correl_limit = parameters["MIN_HIST_SCORE"].toDouble();
    }
    if(parameters.contains("MAX_ENG_PNCT")) {
        ocr_fset.excess_eng_symbols = parameters["MAX_ENG_PNCT"].toDouble();
    }
    if(parameters.contains("MAX_FRN_CHAR")) {
        ocr_fset.excess_non_eng_symbols = parameters["MAX_FRN_CHAR"].toDouble();
    }
    if(parameters.contains("VOWEL_MIN")) {
        ocr_fset.vowel_min = parameters["VOWEL_MIN"].toDouble();
    }
    if(parameters.contains("VOWEL_MAX")) {
        ocr_fset.vowel_max = parameters["VOWEL_MAX"].toDouble();
    }
}

/*
 * Counts whitespace, alphanumeric, non-english characters in string.
 */
TesseractOCRTextDetection::OCR_char_stats TesseractOCRTextDetection::char_count(const std::string &s,const std::string &white_space, const std::string &eng_symbol, const std::string &eng_num) {
    TesseractOCRTextDetection::OCR_char_stats stats = {
        0,  //alphabet_count
        0,  //num_count
        0,  //whspace_count
        0,  //punct_count
        0,  //non_eng_count
        {0} //char_list
    };

    for (char c : s) {
        if (white_space.find(c) != std::string::npos) {
            stats.whspace_count++;
            continue;
        }
        if (eng_symbol.find(c) != std::string::npos) {
            stats.punct_count++;
            continue;
        }
        if (eng_num.find(c) != std::string::npos) {
            stats.num_count++;
            continue;
        }
        int x = tolower(c) - (int)'a';
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
std::string TesseractOCRTextDetection::check_string(const std::string &s,const TesseractOCRTextDetection::OCR_filter_settings &ocrset) {
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
    std::string white_space = " \n\t\f\v\r";

    //Common english characters and punctuation.
    //May need to penalize if these occur too frequently.
    //If a large portion of the sentences are composed of these characters,
    //it's likely gibberish so toss them out.
    std::string eng_symbol = ".,?!-()[]{}<>:;/@#$%^&*-+_='\\~\"";
    
    //Allow numbers by default.
    //Text could be from an academic source, or a phone number.
    std::string eng_num = "0123456789";



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
    if(alphabet_count > 0){
       for(int i = 0; i < 26; i++){
           float f = (float)char_list[i]/(float)alphabet_count*100.0;
           char_f_list[i] = f;
       }
       char_mat= cv::Mat(26, 1, CV_32F, char_f_list);
       cv::calcHist(&char_mat, 1, channels, cv::Mat(), // do not use mask
                 char_hist, 1, histSize, ranges);
    }


    if(threshold_check){
        int total_eng_char = num_count+alphabet_count+punct_count;
        if((alphabet_count + num_count < min_word_len)) return "";
        
        float eng_symb_fraction = float(punct_count) / float(total_eng_char);
        if(eng_symb_fraction > excess_eng_symbols) return "";

        float non_eng_fraction = float(non_eng_count)/float(total_eng_char+non_eng_count);
        if(non_eng_fraction > eng_symb_fraction) return "";

        int max_wsize = 0;
        istringstream iss(s);

        do
        {
            std::string subs;
            iss >> subs;
            if(subs.length()>max_wsize) max_wsize = subs.length();
        } while (iss);
        if (max_wsize < min_word_len) return "";
    }
    if (alphabet_count == 0) {
        if (num_only_ok) return s;
        else return "";
    }
    //Calculate vowel percentage and check if threshold is met.
    float vowel_percent = (float)(char_list[0]+char_list[4]+char_list[8]+char_list[14]+char_list[20]+char_list[24])/(float)alphabet_count;
    if((vowel_percent< vowel_min || vowel_percent>vowel_max)&&threshold_check) return "";

    if(alphabet_count >= hist_min_char){
        double result = abs(cv::compareHist( eng_hist, char_hist, CV_COMP_CORREL));
    }
    return s;

}

bool TesseractOCRTextDetection::Init() {
    job_name = "TesseractOCR initialization";
    
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


    regTable["\\\\d"] = "[[:digit:]]";
    regTable["\\\\l"] = "[[:lower:]]";
    regTable["\\\\s"] = "[[:space:]]";
    regTable["\\\\u"] = "[[:upper:]]";
    regTable["\\\\w"] = "[[:word:]]";
    regTable["\\\\D"] = "[^[:digit:]]";
    regTable["\\\\L"] = "[^[:lower:]]";
    regTable["\\\\S"] = "[^[:space:]]";
    regTable["\\\\U"] = "[^[:upper:]]";
    regTable["\\\\W"] = "[^[:word:]]";
    regTable["\\b"] = "\\b";
    regTable["\\B"] = "\\B";
    regTable["\\p"] = "\\p";
    regTable["\\P"] = "\\P";
    

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
 * Sharpen image.
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
inline string to_lowercase(const string &data)
{
    string d2(data);
    for(auto& c :d2){tolower(c);}
    return d2;
}

/*
 * Helper function for string processing.
 */
inline string trim_punc(const string &in)
{
    string d2(in);
    boost::trim_if(d2, [](char c) { return std::ispunct(c); });
    return d2;

}

/*
 * Helper function for string processing.
 */
std::string clean_whitespace(const std::string &input)
{

    boost::regex re("\n(\n|[[:space:]])+");
    boost::regex re2("\\\\n(\\\\n|[[:space:]])+");
    string result = boost::regex_replace(input,re,"\n");
    string result2 = boost::regex_replace(result,re2,"\\\\n");
    return result2;

}

/*
 * Split a string into a vector of tokens (for split-search).
 */
std::vector<std::string> TesseractOCRTextDetection::get_tokens(const std::string &str) {
    vector<string> dt;
    stringstream ss;
    string tmp; 
    ss << str;
    for (size_t i; !ss.eof(); ++i) {
        ss >> tmp;
        dt.push_back(trim_punc(tmp));
    }
    return dt;
}

/*
 * Reads JSON Tag filter file.
 * Setup tags for split-string and regex filters.
 */
std::map<std::string,std::map<std::string,std::vector<std::string>>> TesseractOCRTextDetection::parse_json(const std::string &jsonfile_name)
{
    std::map<std::string,std::map<std::string,std::vector<std::string>>> json_kvs;
    std::ifstream ifs(jsonfile_name);
    if(!ifs.is_open())
        LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] ERROR READING JSON FILE AT " + jsonfile_name));
    std::string j;
    std::stringstream buffer2; 
    buffer2 << ifs.rdbuf();
    j = buffer2.str();
    JSONValue *value = JSON::Parse(j.c_str());
    if(value == NULL)
    {
        LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] JSON is corrupted."));
    }
    JSONObject root;
    root = value->AsObject();


    std::map<std::string,std::vector<std::string>> json_kvs_string;
    std::map<std::string,std::vector<std::string>> json_kvs_regex;
    std::map<std::string,std::vector<std::string>> json_kvs_string_split;
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
            if(!root3[term]->IsArray())
            {
                LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Invalid JSON Arrayin STRING tags!"));
            } 
            for (unsigned int i = 0; i < array.size(); i++)
            {
                wstringstream output;
                output << array[i]->Stringify();
                wstring ws(output.str());
                string str(ws.begin(),ws.end());
                str = str.substr(1,str.size()-2);
                json_kvs_string[term_str].push_back(str);
            }
            iter++;
        }
    }
    else{
        LOG4CXX_WARN(hw_logger_, log_print_str( "[" + job_name + "] TAGS_STRING NOT FOUND."));
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
                wstringstream output;
                output << array[i]->Stringify();
                wstring ws(output.str());
                string str(ws.begin(),ws.end());
                str = str.substr(1,str.size()-2);
                json_kvs_string_split[term_str].push_back(str);
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
            if(!root3[term]->IsArray())
            {
                LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Invalid JSON Array in REGEX tags!"));              
            } 
            for (unsigned int i = 0; i < array.size(); i++)
            {
                wstringstream output;
                output << array[i]->Stringify();
                wstring ws(output.str());
                string str(ws.begin(),ws.end());
                str = str.substr(1,str.size()-2);
                str = fix_regex(str);
                json_kvs_regex[term_str].push_back(str);
            }
            iter++;
        }
    }
    else{
        LOG4CXX_WARN(hw_logger_, log_print_str( "[" + job_name + "] TAGS_BY_REGEX NOT FOUND."));
    }


    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] successfully read JSON."));
    json_kvs["TAGS_STRING"] = json_kvs_string;
    json_kvs["TAGS_STRING_SPLIT"] = json_kvs_string_split;
    json_kvs["TAGS_REGEX"] = json_kvs_regex;
    return json_kvs;
}

/*
 * Verify that two strings are identical (ignore letter case).
 */
bool TesseractOCRTextDetection::comp_strcmp(const std::string &strHaystack, const std::string &strNeedle)
{
    auto it = std::search(
    strHaystack.begin(), strHaystack.end(),
    strNeedle.begin(),   strNeedle.end(),
    [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); }
  );
  return (it != strHaystack.end() );
}

/*
 * Check if detection string contains regstr pattern.
 */
bool TesseractOCRTextDetection::comp_regex(const std::string &detection, const std::string  &regstr)
{
    bool found = false;
     try {
        boost::regex reg_matcher(regstr, boost::regex_constants::extended);
        boost::smatch m;
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
    cout << text << endl;
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
bool TesseractOCRTextDetection::get_tesseract_detections(const MPFImageJob &job, std::string &detection, cv::Mat &original, double weight, int psm, std::string lang)
{
    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    original = cv::imread(job.data_uri, CV_LOAD_IMAGE_COLOR);

    if( !original.data ) {
        LOG4CXX_WARN(hw_logger_,  "[" + job_name + "] Could not open original image and will not return detections" );
        return false;
    }else{
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Original image opened." );
    }

    MPFImageReader video_capture(job);
    cv::Mat image_data;
    image_data = video_capture.GetImage();
    if( image_data.empty() ) {
        LOG4CXX_WARN(hw_logger_,  "[" + job_name + "] Could not open transformed image and will not return detections" );
        return false;
    }else{
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Transformed image opened." );
    }
    cv::cvtColor(image_data,image_data,cv::COLOR_BGR2GRAY);

    //Modification to resize image via user input:
    ocr_fset.scale = DetectionComponentUtils::GetProperty<double>(job.job_properties,"SCALE",ocr_fset.scale);
    cv::resize(image_data,image_data,cv::Size(),ocr_fset.scale ,ocr_fset.scale );
    Sharpen(image_data,weight);
    cv::Mat imb,imi;
    double thresh_val = cv::threshold(image_data,imb,0,255,cv::THRESH_BINARY|cv::THRESH_OTSU);
    double min, max;
    cv::Mat tmp_imb(imb.size(),imb.type());
    cv::minMaxLoc(imb, &min, &max);
    tmp_imb.setTo(cv::Scalar::all(max));
    cv::subtract(tmp_imb,imb,imi);
    vector<string> results;
    boost::algorithm::split(results, job_name, boost::algorithm::is_any_of(" :"));
    string plugin_path = run_dir + "/TesseractOCR";
    string imname = random_string(20)+(results.size()>1 ? results[1] : "_")+".png";
    LOG4CXX_DEBUG(hw_logger_,"[" + job_name + "] Creating temporary image "+plugin_path + "/"+imname);
    cv::imwrite(plugin_path + "/"+imname,imi);

    std::array<char, 128> buffer;
    std::string result;
    string bin_path = plugin_path+"/bin";
    string cmdex = bin_path + "/tesseract -l " + lang + " -psm " + std::to_string(psm) + " " + plugin_path + "/"+imname+" stdout";

    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] About to call tesseract with command: " + cmdex);
    string tesspref = "";
    char* ldpath_v = getenv("LD_LIBRARY_PATH");
    string ldpath = ldpath_v==NULL? "" : string(ldpath_v);
    cout << ldpath <<endl;
    ldpath = "LD_LIBRARY_PATH="+plugin_path+"/lib/:"+ldpath;
    tesspref = "TESSDATA_PREFIX="+plugin_path+"/bin/";
    cmdex = tesspref + " "+ldpath + " " + cmdex;
    std::shared_ptr<FILE> pipe(popen(cmdex.c_str(),"r"),pclose);
    if (!pipe){
     throw std::runtime_error("popen() failed!");
     LOG4CXX_ERROR(hw_logger_,  "[" + job_name + "] popen() failed! Tesseract can't be found?");
    }
    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Tesseract ran");
    while (!feof(pipe.get())){
      if( fgets(buffer.data(),128,pipe.get())!=NULL)
        result+=buffer.data();
    }
    detection = result;
    if(remove((plugin_path + "/"+imname).c_str())!=0)
      LOG4CXX_ERROR(hw_logger_, "[" + job_name + "] error deleting temp image");

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

std::string TesseractOCRTextDetection::fix_regex(std::string inreg)
{
    for(auto const& x : regTable)
    {
            replace(inreg, x.first, x.second);
    }
    return inreg;
}

bool is_only_ascii_whitespace( const std::string& str )
{
    auto it = str.begin();
    do {
        if (it == str.end()) return true;
    } while (*it >= 0 && *it <= 0x7f && std::isspace(*(it++)));
             // one of these conditions will be optimized away by the compiler,
             // which one depends on whether char is signed or not
    return false;
}



/*
 * Performs regex-tagging of ocr text detection.
 */
set<string> TesseractOCRTextDetection::search_regex(const std::string &ocr_detections, const std::map<std::string,std::vector<std::string>> &json_kvs_regex)
{
        string found_tags_regex = "";
        set<string> found_keys_regex;
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
            found_tags_regex +=tags + ", ";
            num_found++;
        }
        if (num_found > 0)
        {
            found_tags_regex = found_tags_regex.substr(0,found_tags_regex.size()-2);
        }
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Done searching for regex tags, found: " + to_string(num_found));
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Found regex tags are: "+ found_tags_regex);

        return found_keys_regex;
}

/*
 * Performs split-string-tagging of ocr text detection.
 */
set<string> TesseractOCRTextDetection::search_string_split(const std::vector<std::string> &tokenized, const std::map<std::string,std::vector<std::string>> &json_kvs_string)
{
    string found_tags_string = "";
    set<string> found_keys_string;
    if (json_kvs_string.size() == 0) return found_keys_string;
        boost::regex rgx("\\s+");
        for (const auto& kv : json_kvs_string)
        {
            auto key = kv.first;
            auto values = kv.second;
            bool breaker = false;
            for(auto value : values)
            {
                vector<string> tag_tokens;
                boost::split_regex(tag_tokens,value,rgx);
                if (tag_tokens.size() == 1)
                {
                    for (const auto& token : tokenized)
                    {
                        if (strcasecmp(token.c_str(), value.c_str()) == 0)
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
                        else if (strcasecmp(token.c_str(), tag_tokens[word_id].c_str()) == 0)
                        { 
                            word_id++;
                        }
                        else if (word_id > 0)
                        {
                            if (strcasecmp(token.c_str(), tag_tokens[0].c_str()) == 0)
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
            found_tags_string += tags + ", ";
            num_found++;
        }
        if (num_found > 0)
        {
            found_tags_string = found_tags_string.substr(0,found_tags_string.size() - 2);
        }
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Done searching for string tags, found: " + to_string(num_found));
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Found string tags are: "+ found_tags_string);
        return found_keys_string;
}

/*
 * Performs full-string tagging of ocr text detection.
 */
set<string>  TesseractOCRTextDetection::search_string(const std::string &ocr_detections, const std::map<std::string,std::vector<std::string>> &json_kvs_string)
{
    string found_tags_string = "";
    set<string> found_keys_string;
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
            found_tags_string += tags + ", ";
            num_found++;
        }
        if (num_found > 0)
        {
            found_tags_string = found_tags_string.substr(0,found_tags_string.size() - 2);
        }
        LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Done searching for full string tags, found: " + to_string(num_found));
        LOG4CXX_DEBUG(hw_logger_,  "[" + job_name + "] Found full string tags are: "+ found_tags_string);
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
    std::string lang = DetectionComponentUtils::GetProperty<std::string>(job.job_properties,"TESSERACT_LANGUAGE","eng");

    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] About to read JSON from: " + jsonfile_name));
    auto json_kvs_full = parse_json(jsonfile_name);
    std::map<std::string,std::vector<std::string>> json_kvs_string = json_kvs_full["TAGS_STRING"];
    std::map<std::string,std::vector<std::string>> json_kvs_string_split = json_kvs_full["TAGS_STRING_SPLIT"];
    std::map<std::string,std::vector<std::string>> json_kvs_regex = json_kvs_full["TAGS_REGEX"];
    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Read JSON"));
    string ocr_detections;
    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] About to run tesseract"));
    cv::Mat image;
    ocr_fset.sharpen = DetectionComponentUtils::GetProperty<double>(job.job_properties,"SHARPEN",ocr_fset.sharpen);
    
    if (!get_tesseract_detections(job,ocr_detections, image, ocr_fset.sharpen, psm, lang))
    {
        LOG4CXX_ERROR(hw_logger_, log_print_str( "[" + job_name + "] Could not read image!"));
        return MPF_IMAGE_READ_ERROR;
    }

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
        ocr_detections = TesseractOCRTextDetection::check_string(ocr_detections,ocr_fset);


    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Ran tesseract"));
    LOG4CXX_DEBUG(hw_logger_, log_print_str( "[" + job_name + "] Tesseract output was: " + ocr_detections));


    MPFImageLocation image_location(0, 0, image.cols, image.rows);


    if(is_only_ascii_whitespace(ocr_detections))
    {
        LOG4CXX_WARN(hw_logger_, log_print_str("[" + job_name + "] empty OCR image!"));
    }
    else {
        auto tokenized = get_tokens(ocr_detections);
        vector<string> lower_tokenized;
        auto found_tags_regex = search_regex(ocr_detections,json_kvs_regex);
        auto found_tags_string_split = search_string_split(tokenized,json_kvs_string_split);
        auto found_tags_string = search_string(ocr_detections,json_kvs_string);

        for ( auto tag: found_tags_string_split) {
            found_tags_string.insert(tag);
        }

        for ( auto tag: found_tags_regex) {
            found_tags_string.insert(tag);
        }

        std::string tag_string = "";
        int num_found = 0;

        for (auto tags : found_tags_string)
        {
            tag_string += tags + ", ";
            num_found++;
        }

        if (num_found > 0)
        {
            tag_string = tag_string.substr(0,tag_string.size() - 2);
        }

        image_location.detection_properties["TEXT"] = ocr_detections;
        //if(found_tags_string_split.size() > 0 && found_tags_string.size() > 0)
        //{
        //    found_tags_string += ", ";
        //}
        //found_tags_string += found_tags_string_split;

        image_location.detection_properties["TAGS"] = tag_string;
        locations.push_back(image_location);
    }

    LOG4CXX_DEBUG(hw_logger_, "[" + job_name + "] Processing complete. Generated " + std::to_string(locations.size()) + " dummy image locations.");
    return MPF_DETECTION_SUCCESS;
}

bool TesseractOCRTextDetection::Supports(MPFDetectionDataType data_type) {

    return data_type == MPFDetectionDataType::IMAGE;
}

MPF_COMPONENT_CREATOR(TesseractOCRTextDetection);
MPF_COMPONENT_DELETER();


