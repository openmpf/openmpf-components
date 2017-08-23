/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2017 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2017 The MITRE Corporation                                       *
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

#include <iostream>
#include <fstream>
#include <map>
#include <QString>
#include <QDir>
#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <detectionComponentUtils.h>
#include <MPFImageReader.h>
#include "LicensePlateTextDetection.h"
#include <Utils.h>
#include <MPFSimpleConfigLoader.h>

using namespace MPF;
using namespace COMPONENT;

using namespace std;
using namespace alpr;

using log4cxx::Logger;
using log4cxx::xml::DOMConfigurator;

//-----------------------------------------------------------------------------
LicensePlateTextDetection::LicensePlateTextDetection() {

}

//-----------------------------------------------------------------------------
LicensePlateTextDetection::~LicensePlateTextDetection() {

}

//-----------------------------------------------------------------------------
/* virtual */ std::string LicensePlateTextDetection::GetDetectionType() {
    return "TEXT";
}

//-----------------------------------------------------------------------------
/* virtual */ bool LicensePlateTextDetection::Init() {

    // Determine where the executable is running
    string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    string plugin_path = run_dir + "/OalprLicensePlateTextDetection";
    string config_path = plugin_path + "/config";

    // Configure logger

    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    td_logger_ = log4cxx::Logger::getLogger("OalprLicensePlateTextDetection");

    LOG4CXX_DEBUG(td_logger_, "Plugin path: " << plugin_path);

    // Load the config file
    int rc;
    rc = LoadConfig(config_path + "/mpfOALPRLicensePlateTextDetection.ini", parameters);
    if (rc) {
        LOG4CXX_ERROR(td_logger_, "loadConfig failed: config_path = "
                << config_path);
        return (false);
    }

    // Instantiate and initialize ALPR

    // Check whether the TESSDATA_PREFIX variable is set in the environment
    // If not, then read the value stored in the config file, and set it in the
    // environment.
    string env_name("TESSDATA_PREFIX");
    char *tmp = getenv(env_name.c_str());
    if (NULL != tmp) {
        LOG4CXX_DEBUG(td_logger_, "TESSDATA_PREFIX current value: " << tmp);
    }
    else {
        string env_value = parameters["TESSDATA_PREFIX"].toStdString();
        LOG4CXX_DEBUG(td_logger_, "setting TESSDATA_PREFIX to " << env_value);
        rc = setenv(env_name.c_str(), env_value.c_str(), 1 /* overwrite existing */);
        if (rc) {
            LOG4CXX_ERROR(td_logger_, "setenv failed: env_name = " << env_name
                                                                   << " env_value = " << env_value);
            return (false);
        }
    }
    string config_file_name = parameters["OPENALPR_CONFIG_FILE"].toStdString();
    string config_file = config_path + "/" + config_file_name;
    LOG4CXX_DEBUG(td_logger_, "OALPR config file: " << config_file);

    // Put the path to the runtime_data directory in the config file, if it
    // hasn't already been done
    std::ifstream conf_file_in(config_file);
    if (!conf_file_in.is_open()) {
        LOG4CXX_ERROR(td_logger_, "Failed to open config file " << config_file
                                                                << ": " << strerror(errno));
        return false;
    }
    bool found = false;
    string line;
    size_t pos;
    while (!conf_file_in.eof()) {
        while ( getline(conf_file_in, line) ) {
            pos = line.find("runtime_dir");
            if (pos != string::npos) {
                found = true;
                break;
            }
        }
    }
    conf_file_in.close();
    if (!found) {
        std::ofstream conf_file_out(config_file, ios::app);
        if (!conf_file_out.is_open()) {
            LOG4CXX_ERROR(td_logger_, "Failed to open config file " << config_file
                                                                    << ": " << strerror(errno));
            return false;
        }
        std::string outpath = "runtime_dir = " + plugin_path + "/runtime_data\n";
        conf_file_out << outpath;
    }
    string country = parameters["OPENALPR_COUNTRY_CODE"].toStdString();
    string template_region;
    bool detect_region = false;
    int top_n = parameters["OPENALPR_TOP_N"].toInt();
    rectangle_intersection_min_ = parameters["RECTANGLE_INTERSECTION_MIN"].toFloat();
    levenshtein_score_min_ = parameters["LEVENSHTEIN_SCORE_MIN"].toFloat();

    string runtimeDir = plugin_path + "/runtime_data";
    LOG4CXX_DEBUG(td_logger_, "config_file = " << config_file << " runtimeDir = " << runtimeDir);
    alpr_ = new Alpr(country, config_file, runtimeDir);
    alpr_->setTopN(top_n);

    if (detect_region)
        alpr_->setDetectRegion(detect_region);

    if (template_region.empty() == false)
        alpr_->setDefaultRegion(template_region);

    if (alpr_->isLoaded() == false) {
        LOG4CXX_ERROR(td_logger_, "Error loading OpenALPR");
        return false;
    }

    return true;

}

//-----------------------------------------------------------------------------
/* virtual */ bool LicensePlateTextDetection::Close() {

    delete alpr_;
    return true;

}

//-----------------------------------------------------------------------------

MPFDetectionError LicensePlateTextDetection::GetDetections(const MPFImageJob &job, vector<MPFImageLocation> &locations) {
    try {

        // No algorithm properties are relevant to the image case
        LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Data_uri: " << job.data_uri);

        assert(!job.data_uri.empty());
        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(td_logger_, "[" << job.job_name << "] Image file URI is empty.");
            return MPF_INVALID_DATAFILE_URI;
        }

        MPFImageReader image_reader(job);
        cv::Mat frame = image_reader.GetImage();

        if (frame.empty()) {
            LOG4CXX_ERROR(td_logger_, "[" << job.job_name << "] Failed to read image.");
            return MPF_IMAGE_READ_ERROR;
        }

        vector<AlprRegionOfInterest> roi;  // unused
        AlprResults RecognizedOutput = alpr_->recognize(frame.data,
                                               frame.elemSize(),
                                               frame.cols,
                                               frame.rows,
                                               roi);

        vector<AlprPlateResult> results = RecognizedOutput.plates;

        LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Results size: " << results.size());

        // NOTE:  for each result, only the detection with the highest confidence is
        // returned since the detection vector passed in is not intended to hold
        // multiple possible detections for a single distinct text object.  However,
        // setting the number of detections (top_n) that alpr should consider to a
        // value greater than 1 tends to improve the quality of all detections.
        LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Returning highest confidence results for detection");
        for (int i = 0; i < results.size(); i++) {
            MPFImageLocation detection;
            detection.x_left_upper = results[i].plate_points[0].x;
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] X Left Upper: " << detection.x_left_upper);
            detection.y_left_upper = results[i].plate_points[0].y;
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Y Left Upper: " << detection.y_left_upper);
            detection.width = results[i].plate_points[1].x - results[i].plate_points[0].x;
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Width: " << detection.width);
            detection.height = results[i].plate_points[3].y - results[i].plate_points[0].y;
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Height: " << detection.height);
            detection.confidence = results[i].topNPlates[0].overall_confidence;
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Confidence: " << detection.confidence);
            SetText(detection, results[i].topNPlates[0].characters);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Text: " << GetText(detection));
            locations.push_back(detection);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Bounding Polygon points: " <<
                                          "(" << results[i].plate_points[0].x << "," << results[i].plate_points[0].y
                                          << ") " <<
                                          "(" << results[i].plate_points[1].x << ", " << results[i].plate_points[1].y
                                          << ") " <<
                                          "(" << results[i].plate_points[2].x << ", " << results[i].plate_points[2].y
                                          << ") " <<
                                          "(" << results[i].plate_points[3].x << ", " << results[i].plate_points[3].y
                                          << ")");


            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] All results");
            for (int k = 0; k < results[i].topNPlates.size(); k++) {
                LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Text: " << results[i].topNPlates[k].characters);
                LOG4CXX_DEBUG(td_logger_,
                              "[" << job.job_name << "] Confidence: " << results[i].topNPlates[k].overall_confidence);
                LOG4CXX_DEBUG(td_logger_,
                              "[" << job.job_name << "] Template Match: " << results[i].topNPlates[k].matches_template);
                LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Bounding Polygon points: " <<
                                              "(" << results[i].plate_points[0].x << ", "
                                              << results[i].plate_points[0].y << ") " <<
                                              "(" << results[i].plate_points[1].x << ", "
                                              << results[i].plate_points[1].y << ") " <<
                                              "(" << results[i].plate_points[2].x << ", "
                                              << results[i].plate_points[2].y << ") " <<
                                              "(" << results[i].plate_points[3].x << ", "
                                              << results[i].plate_points[3].y << ")");
            }
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        LOG4CXX_INFO(td_logger_,
                     "[" << job.job_name << "] Processing complete. Found " << locations.size() << " detections.");
        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, td_logger_);
    }
}

MPFDetectionError LicensePlateTextDetection::GetDetections(const MPFVideoJob &job, vector<MPFVideoTrack> &tracks) {
    try {
        MPFVideoCapture video_capture(job, true, true);
        if (!video_capture.IsOpened()) {
            LOG4CXX_ERROR(td_logger_, "[" << job.job_name << "] Could not initialize OpenCV video capturing.");
            return MPF_COULD_NOT_OPEN_DATAFILE;
        }

        auto detection_result = GetDetectionsFromVideoCapture(job, video_capture, tracks);
        for (auto &track : tracks) {
            video_capture.ReverseTransform(track);
        }
        return detection_result;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, td_logger_);
    }
}

//-----------------------------------------------------------------------------
// Video case
MPFDetectionError LicensePlateTextDetection::GetDetectionsFromVideoCapture(const MPFVideoJob &job,
                                                                           MPFVideoCapture &video_capture,
                                                                           vector<MPFVideoTrack> &tracks) {

    int frame_num = 0;
    int frame_count = 0;
    cv::Mat frame;
    multimap <string, MPFVideoTrack> tracks_map;


    frame_count = video_capture.GetFrameCount();

    LOG4CXX_DEBUG(td_logger_, "frame_count = " << frame_count);
    LOG4CXX_DEBUG(td_logger_, "start_frame = " << job.start_frame);
    LOG4CXX_DEBUG(td_logger_, "stop_frame = " << job.stop_frame);


    while (video_capture.Read(frame)) {
        vector<AlprRegionOfInterest> roi;  // unused
        AlprResults RecognizedOutput = alpr_->recognize(frame.data,
                                                       frame.elemSize(),
                                                       frame.cols,
                                                       frame.rows,
                                                       roi);

        vector <AlprPlateResult> results = RecognizedOutput.plates;
        LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Frame: " << frame_num << " results size: " << results.size());

        // NOTE:  as in image case, for each result only the detection with the
        // highest confidence is used in forming tracks since the track detection
        // vector is not intended to hold multiple possible detections for a
        // single distinct text object.  However, setting the number of detections
        // (top_n) that alpr should consider to a value greater than 1 tends to
        // improve the overall quality of all detections.
        for (int i = 0; i < results.size(); i++) {
            MPFImageLocation detection;
            detection.x_left_upper = results[i].plate_points[0].x;
            detection.y_left_upper = results[i].plate_points[0].y;
            detection.width =
                    results[i].plate_points[1].x - results[i].plate_points[0].x;
            detection.height =
                    results[i].plate_points[3].y - results[i].plate_points[0].y;
            detection.confidence = results[i].topNPlates[0].overall_confidence;
            SetText(detection, results[i].topNPlates[0].characters);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] X Left Upper: " << detection.x_left_upper);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Y Left Upper: " << detection.y_left_upper);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Width: " << detection.width);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Height: " << detection.height);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Confidence: " << detection.confidence);
            LOG4CXX_DEBUG(td_logger_, "[" << job.job_name << "] Text: " << GetText(detection));

            // Determine whether to create a new track
            // or add this detection to an existing track.
            bool create_new_track = true;
            string key_to_use = GetText(detection);

            //TODO:  use equal_range to reduce portion of multimap that is searched
            for (multimap<string, MPFVideoTrack>::iterator iter = tracks_map.begin();
                 iter != tracks_map.end(); iter++) {

                // Compare metadata text to keys in map and determine whether it
                // was also detected in previous contiguous frame
                const string &text = GetText(detection);
                if ((CompareKeys(text, iter->first)) &&
                    (iter->second.stop_frame == frame_num - 1)) {
                    key_to_use = iter->first;
                    // Perform a rectangle intersection to see whether adding to
                    // existing track is reasonable with regard to current
                    // detection's location
                    MPFImageLocation track_obj = iter->second.frame_locations.rbegin()->second; // last element in map
                    cv::Rect current_rect = Utils::ImageLocationToCvRect(
                            detection);
                    cv::Rect track_rect = Utils::ImageLocationToCvRect(
                            track_obj);
                    cv::Rect intersection = current_rect & track_rect;
                    if (intersection.area() > ceil(
                            static_cast<float>(
                                    track_rect.area()) * rectangle_intersection_min_)) {
                        // Add detection to this track and update stop_value
                        iter->second.stop_frame = frame_num;
                        iter->second.frame_locations.insert(pair<int, MPFImageLocation>(frame_num, detection));
                        create_new_track = false;
                    }
                    break;
                }
            }

            if (create_new_track) {
                MPFVideoTrack new_track;
                new_track.start_frame = frame_num;
                new_track.stop_frame = frame_num;
                new_track.frame_locations.insert(pair<int, MPFImageLocation>(frame_num, detection));
                tracks_map.insert(pair<string, MPFVideoTrack>(
                        key_to_use, new_track));
            }
        }

        frame_num++;
    }

    // Return all tracks from map in output vector
    for (multimap<string, MPFVideoTrack>::iterator tracks_map_iter =
            tracks_map.begin();
         tracks_map_iter != tracks_map.end();
         tracks_map_iter++) {
        tracks.push_back(tracks_map_iter->second);
    }

    LOG4CXX_INFO(td_logger_, "[" << job.job_name << "] Processing complete. Found " << tracks.size() << " detections.");

    return MPF_DETECTION_SUCCESS;
}

//-----------------------------------------------------------------------------
bool LicensePlateTextDetection::CompareKeys(
        const string &key_1,
        const string &key_2) {

    // First check for equality
    if (key_1 == key_2) {
        return true;
    }

    // Next, see whether one is a subset of the other
    if ((key_1.find(key_2) != std::string::npos) ||
        (key_2.find(key_1) != std::string::npos)) {
        return true;
    }

    // Finally, calculate a similarity score based on the
    // Levenshtein distance between the two strings
    if (LevenshteinScore(key_1, key_2) > levenshtein_score_min_) {
        return true;
    }

    return false;
}

//-----------------------------------------------------------------------------
// Compute Levenshtein Distance
// Martin Ettl, 2012-10-05
float LicensePlateTextDetection::LevenshteinScore(
        const string &s1,
        const string &s2) {
    const size_t m(s1.size());
    const size_t n(s2.size());

    size_t *costs = new size_t[n + 1];

    for (size_t k = 0; k <= n; k++) costs[k] = k;

    size_t i = 0;
    for (string::const_iterator it1 = s1.begin(); it1 != s1.end(); ++it1, ++i) {
        costs[0] = i + 1;
        size_t corner = i;

        size_t j = 0;
        for (string::const_iterator it2 = s2.begin(); it2 != s2.end(); ++it2, ++j) {
            size_t upper = costs[j + 1];
            if (*it1 == *it2) {
                costs[j + 1] = corner;
            }
            else {
                size_t t(upper < corner ? upper : corner);
                costs[j + 1] = (costs[j] < t ? costs[j] : t) + 1;
            }

            corner = upper;
        }
    }

    size_t result = costs[n];
    delete[] costs;

    int den;
    if (m > n) den = m; else den = n;

    float score = (float) (1.0 - (float) (result) / (float) (den));

    return score;
}

void LicensePlateTextDetection::SetText(MPFImageLocation &detection, const string &text) {
    detection.detection_properties["TEXT"] = text;
}

std::string LicensePlateTextDetection::GetText(const MPFImageLocation &detection) {
    return DetectionComponentUtils::GetProperty<std::string>(detection.detection_properties, "TEXT", "");
}


MPF_COMPONENT_CREATOR(LicensePlateTextDetection);
MPF_COMPONENT_DELETER();
