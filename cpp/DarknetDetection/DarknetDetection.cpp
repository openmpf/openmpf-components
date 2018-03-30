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

#include <fstream>
#include <unordered_map>

#include <log4cxx/xml/domconfigurator.h>

#include <MPFImageReader.h>
#include <MPFVideoCapture.h>
#include <detectionComponentUtils.h>
#include <Utils.h>
#include <unordered_set>
#include <MPFInvalidPropertyException.h>

#include "Trackers.h"
#include "DarknetDetection.h"


using namespace MPF::COMPONENT;


namespace {
    class NoOpFilter {
    public:

        NoOpFilter(const Properties&, const std::vector<std::string>&) {

        }

        bool operator()(const std::string&) {
            return true;
        }
    };


    void trim(std::string& str) {
        auto first_non_space = std::find_if_not(str.begin(), str.end(), [](char c) { return std::isspace(c); });
        str.erase(str.begin(), first_non_space);

        auto last_non_space = std::find_if_not(str.rbegin(), str.rend(),  [](char c) { return std::isspace(c); });
        str.erase(last_non_space.base(), str.end());
    }


    class WhitelistFilter {
    public:

        WhitelistFilter(const MPF::COMPONENT::Properties& props, const std::vector<std::string>& names) {
            const std::string &whitelist_path = props.at("CLASS_WHITELIST_FILE");
            std::string expanded_file_path;
            std::string error = Utils::expandFileName(whitelist_path, expanded_file_path);
            if (!error.empty()) {
                throw MPFInvalidPropertyException(
                        "CLASS_WHITELIST_FILE",
                        "The value, \"" + whitelist_path + "\", could not be expanded due to: " + error);
            }

            std::ifstream whitelist_file(expanded_file_path);
            if (!whitelist_file.good()) {
                throw MPFDetectionException(
                        MPF_COULD_NOT_OPEN_DATAFILE,
                        "Failed to load class whitelist that was supposed to be located at \""
                            + expanded_file_path + "\".");
            }

            std::unordered_set<std::string> temp_whitelist;
            std::string line;
            while (std::getline(whitelist_file, line)) {
                trim(line);
                if (!line.empty()) {
                    temp_whitelist.insert(line);
                }
            }

            if (temp_whitelist.empty()) {
                throw MPFDetectionException(
                        MPF_COULD_NOT_READ_DATAFILE,
                        "The class whitelist file located at \"" + expanded_file_path + "\" was empty.");
            }

            for (const std::string &name : names) {
                if (temp_whitelist.count(name) > 0) {
                    whitelist_.insert(name);
                }
            }

            if (whitelist_.empty()) {
                throw MPFDetectionException(
                        MPF_COULD_NOT_READ_DATAFILE,
                        "None of the class names specified in the whitelist file located at \""
                             + expanded_file_path + "\" were found in the names file.");
            }
        }

        bool operator()(const std::string& class_name) {
            return whitelist_.count(class_name) > 0;
        }

    private:
        std::unordered_set<std::string> whitelist_;

    };


    // Darknet functions that accept C style strings, accept a char* instead of a const char*.
    std::unique_ptr<char[]> ToNonConstCStr(const std::string &str) {
        std::unique_ptr<char[]> result(new char[str.size() + 1]);
        size_t length = str.copy(result.get(), str.size());
        result[length] = '\0';
        return result;
    }

    DarknetDetection::network_ptr_t LoadNetwork(const ModelSettings &model_settings) {
        auto cfg_file = ToNonConstCStr(model_settings.network_config_file);
        auto weights_file = ToNonConstCStr(model_settings.weights_file);

        return { load_network(cfg_file.get(), weights_file.get(), 0),
                 free_network };
    }


    int GetOutputLayerSize(const network &network) {
        layer output_layer = network.layers[network.n - 1];
        return output_layer.w * output_layer.h * output_layer.n;
    }


    int GetNumClasses(const network &network) {
        layer output_layer = network.layers[network.n - 1];
        return output_layer.classes;
    }


    std::vector<std::string> LoadNames(const ModelSettings &model_settings, int expected_name_count) {
        std::ifstream in_file(model_settings.names_file);
        if (!in_file.good()) {
            throw std::runtime_error("Failed to open names file at: " + model_settings.names_file);
        }

        std::vector<std::string> names;
        names.reserve(static_cast<size_t>(expected_name_count));

        std::string line;
        while (std::getline(in_file, line).good()) {
            trim(line);
            names.push_back(line);
        }

        if (names.size() != expected_name_count) {
            std::stringstream error;
            error << "Error: The network config file at " << model_settings.network_config_file << " specifies "
                  << expected_name_count << " classes, but the names file at " << model_settings.names_file << " contains "
                  << names.size() << " classes. This is probably because given names file does not correspond to the "
                  << "given network configuration file.";
            throw std::runtime_error(error.str());
        }

        return names;
    }



    cv::Rect BoxToRect(const box &box, const cv::Size &image_size) {
        // box.x and box.y refer to the center of the rectangle,
        // but cv::Rect uses the top left x and y coordinates.
        auto tlX = static_cast<int>(box.x - box.w / 2.0f);
        auto tlY = static_cast<int>(box.y - box.h / 2.0f);
        auto width = static_cast<int>(box.w);
        auto height = static_cast<int>(box.h);

        return cv::Rect(tlX, tlY, width, height) & cv::Rect(cv::Point(0, 0), image_size);
    }

    bool HasWhitelist(const MPFJob &job) {
        return !DetectionComponentUtils::GetProperty(job.job_properties, "CLASS_WHITELIST_FILE", std::string())
                .empty();

    }
}



bool DarknetDetection::Init() {
    std::string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }

    std::string plugin_path = run_dir + "/DarknetDetection";

    log4cxx::xml::DOMConfigurator::configure(plugin_path + "/config/Log4cxxConfig.xml");
    logger_ = log4cxx::Logger::getLogger("DarknetDetection");

    try {
        models_parser_.Init(plugin_path + "/models")
                .RegisterField("network_config", &ModelSettings::network_config_file)
                .RegisterField("names", &ModelSettings::names_file)
                .RegisterField("weights", &ModelSettings::weights_file);
    }
    catch (const std::exception &ex) {
        LOG4CXX_ERROR(logger_, "Failed to initialize ModelsIniParser due to: " << ex.what())
        return false;
    }

    LOG4CXX_INFO(logger_, "Initialized DarknetDetection component.");

    return true;
}


MPFDetectionError DarknetDetection::GetDetections(const MPFVideoJob &job, std::vector<MPFVideoTrack> &tracks) {
    LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");
    if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
        PreprocessorTracker tracker;
        return GetDetections(job, tracks, tracker);
    }
    else {
        int number_of_classifications = DetectionComponentUtils::GetProperty(
                job.job_properties, "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5);
        SingleDetectionPerTrackTracker tracker(number_of_classifications);
        return GetDetections(job, tracks, tracker);
    }
}


template<typename Tracker>
MPFDetectionError DarknetDetection::GetDetections(const MPFVideoJob &job, std::vector<MPFVideoTrack> &tracks,
                                                  Tracker &tracker) {
    if (HasWhitelist(job)) {
        return GetDetectionsWithFilter<Tracker, WhitelistFilter>(job, tracks, tracker);
    }
    else {
        return GetDetectionsWithFilter<Tracker, NoOpFilter>(job, tracks, tracker);
    }
}


template<typename Tracker, typename ClassFilter>
MPFDetectionError DarknetDetection::GetDetectionsWithFilter(
        const MPFVideoJob &job, std::vector<MPFVideoTrack> &tracks, Tracker &tracker) {
    try {
        Detector<ClassFilter> detector(job.job_properties, GetModelSettings(job.job_properties));

        MPFVideoCapture video_cap(job);
        if (!video_cap.IsOpened()) {
            LOG4CXX_ERROR(logger_, "[" << job.job_name << "] Could not open video file: " << job.data_uri);
            return MPF_COULD_NOT_OPEN_DATAFILE;
        }

        cv::Mat frame;
        int frame_number = -1;

        while (video_cap.Read(frame)) {
            frame_number++;
            tracker.ProcessFrameDetections(detector.Detect(frame), frame_number);
        }

        tracks = tracker.GetTracks();

        for (MPFVideoTrack &track : tracks) {
            video_cap.ReverseTransform(track);
        }

        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << tracks.size() << " tracks.");

        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }
}



MPFDetectionError DarknetDetection::GetDetections(const MPFImageJob &job, std::vector<MPFImageLocation> &locations) {
    try {
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");
        MPFImageReader image_reader(job);
        ModelSettings model_settings = GetModelSettings(job.job_properties);
        const cv::Mat image = image_reader.GetImage();
        std::vector<DarknetResult> results = HasWhitelist(job)
                                        ? Detector<WhitelistFilter>(job.job_properties, model_settings).Detect(image)
                                        : Detector<NoOpFilter>(job.job_properties, model_settings).Detect(image);

        if (DetectionComponentUtils::GetProperty(job.job_properties, "USE_PREPROCESSOR", false)) {
            ConvertResultsUsingPreprocessor(results, locations);
        }
        else {
            int number_of_classifications
                    = std::max(1, DetectionComponentUtils::GetProperty(job.job_properties,
                                                                       "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5));
            for (auto& result : results) {
                locations.push_back(
                        SingleDetectionPerTrackTracker::CreateImageLocation(number_of_classifications, result));
            }
        }

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << locations.size() << " detections.");

        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }
}


ModelSettings DarknetDetection::GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const {
    const std::string &model_name = DetectionComponentUtils::GetProperty<std::string>(
            job_properties, "MODEL_NAME", "tiny yolo");

    const std::string &models_dir_path = DetectionComponentUtils::GetProperty<std::string>(
            job_properties, "MODELS_DIR_PATH", ".");

    return models_parser_.ParseIni(model_name, models_dir_path + "/DarknetDetection");
}


void DarknetDetection::ConvertResultsUsingPreprocessor(std::vector<DarknetResult> &darknet_results,
                                                       std::vector<MPFImageLocation> &locations) {

    std::unordered_map<std::string, MPFImageLocation> type_to_image_loc;

    for (DarknetResult &darknet_result : darknet_results) {
        const cv::Rect &rect = darknet_result.detection_rect;

        for (std::pair<float, std::string> &class_prob : darknet_result.object_type_probs) {
            auto it = type_to_image_loc.find(class_prob.second);
            if (it == type_to_image_loc.cend()) {
                type_to_image_loc.emplace(
                        class_prob.second,
                        MPFImageLocation(rect.x, rect.y, rect.width, rect.height, class_prob.first,
                                         Properties{ {"CLASSIFICATION", class_prob.second} }));
            }
            else {
                PreprocessorTracker::CombineImageLocation(rect, class_prob.first, it->second);
            }
        }
    }

    for (auto &image_loc_pair : type_to_image_loc) {
        locations.push_back(std::move(image_loc_pair.second));
    }
}


std::string DarknetDetection::GetDetectionType() {
    return "CLASS";
}



bool DarknetDetection::Close() {
    return true;
}



DarknetDetection::DarknetImageHolder::DarknetImageHolder(const cv::Mat &cv_image) {
    darknet_image = make_image(cv_image.cols, cv_image.rows, cv_image.channels());

    // This code is mostly copied from Darknet's "ipl_into_image" function. The main difference is that
    // this function works with cv::Mat instead of IplImage. IplImage is a legacy OpenCV image type.
    // The OpenCV documentation for cv::Mat says that cv::Mat and IplImage use a compatible data layout.
    int height = darknet_image.h;
    int width = darknet_image.w;
    int channels = cv_image.channels();
    size_t step = cv_image.step[0];

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            for (int channel = 0; channel < channels; channel++) {
                darknet_image.data[channel * width * height + row * width + col]
                        = cv_image.data[row * step + col * channels + channel] / 255.0f;
            }
        }
    }
    rgbgr_image(darknet_image);
}


DarknetDetection::DarknetImageHolder::~DarknetImageHolder() {
    free_image(darknet_image);
}


DarknetDetection::ProbHolder::ProbHolder(int output_layer_size, int num_classes)
    // The make_probs function from the Darknet library allocates num_classes + 1 floats per row.
    // There is no documentation so it isn't clear why it uses num_classes + 1 instead of just num_classes.
    : mat_size_(output_layer_size * (num_classes + 1))
    , prob_mat_(new float[mat_size_]())
    , prob_row_ptrs_(new float*[output_layer_size])
{
    for (int i = 0; i < output_layer_size; i++) {
        prob_row_ptrs_[i] = &prob_mat_[i * (num_classes + 1)];
    }
}


float** DarknetDetection::ProbHolder::Get() {
    return prob_row_ptrs_.get();
}


float* DarknetDetection::ProbHolder::operator[](size_t box_idx) {
    return prob_row_ptrs_[box_idx];
}


void DarknetDetection::ProbHolder::Clear() {
    for (int i = 0; i < mat_size_; i++) {
        prob_mat_[i] = 0.0f;
    }
}


template <typename ClassFilter>
DarknetDetection::Detector<ClassFilter>::Detector(const Properties &props, const ModelSettings &settings)
    : network_(LoadNetwork(settings))
    , output_layer_size_(GetOutputLayerSize(*network_))
    , num_classes_(GetNumClasses(*network_))
    , names_(LoadNames(settings, num_classes_))
    , class_filter_(props, names_)
    // Darknet will output of a probability for every possible class regardless of the content of the image.
    // Most of these classes will have a probability of zero or a number very close to zero.
    // If the confidence threshold is zero or smaller it will report every possible classification.
    , confidence_threshold_(DetectionComponentUtils::GetProperty(props, "CONFIDENCE_THRESHOLD", 0.5f))
    , probs_(output_layer_size_, num_classes_)
    , boxes_(new box[output_layer_size_])
{


}


template <typename ClassFilter>
std::vector<DarknetResult> DarknetDetection::Detector<ClassFilter>::Detect(const cv::Mat &cv_image) {
    DarknetImageHolder image(cv_image);
    probs_.Clear();

    // There is no documentation explaining what hier_thresh and nms do,
    // so we are just using the default values from the Darknet library.
    float hier_thresh = 0.5;
    float nms = 0.3;
    network_detect(network_.get(), image.darknet_image, confidence_threshold_, hier_thresh, nms, boxes_.get(),
                   probs_.Get());

    std::vector<DarknetResult> detections;
    cv::Size frame_size = cv_image.size();

    for (int box_idx = 0; box_idx < output_layer_size_; box_idx++) {
        DarknetResult detection { BoxToRect(boxes_[box_idx], frame_size) };

        for (int name_idx = 0; name_idx < num_classes_; name_idx++) {
            float prob = probs_[box_idx][name_idx];
            const std::string &name = names_[name_idx];

            if (prob >= confidence_threshold_ && class_filter_(name)) {
                detection.object_type_probs.emplace_back(prob, name);
            }
        }

        if (!detection.object_type_probs.empty()) {
            detections.push_back(std::move(detection));
        }
    }

    return detections;
}




MPF_COMPONENT_CREATOR(DarknetDetection);
MPF_COMPONENT_DELETER();