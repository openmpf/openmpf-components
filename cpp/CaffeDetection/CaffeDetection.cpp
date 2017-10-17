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

#include "CaffeDetection.h"

#include <unistd.h>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>

#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>
#include <boost/algorithm/string.hpp>

#include <Utils.h>
#include <detectionComponentUtils.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>

#include "ModelFileParser.h"


using CONFIG4CPP_NAMESPACE::StringVector;
using namespace MPF::COMPONENT;


//-----------------------------------------------------------------------------
std::string CaffeDetection::GetDetectionType() {
    return "CLASS";
}

//-----------------------------------------------------------------------------
bool CaffeDetection::Init() {

    // Determine where the executable is running
    std::string run_dir = GetRunDirectory();
    if (run_dir.empty()) {
        run_dir = ".";
    }
    std::string plugin_path = run_dir + "/CaffeDetection";
    std::string config_path = plugin_path + "/config";

    // Configure logger

    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    logger_ = log4cxx::Logger::getLogger("CaffeDetection");

    LOG4CXX_DEBUG(logger_, "Plugin path: " << plugin_path);

    LOG4CXX_INFO(logger_, "Initializing Caffe");

    // Load model info from config file
    // A model is defined by a txt file, a bin file, and a synset

    ModelFileParser* parser;
    StringVector model_scopes;
    const char* scope = "";

    std::string config_filepath = config_path + "/models.cfg";
    const char* model_filename = config_filepath.c_str();

    std::string model_path = config_path + "/";
    try {
        parser = new ModelFileParser();
        parser->parse(model_filename, scope);
        parser->listModelScopes(model_scopes);
        int len = model_scopes.length();
        if (len <= 0) {
            LOG4CXX_ERROR(logger_, "Could not parse model file: " << model_path);
            return false;
        } else {
            for (int i = 0; i < len; i++) {
                ModelFiles model_files;
                model_files.model_txt = model_path + parser->getModelTxt(model_scopes[i]);
                model_files.model_bin = model_path + parser->getModelBin(model_scopes[i]);
                model_files.synset_file = model_path + parser->getSynsetTxt(model_scopes[i]);
                model_defs_.insert(std::pair<std::string, ModelFiles>(
                        parser->getName(model_scopes[i]),
                        model_files));
            }
        }
        delete parser;
    } catch(const ModelFileParserException & ex) {
        LOG4CXX_ERROR(logger_, "Could not parse model file: " << model_path << ". " << ex.c_str());
        delete parser;
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
bool CaffeDetection::Close() {
    return true;
}


void addToTrack(MPFImageLocation &location, int frame_index, MPFVideoTrack &track) {
    track.stop_frame = frame_index;
    if (location.confidence > track.confidence) {
        track.confidence = location.confidence;
        track.detection_properties = location.detection_properties;
    }
    track.frame_locations[frame_index] = std::move(location);
}


void defaultTracker(MPFImageLocation &location, int frame_index, std::vector<MPFVideoTrack> &tracks) {
    bool should_start_new_track = tracks.empty()
                               || tracks.back().detection_properties["CLASSIFICATION"]
                                  != location.detection_properties["CLASSIFICATION"];

    if (should_start_new_track) {
        tracks.emplace_back(frame_index, frame_index, location.confidence, location.detection_properties);
    }
    addToTrack(location, frame_index, tracks.back());
}


void feedForwardTracker(MPFImageLocation &location, int frameIndex, std::vector<MPFVideoTrack> &tracks) {
    if (tracks.empty()) {
        tracks.emplace_back(frameIndex, frameIndex, location.confidence, location.detection_properties);
    }
    addToTrack(location, frameIndex, tracks.back());
}



MPFDetectionError CaffeDetection::GetDetections(const MPFVideoJob &job, std::vector<MPFVideoTrack> &tracks) {
    if (job.has_feed_forward_track) {
        return getDetections(job, tracks, feedForwardTracker);
    }
    else {
        return getDetections(job, tracks, defaultTracker);
    }
}


template<typename Tracker>
MPF::COMPONENT::MPFDetectionError CaffeDetection::getDetections(const MPF::COMPONENT::MPFVideoJob &job,
                                                                std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
                                                                Tracker tracker) const {
    try {
        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(logger_, "[" << job.job_name << "] Input video file path is empty");
            return MPF_INVALID_DATAFILE_URI;
        }

        CaffeJobConfig config(job.job_properties, model_defs_, logger_);
        if (config.error != MPF_DETECTION_SUCCESS) {
            return config.error;
        }

        MPFVideoCapture video_cap(job);
        if (!video_cap.IsOpened()) {
            LOG4CXX_ERROR(logger_, "[" << job.job_name << "] Could not initialize capturing");
            return MPF_COULD_NOT_OPEN_DATAFILE;
        }

        cv::Mat frame;
        int frame_index = -1;
        while (video_cap.Read(frame)) {
            frame_index++;
            std::unique_ptr<MPFImageLocation> location;
            MPFDetectionError rc = getDetections(config, frame, location);

            if (rc != MPF_DETECTION_SUCCESS) {
                return rc;
            }
            if (!location) {
                // Nothing found in current frame.
                continue;
            }

            tracker(*location, frame_index, tracks);
        }

        for (MPFVideoTrack &track : tracks) {
            video_cap.ReverseTransform(track);
        }

        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Processing complete. Found " << tracks.size() << " tracks.");

        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }
}


//-----------------------------------------------------------------------------
MPFDetectionError CaffeDetection::GetDetections(const MPFImageJob &job, std::vector<MPFImageLocation> &locations) {
    try {
        CaffeJobConfig config(job.job_properties, model_defs_, logger_);
        if (config.error != MPF_DETECTION_SUCCESS) {
            return config.error;
        }
        LOG4CXX_DEBUG(logger_, "Data URI = " << job.data_uri);

        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(logger_, "Invalid image file");
            return MPF_INVALID_DATAFILE_URI;
        }

        MPFImageReader image_reader(job);
        cv::Mat img = image_reader.GetImage();

        if (img.empty()) {
            LOG4CXX_ERROR(logger_, "Could not read image file: " << job.data_uri);
            return MPF_IMAGE_READ_ERROR;
        }

        std::unique_ptr<MPFImageLocation> detection;
        MPFDetectionError rc = getDetections(config, img, detection);
        if (detection) {
            locations.push_back(std::move(*detection));
        }

        for (MPFImageLocation &location : locations) {
            image_reader.ReverseTransform(location);
        }
        return rc;

    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }

}


//-----------------------------------------------------------------------------
void CaffeDetection::getTopNClasses(cv::Mat &prob_blob,
                                    int num_classes, double threshold,
                                    std::vector< std::pair<int, float> > &classes) const {

    LOG4CXX_DEBUG(logger_, "prob blob mat rows = " << prob_blob.rows << " cols = " << prob_blob.cols);

    cv::Mat prob_mat = prob_blob.reshape(1, 1); // reshape the blob to 1x1000 matrix (googlenet)

    LOG4CXX_DEBUG(logger_, "reshaped prob blob mat rows = " << prob_blob.rows << " cols = " << prob_blob.cols);

    cv::Mat sort_mat;
    cv::sortIdx(prob_mat, sort_mat, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

    for (int i = 0; i < num_classes; i++) {
        int idx = sort_mat.at<int>(i);
        // Keep accumulating until we go below the confidence threshold.
        if (prob_mat.at<float>(0,idx) < threshold) break;
        classes.push_back(std::make_pair(idx, prob_mat.at<float>(0,idx)));
    }
}



MPFDetectionError CaffeDetection::getDetections(CaffeJobConfig &config, const cv::Mat &input_frame,
                                                std::unique_ptr<MPFImageLocation> &location) const {

    cv::Mat prob;
    std::vector<std::pair<std::string, cv::Mat>> activation_layer_mats;
    std::vector<std::pair<SpectralHashInfo, cv::Mat>> spectral_hash_mats;
    getNetworkOutput(config, input_frame, prob, activation_layer_mats, spectral_hash_mats);

    LOG4CXX_DEBUG(logger_, "output prob mat rows = " << prob.rows << " cols = " << prob.cols);
    LOG4CXX_DEBUG(logger_, "output prob mat total: " << prob.total());

    // The number of classifications requested must be greater
    // than 0 and less than the total size of the output blob.
    if ((config.number_of_classifications <= 0) || (config.number_of_classifications > prob.total())) {
        LOG4CXX_ERROR(logger_, "Number of classifications requested: "
                << config.number_of_classifications
                << " is invalid. It must be greater than 0, and less than the total returned by the net output layer = "
                << prob.total());
        return MPF_INVALID_PROPERTY;
    }

    std::vector<std::pair<int, float>> class_info;
    getTopNClasses(prob, config.number_of_classifications, config.confidence_threshold, class_info);

    if (class_info.empty() && activation_layer_mats.empty() && spectral_hash_mats.empty()) {
        return MPF_DETECTION_SUCCESS;
    }


    location = std::unique_ptr<MPFImageLocation>(new MPFImageLocation(0, 0, input_frame.cols, input_frame.rows));

    if (!class_info.empty()) {
        // Save the highest confidence classification as the
        // "CLASSIFICATION" property, and its corresponding confidence
        // as the MPFImageLocation confidence.
        LOG4CXX_DEBUG(logger_, "class id #0: " << class_info[0].first);
        LOG4CXX_DEBUG(logger_, "confidence: " << class_info[0].second);
        location->confidence =  class_info[0].second;
        location->detection_properties["CLASSIFICATION"] = config.class_names.at(class_info[0].first);

        // Begin accumulating the classifications in a stringstream
        // for the "CLASSIFICATION LIST"
        std::stringstream ss_ids;
        ss_ids << config.class_names.at(class_info[0].first);

        // Use another stringstream for the
        // "CLASSIFICATION CONFIDENCE LIST"
        std::stringstream ss_conf;
        ss_conf << class_info[0].second;

        for (int i = 1; i < class_info.size(); i++) {
            LOG4CXX_DEBUG(logger_, "class id #" << i << ": " << class_info[i].first);
            LOG4CXX_DEBUG(logger_, "confidence: " << class_info[i].second);
            ss_ids << "; " << config.class_names.at(class_info[i].first);
            ss_conf << "; " << class_info[i].second;
        }
        location->detection_properties["CLASSIFICATION LIST"] = ss_ids.str();
        location->detection_properties["CLASSIFICATION CONFIDENCE LIST"] = ss_conf.str();
    }

    addActivationLayerInfo(config, activation_layer_mats, location->detection_properties);
    addSpectralHashInfo(config, spectral_hash_mats, location->detection_properties);

    return MPF_DETECTION_SUCCESS;
}



void CaffeDetection::addActivationLayerInfo(const CaffeDetection::CaffeJobConfig &config,
                                            const std::vector<std::pair<std::string, cv::Mat>> &activation_layer_mats,
                                            MPF::COMPONENT::Properties &detection_properties) {

    for (const auto &activation_pair : activation_layer_mats) {
        // Create a JSON-formatted string to represent the activation
        // values matrix.
        std::string name = activation_pair.first;
        std::string filename = name + ".json";
        cv::FileStorage act_store(filename, cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
        act_store << "activation values" << activation_pair.second;
        std::string actString = act_store.releaseAndGetString();
        std::transform(name.begin(), name.end(), name.begin(), ::toupper);
        name += " ACTIVATION MATRIX";
        detection_properties[name] = actString;
    }

    if (!config.bad_activation_layer_names.empty()) {
        detection_properties["INVALID ACTIVATION LAYER LIST"] = boost::algorithm::join(config.bad_activation_layer_names, "; ");
    }
}



void CaffeDetection::addSpectralHashInfo(CaffeDetection::CaffeJobConfig &config,
                                         const std::vector<std::pair<SpectralHashInfo, cv::Mat>> &spectral_hash_mats,
                                         Properties &detection_properties) const {

    for (const auto& hash_info_pair : spectral_hash_mats) {
        try {
            detection_properties.emplace(computeSpectralHash(hash_info_pair.second, hash_info_pair.first));
        } catch (const cv::Exception &err) {
            LOG4CXX_ERROR(logger_, "OpenCV exception caught while calculating the spectral hash for layer \""
                          << hash_info.layer_name << "\" in model named \""
                          << hash_info.model_name << "\": " << err.what());
            config.bad_hash_file_names.push_back(hash_info_pair.first.file_name);
        }
    }

    if (!config.bad_hash_file_names.empty()) {
        detection_properties["INVALID SPECTRAL HASH FILENAME LIST"]
                = boost::algorithm::join(config.bad_hash_file_names, "; ");
    }
}



std::pair<std::string, std::string> CaffeDetection::computeSpectralHash(const cv::Mat &activations,
                                                                        const SpectralHashInfo &hash_info) const {
    cv::Mat omega0;
    cv::Mat omegas;
    std::string bitset;

        omega0 = CV_PI / (hash_info.mx - hash_info.mn);
        omegas = cv::repeat(omega0, hash_info.modes.rows, 1);
        omegas = omegas.mul(hash_info.modes);
        cv::Mat x = cv::repeat( ((activations * hash_info.pc) - hash_info.mn), omegas.rows, 1).mul(omegas);
        if (hash_info.nbits != x.rows) {
            LOG4CXX_WARN(logger_, "Number of bits in the spectral hash for layer \"" << hash_info.layer_name
                         << "\" in model named \"" << hash_info.model_name
                         << "\" is not equal to the input nbits value: nbits = " << hash_info.nbits
                         << ", spectral hash size = " << x.rows);
        }

        for(int r = 0; r < x.rows; r++){
            int bit = 1;
            for(int c = 0; c < x.cols; c++){
                bit *= ((cos(x.at<float>(r, c)) > 0) ? 1 : -1);
            }
            if(bit > 0) {
                bitset += "1";
            } else {
                bitset += "0";
            }
        }

    std::string name(hash_info.layer_name);
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    name += " SPECTRAL HASH VALUE";
    return std::make_pair(name, bitset);
}


void CaffeDetection::getNetworkOutput(CaffeDetection::CaffeJobConfig &config,
                                      const cv::Mat &input_frame,
                                      cv::Mat &output_layer,
                                      std::vector<std::pair<std::string, cv::Mat>> &activation_layer_info,
                                      std::vector<std::pair<SpectralHashInfo, cv::Mat>> &spectral_hash_info) {
    cv::Mat frame;
    cv::resize(input_frame, frame, config.resize_size);

    cv::Rect roi(config.crop_size, frame.size() - (config.crop_size * 2));
    frame = frame(roi);

    // convert Mat to batch of images (BGR)
    cv::Mat input_blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(), config.subtract_colors, false); // swapRB = false

    config.net.setInput(input_blob, "data");


    std::vector<cv::Mat> net_output;
    config.net.forward(net_output, config.output_layers);
    assert(net_output.size() == 1 + config.requested_activation_layer_names.size() + config.spectral_hash_info.size());

    auto net_out_iter = std::make_move_iterator(net_output.begin());
    output_layer = *net_out_iter++;

    for (const auto &layer_name : config.requested_activation_layer_names) {
        activation_layer_info.emplace_back(layer_name, *net_out_iter++);
    }

    for (const auto &hash_info : config.spectral_hash_info) {
        spectral_hash_info.emplace_back(hash_info, *net_out_iter++);
    }
}


//-----------------------------------------------------------------------------

MPF_COMPONENT_CREATOR(CaffeDetection);
MPF_COMPONENT_DELETER();


CaffeDetection::CaffeJobConfig::CaffeJobConfig(const Properties &props,
                                               const std::map<std::string, ModelFiles> &model_defs,
                                               const log4cxx::LoggerPtr &logger) {

    using namespace DetectionComponentUtils;

    std::string model_name = GetProperty<std::string>(props, "MODEL_NAME", "googlenet");
    auto iter = model_defs.find(model_name);
    if (iter == model_defs.end()) {
        LOG4CXX_ERROR(logger, "Could not load specified model: " << model_name);
        error = MPF_DETECTION_NOT_INITIALIZED;
        return;
    }

    LOG4CXX_INFO(logger, "Get detections using model: " << model_name);

    ModelFiles model_files = iter->second;

    error = readClassNames(model_files.synset_file, class_names);
    if (error != MPF_DETECTION_SUCCESS) {
        LOG4CXX_ERROR(logger, "Failed to read class labels for the network");
        return;
    }

    if (class_names.empty()) {
        LOG4CXX_ERROR(logger, "No network class labels found");
        error = MPF_DETECTION_FAILED;
        return;
    }

    // try to import Caffe model
    net = cv::dnn::readNetFromCaffe(model_files.model_txt, model_files.model_bin);
    if (net.empty()) {
        LOG4CXX_ERROR(logger, "Can't load network specified by the following files: ");
        LOG4CXX_ERROR(logger, "prototxt:   " << model_files.model_txt);
        LOG4CXX_ERROR(logger, "caffemodel: " << model_files.model_bin);
        error = MPF_DETECTION_NOT_INITIALIZED;
        return;
    }
    LOG4CXX_DEBUG(logger, "Created neural network");

    resize_size = cv::Size(GetProperty(props, "RESIZE_WIDTH", 224), GetProperty(props, "RESIZE_HEIGHT", 224));

    crop_size = cv::Size(GetProperty(props, "LEFT_AND_RIGHT_CROP", 0), GetProperty(props, "TOP_AND_BOTTOM_CROP", 0));

    subtract_colors = cv::Scalar(GetProperty(props, "SUBTRACT_BLUE_VALUE", 0.0),
                                 GetProperty(props, "SUBTRACT_GREEN_VALUE", 0.0),
                                 GetProperty(props, "SUBTRACT_RED_VALUE", 0.0));


    const std::vector<cv::String> &net_layer_names = net.getLayerNames();

    model_output_layer = GetProperty(props, "MODEL_OUTPUT_LAYER", std::string("prob"));
    bool output_layer_missing = std::find(net_layer_names.begin(), net_layer_names.end(), model_output_layer)
                                    == net_layer_names.end();
    if (output_layer_missing) {
        LOG4CXX_WARN(logger, "The requested output layer: " << model_output_layer << " does not exist");
        error = MPF_INVALID_PROPERTY;
    }


    validateLayerNames(
            GetProperty(props, "ACTIVATION_LAYER_LIST", std::string()),
            net_layer_names, model_name, logger);

    getSpectralHashInfo(
            GetProperty(props, "SPECTRAL_HASH_FILE_LIST", std::string()),
            net_layer_names, model_name, logger);


    output_layers.reserve(1 + requested_activation_layer_names.size() + spectral_hash_info.size());
    output_layers.emplace_back(model_output_layer);
    output_layers.insert(output_layers.end(),
                         requested_activation_layer_names.begin(), requested_activation_layer_names.end());
    for (const auto &hash_info : spectral_hash_info) {
        output_layers.emplace_back(hash_info.layer_name);
    }

    number_of_classifications = GetProperty(props, "NUMBER_OF_CLASSIFICATIONS", 1);
    confidence_threshold = GetProperty(props, "CONFIDENCE_THRESHOLD", 0.0);
    if (confidence_threshold < 0.0) {
        LOG4CXX_ERROR(logger, "The confidence threshold requested: "
                << confidence_threshold
                << " is invalid. It must be greater than 0.0.");
        error = MPF_INVALID_PROPERTY;
    }
}



MPFDetectionError CaffeDetection::CaffeJobConfig::readClassNames(std::string synset_file,
                                                                 std::vector<std::string> &class_names) {
    std::ifstream fp(synset_file);
    if (!fp.is_open()) {
        return MPF_COULD_NOT_OPEN_DATAFILE;
    }
    std::string name;
    while (!fp.eof())
    {
        getline(fp, name);
        if (name.length())
            class_names.push_back(name.substr(name.find(' ') + 1));
    }
    fp.close();
    return MPF_DETECTION_SUCCESS;
}



void CaffeDetection::CaffeJobConfig::validateLayerNames(std::string requested_activation_layers,
                                                        const std::vector<cv::String> &net_layers,
                                                        const std::string &model_name,
                                                        const log4cxx::LoggerPtr &logger) {
    // Get the layers in the net and check that each layer
    // requested is actually part of the net. If it is, add it to the
    // vector of layer names for which we need the layer output. If
    // not, remember the name so that we can indicate in the output
    // that it was not found.

    if (!requested_activation_layers.empty()) {
        boost::trim(requested_activation_layers);
        std::vector<std::string> names;
        boost::split(names, requested_activation_layers,
                     boost::is_any_of(" ;"),
                     boost::token_compress_on);
        for (const std::string &name : names) {
            if (!name.empty()) {
                if (std::find(net_layers.begin(), net_layers.end(), name) != net_layers.end()) {
                    requested_activation_layer_names.push_back(name);
                }
                else {
                    LOG4CXX_WARN(logger, "Layer named \"" << name << "\" was not found in model named \""
                                                          << model_name << "\"");
                    bad_activation_layer_names.push_back(name);
                }
            }
        }
    }
}


bool CaffeDetection::CaffeJobConfig::parseAndValidateHashInfo(const std::string &file_name,
                                                              cv::FileStorage &sp_params,
                                                              SpectralHashInfo &hash_info,
                                                              const log4cxx::LoggerPtr &logger) {
    bool is_good_file_name = true;

    if (sp_params["nbits"].empty()) {
        LOG4CXX_WARN(logger, "The \"nbits\" field in file \"" << file_name << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        sp_params["nbits"] >> hash_info.nbits;
        if (hash_info.nbits <= 0) {
            LOG4CXX_WARN(logger, "The \"nbits\" value in file \"" << file_name << "\" is less than or equal to zero.");
            is_good_file_name = false;
        }
    }

    if (sp_params["mx"].empty()) {
        LOG4CXX_WARN(logger, "The \"mx\" field in file \"" << file_name << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        sp_params["mx"] >> hash_info.mx;
        if (hash_info.mx.empty()) {
            LOG4CXX_WARN(logger, "The \"mx\" matrix in file \"" << file_name << "\" is empty.");
            is_good_file_name = false;
        }
    }
    if (sp_params["mn"].empty()) {
        LOG4CXX_WARN(logger, "The \"mn\" field in file \"" << file_name << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        sp_params["mn"] >> hash_info.mn;
        if (hash_info.mn.empty()) {
            LOG4CXX_WARN(logger, "The \"mn\" matrix in file \"" << file_name << "\" is empty.");
            is_good_file_name = false;
        }
    }
    if (sp_params["modes"].empty()) {
        LOG4CXX_WARN(logger, "The \"modes\" field in file \"" << file_name << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        sp_params["modes"] >> hash_info.modes;
        if (hash_info.modes.empty()) {
            LOG4CXX_WARN(logger, "The \"modes\" matrix in file \"" << file_name << "\" is empty.");
            is_good_file_name = false;
        }
    }
    if(sp_params["pc"].empty()) {
        LOG4CXX_WARN(logger, "The \"pc\" field in file \"" << file_name << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        sp_params["pc"] >> hash_info.pc;
        if (hash_info.pc.empty()) {
            LOG4CXX_WARN(logger, "The \"pc\" matrix in file \"" << file_name << "\" is empty.");
            is_good_file_name = false;
        }
    }
    sp_params.release();

    return is_good_file_name;
}


void CaffeDetection::CaffeJobConfig::getSpectralHashInfo(std::string hash_file_list,
                                                         const std::vector<cv::String> &net_layers,
                                                         const std::string &model_name,
                                                         const log4cxx::LoggerPtr &logger) {
    LOG4CXX_DEBUG(logger, "Loading spectral hash parameters");
    if (!hash_file_list.empty()) {
        boost::trim(hash_file_list);
        std::vector<std::string> files;
        boost::split(files, hash_file_list,
                     boost::is_any_of(" ;"),
                     boost::token_compress_on);

        for (const std::string &file_name : files) {
            LOG4CXX_DEBUG(logger, "file_name = " << file_name);
            std::string err_string;
            std::string exp_filename;
            err_string = Utils::expandFileName(file_name, exp_filename);
            if (!err_string.empty()) {
                LOG4CXX_WARN(logger, "Expansion of spectral hash input filename \"" << file_name << "\" failed: error reported was \"" << err_string << "\"");
                bad_hash_file_names.push_back(file_name);
            }
            else {
                SpectralHashInfo hash_info;
                try {
                    cv::FileStorage sp_params(exp_filename, cv::FileStorage::READ);
                    if (!sp_params.isOpened()) {
                        LOG4CXX_WARN(logger, "Failed to open spectral hash file named \"" << exp_filename << "\"");
                        bad_hash_file_names.push_back(file_name);
                    }
                    else {
                        if (sp_params["layer_name"].empty()) {
                            LOG4CXX_WARN(logger, "The \"layer_name\" field in file \"" << exp_filename << "\" is missing.");
                            bad_hash_file_names.push_back(file_name);
                        }
                        else {
                            sp_params["layer_name"] >> hash_info.layer_name;
                            LOG4CXX_DEBUG(logger, "layer_name = " << hash_info.layer_name);
                            if (std::find(net_layers.begin(),
                                          net_layers.end(),
                                          hash_info.layer_name) != net_layers.end()) {
                                bool is_good_file = parseAndValidateHashInfo(exp_filename,
                                                                             sp_params,
                                                                             hash_info,
                                                                             logger);
                                if (is_good_file) {
                                    // Everything checks out ok, so
                                    // save the hash info and the
                                    // layer name. Also save the
                                    // original file name in case
                                    // there is a subsequent error in
                                    // the spectral hash calculation;
                                    // we can then add the file to the
                                    // list of bad files.
                                    hash_info.file_name = file_name;
                                    hash_info.model_name = model_name;
                                    spectral_hash_info.push_back(hash_info);
                                }
                                else {
                                    bad_hash_file_names.push_back(file_name);
                                }
                            }
                            else {
                                LOG4CXX_WARN(logger, "Layer named \"" << hash_info.layer_name
                                             << "\" from spectral hash file \"" << file_name
                                             << "\" was not found in the model named \"" << model_name << "\"");
                                bad_hash_file_names.push_back(file_name);
                            }
                        }
                    }
                } catch(const cv::Exception &err){
                    LOG4CXX_WARN(logger, "Exception caught when processing spectral hash file named \""
                                 << file_name << "\": " << err.what());
                    bad_hash_file_names.push_back(file_name);
                }
            }
        }
    }
}
