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
#include <sstream>
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

#include "ModelFileParser.h"

using CONFIG4CPP_NAMESPACE::StringVector;
using namespace MPF::COMPONENT;

void CaffeDetection::getLayerNameLists(const std::vector<cv::String> &names_to_search,
                                       const std::string &model_name,
                                       std::string &name_list,
                                       std::vector<std::string> &good_names,
                                       std::vector<std::string> &bad_names) {

    if (!name_list.empty()) {
        boost::trim(name_list);
        std::vector<std::string> names;
        boost::split(names, name_list,
                     boost::is_any_of(" ;"),
                     boost::token_compress_on);
        for (const std::string &name : names) {
            if (!name.empty()) {
                if (std::find(names_to_search.begin(),
                              names_to_search.end(), name) != names_to_search.end()) {
                    good_names.push_back(name);
                }
                else {
                    LOG4CXX_WARN(logger_, "Layer named \"" << name << "\" was not found in model named \"" << model_name << "\"");
                    bad_names.push_back(name);
                }
            }
        }
    }
}

bool CaffeDetection::parseAndValidateHashInfo(const std::string filename,
                                              cv::FileStorage &spParams,
                                              SpectralHashInfo &hash_info) {
    bool is_good_file_name = true;

    if (spParams["nbits"].empty()) {
        LOG4CXX_WARN(logger_, "The \"nbits\" field in file \"" << filename << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        spParams["nbits"] >> hash_info.nbits;
        if (hash_info.nbits <= 0) {
            LOG4CXX_WARN(logger_, "The \"nbits\" value in file \"" << filename << "\" is less than or equal to zero.");
            is_good_file_name = false;
        }
    }

    if (spParams["mx"].empty()) {
        LOG4CXX_WARN(logger_, "The \"mx\" field in file \"" << filename << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        spParams["mx"]    >> hash_info.mx;
        if (hash_info.mx.empty()) {
            LOG4CXX_WARN(logger_, "The \"mx\" matrix in file \"" << filename << "\" is empty.");
            is_good_file_name = false;
        }
    }
    if (spParams["mn"].empty()) {
        LOG4CXX_WARN(logger_, "The \"mn\" field in file \"" << filename << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        spParams["mn"]    >> hash_info.mn;
        if (hash_info.mn.empty()) {
            LOG4CXX_WARN(logger_, "The \"mn\" matrix in file \"" << filename << "\" is empty.");
            is_good_file_name = false;
        }
    }
    if (spParams["modes"].empty()) {
        LOG4CXX_WARN(logger_, "The \"modes\" field in file \"" << filename << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        spParams["modes"] >> hash_info.modes;
        if (hash_info.modes.empty()) {
            LOG4CXX_WARN(logger_, "The \"modes\" matrix in file \"" << filename << "\" is empty.");
            is_good_file_name = false;
        }
    }
    if(spParams["pc"].empty()) {
        LOG4CXX_WARN(logger_, "The \"pc\" field in file \"" << filename << "\" is missing.");
        is_good_file_name = false;
    }
    else {
        spParams["pc"]    >> hash_info.pc;
        if (hash_info.pc.empty()) {
            LOG4CXX_WARN(logger_, "The \"pc\" matrix in file \"" << filename << "\" is empty.");
            is_good_file_name = false;
        }
    }
    spParams.release();

    return is_good_file_name;
}


MPFDetectionError 
CaffeDetection::initHashInfoList(const std::vector<cv::String> &names_to_search,
                                 const std::string &model_name,
                                 std::string &hash_file_list,
                                 std::vector<std::string> &good_hash_layer_names,
                                 std::vector<std::string> &bad_hash_file_names) {

    LOG4CXX_DEBUG(logger_, "Loading spectral hash parameters");
    if (!hash_file_list.empty()) {
        boost::trim(hash_file_list);
        std::vector<std::string> files;
        boost::split(files, hash_file_list,
                     boost::is_any_of(" ;"),
                     boost::token_compress_on);

        for (const std::string &filename : files) {
            LOG4CXX_DEBUG(logger_, "filename = " << filename);
            SpectralHashInfo hash_info;
            try{
                cv::FileStorage spParams(filename, cv::FileStorage::READ);
                if (!spParams.isOpened()) {
                    LOG4CXX_WARN(logger_, "Failed to open spectral hash file named \"" << filename << "\"");
                    bad_hash_file_names.push_back(filename);
                }
                else {
                    if (spParams["layer_name"].empty()) {
                        LOG4CXX_WARN(logger_, "The \"layer_name\" field in file \"" << filename << "\" is missing.");
                        bad_hash_file_names.push_back(filename);
                    }
                    else {
                        spParams["layer_name"] >> hash_info.layer_name;
                        LOG4CXX_DEBUG(logger_, "layer_name = " << hash_info.layer_name);
                        if (std::find(names_to_search.begin(),
                                      names_to_search.end(),
                                      hash_info.layer_name) != names_to_search.end()) {
                            bool is_good_file = parseAndValidateHashInfo(filename,
                                                                         spParams,
                                                                         hash_info);
                            if (is_good_file) {
                                // Everything checks out ok, so save the
                                // hash info and the layer name.
                                hashInfoList_.push_back(hash_info);
                                good_hash_layer_names.push_back(hash_info.layer_name);
                            }
                            else {
                                bad_hash_file_names.push_back(filename);
                            }
                        }
                        else {
                            LOG4CXX_WARN(logger_, "Layer named \"" << hash_info.layer_name << "\" from spectral hash file \"" << filename << "\" was not found in the model named \"" << model_name << "\"");
                            bad_hash_file_names.push_back(filename);
                        }
                    }
                }
            } catch(const cv::Exception &err){
                LOG4CXX_WARN(logger_, "Exception caught when processing spectral hash file named \"" << filename << "\": " << err.what());
                bad_hash_file_names.push_back(filename);
            }
        }
    }
    return MPF_DETECTION_SUCCESS;
}


//-----------------------------------------------------------------------------
CaffeDetection::CaffeDetection() {

}

//-----------------------------------------------------------------------------
CaffeDetection::~CaffeDetection() {

}

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

//-----------------------------------------------------------------------------
MPFDetectionError CaffeDetection::GetDetections(const MPFImageJob &job, std::vector<MPFImageLocation> &locations) {

    try {
        LOG4CXX_DEBUG(logger_, "Data URI = " << job.data_uri);

        if (job.data_uri.empty()) {
            LOG4CXX_ERROR(logger_, "Invalid image file");
            return MPF_INVALID_DATAFILE_URI;
        }

        std::string model_name = DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "MODEL_NAME", "googlenet");

        std::map<std::string, ModelFiles>::iterator iter = model_defs_.find(model_name);
        if (iter == model_defs_.end()) {
            LOG4CXX_ERROR(logger_, "Could not load specified model: " << model_name);
            return MPF_DETECTION_NOT_INITIALIZED;
        }
        LOG4CXX_INFO(logger_, "Get detections using model: " << model_name);

        ModelFiles model_files = model_defs_.find(model_name)->second;
        synset_file_ = model_files.synset_file;

        std::vector<std::string> class_names;
        MPFDetectionError rc = readClassNames(class_names);
        if (rc != MPF_DETECTION_SUCCESS) {
            LOG4CXX_ERROR(logger_, "Failed to read class labels for the network");
            return rc;
        }
        if (class_names.size() <= 0) {
            LOG4CXX_ERROR(logger_, "No network class labels found");
            return MPF_DETECTION_FAILED;
        }

        // try to import Caffe model
        cv::dnn::Net net = cv::dnn::readNetFromCaffe(model_files.model_txt, model_files.model_bin);

        if (net.empty()) {
            LOG4CXX_ERROR(logger_, "Can't load network specified by the following files: ");
            LOG4CXX_ERROR(logger_, "prototxt:   " << model_files.model_txt);
            LOG4CXX_ERROR(logger_, "caffemodel: " << model_files.model_bin);
            return MPF_DETECTION_NOT_INITIALIZED;
        }

        LOG4CXX_DEBUG(logger_, "Created neural network");

        MPFImageReader image_reader(job);
        cv::Mat img = image_reader.GetImage();

        if (img.empty()) {
            LOG4CXX_ERROR(logger_, "Could not read image file: " << job.data_uri);
            return MPF_IMAGE_READ_ERROR;
        }

        std::vector< std::pair<int,float> > class_info;
        std::vector< std::pair<std::string,std::string> > activation_info;

        std::vector< std::pair<std::string,std::string> > hash_info;
        rc = GetDetections(job, net, img, class_info, activation_info, hash_info);
        if (rc != MPF_DETECTION_SUCCESS) {
            return rc;
        }

        Properties det_prop;
        MPFImageLocation detection(0,0,0,0);
        if (!class_info.empty()) {

            // Save the highest confidence classification as the
            // "CLASSIFICATION" property, and its corresponding confidence
            // as the MPFImageLocation confidence.
            LOG4CXX_DEBUG(logger_, "class id #0: " << class_info[0].first);
            LOG4CXX_DEBUG(logger_, "confidence: " << class_info[0].second);
            detection.confidence = class_info[0].second;
            det_prop["CLASSIFICATION"] = class_names.at(class_info[0].first);

            // Begin accumulating the classifications in a stringstream
            // for the "CLASSIFICATION LIST"
            std::stringstream ss_ids;
            ss_ids << class_names.at(class_info[0].first);

            // Use another stringstream for the
            // "CLASSIFICATION CONFIDENCE LIST"
            std::stringstream ss_conf;
            ss_conf << class_info[0].second;

            for (int i = 1; i < class_info.size(); i++) {
                LOG4CXX_DEBUG(logger_, "class id #" << i << ": " << class_info[i].first);
                LOG4CXX_DEBUG(logger_, "confidence: " << class_info[i].second);
                ss_ids << "; " << class_names.at(class_info[i].first);
                ss_conf << "; " << class_info[i].second;
            }
            det_prop["CLASSIFICATION LIST"] = ss_ids.str();
            det_prop["CLASSIFICATION CONFIDENCE LIST"] = ss_conf.str();
        }
        // Now put any activation layers in the detection properties
        if (!activation_info.empty()) {
            for (const std::pair<std::string,std::string> &activation : activation_info) {
                det_prop[activation.first] = activation.second;
            }
        }

        // Add spectral hash values to the detection properties
        if (!hash_info.empty()) {
            for (const std::pair<std::string,std::string> &hash : hash_info) {
                det_prop[hash.first] = hash.second;
            }
        }

        if (!class_info.empty() || !activation_info.empty() || !hash_info.empty()) {
            detection.detection_properties = det_prop;
            locations.push_back(detection);
        }
        

        return MPF_DETECTION_SUCCESS;
    }
    catch (...) {
        return Utils::HandleDetectionException(job, logger_);
    }
}

//-----------------------------------------------------------------------------
MPFDetectionError 
CaffeDetection::GetDetections(const MPFJob &job,
                              cv::dnn::Net &net,
                              cv::Mat &frame,
                              std::vector< std::pair<int,float> > &classes,
                              std::vector< std::pair<std::string, std::string> > &activation_layers,
                              std::vector< std::pair<std::string, std::string> > &spectral_hash_values) {

    LOG4CXX_DEBUG(logger_, "original frame mat rows = " << frame.rows << " cols = " << frame.cols);

    int resize_width = DetectionComponentUtils::GetProperty<int>(job.job_properties, "RESIZE_WIDTH", 224);
    int resize_height = DetectionComponentUtils::GetProperty<int>(job.job_properties, "RESIZE_HEIGHT", 224);

    cv::resize(frame, frame, cv::Size(resize_width, resize_height));
    LOG4CXX_DEBUG(logger_, "resized frame mat rows = " << frame.rows << " cols = " << frame.cols);

    int left_and_right_crop = DetectionComponentUtils::GetProperty<int>(job.job_properties, "LEFT_AND_RIGHT_CROP", 0);
    int top_and_bottom_crop = DetectionComponentUtils::GetProperty<int>(job.job_properties, "TOP_AND_BOTTOM_CROP", 0);

    if (left_and_right_crop > 0 || top_and_bottom_crop > 0) {
        cv::Rect roi(left_and_right_crop, top_and_bottom_crop,
                     frame.cols - (2 * left_and_right_crop), frame.rows - (2 * top_and_bottom_crop));
        frame = frame(roi);
        LOG4CXX_DEBUG(logger_, "cropped frame mat rows = " << frame.rows << " cols = " << frame.cols);
    }

    float sub_blue = DetectionComponentUtils::GetProperty<float>(job.job_properties, "SUBTRACT_BLUE_VALUE", 0);
    float sub_green = DetectionComponentUtils::GetProperty<float>(job.job_properties, "SUBTRACT_GREEN_VALUE", 0);
    float sub_red = DetectionComponentUtils::GetProperty<float>(job.job_properties, "SUBTRACT_RED_VALUE", 0);

    cv::Scalar sub_colors(sub_blue, sub_green, sub_red); // BGR

    // convert Mat to batch of images (BGR)
    cv::Mat input_blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(), sub_colors, false); // swapRB = false

    net.setInput(input_blob, "data"); // set the network input

    std::string output_layer_name = DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "MODEL_OUTPUT_LAYER", "prob");
    LOG4CXX_DEBUG(logger_, "Compute and gather output of layer named \"" << output_layer_name << "\"");

    // See if we need to output any internal activation layers. If so
    // add them to the list of layers passed to the forward() method.

    std::string act_layers_list = DetectionComponentUtils::GetProperty<std::string>(job.job_properties,
                                                                           "ACTIVATION_LAYER_LIST",
                                                                           "");

    std::vector<cv::String> output_layers;

    // Get the layers in the net and check that each layer
    // requested is actually part of the net. If it is, add it to the
    // vector of layer names for which we need the layer output. If
    // not, remember the name so that we can indicate in the output
    // that it was not found.
    std::vector<cv::String> net_layers = net.getLayerNames();
    std::string model_name = DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "MODEL_NAME", "googlenet");

    std::vector<std::string> good_act_layer_names;
    std::vector<std::string> bad_act_layer_names;

    getLayerNameLists(net_layers,
                      model_name,
                      act_layers_list,
                      good_act_layer_names,
                      bad_act_layer_names);
    for (std::string name : good_act_layer_names) {
        output_layers.push_back(cv::String(name));
    }

    // See if we need to compute the spectral hash of any internal
    // activation layers. If so add their names to the list of layers
    // passed to the forward() method.
    std::string hash_file_list = DetectionComponentUtils::GetProperty<std::string>(job.job_properties,
                                                                           "SPECTRAL_HASH_FILE_LIST",
                                                                           "");
    std::vector<std::string> good_hash_layer_names;
    std::vector<std::string> bad_hash_file_names;

    MPFDetectionError rc = initHashInfoList(net_layers, model_name,
                                            hash_file_list,
                                            good_hash_layer_names,
                                            bad_hash_file_names);
    if (rc != MPF_DETECTION_SUCCESS) return rc;
    for (std::string name : good_hash_layer_names) {
        output_layers.push_back(cv::String(name));
    }

    output_layers.push_back(cv::String(output_layer_name));
    LOG4CXX_DEBUG(logger_, "number of output layers = " << output_layers.size());

    // This form of the forward() function retrieves the output from
    // each of the layers in the name list. The name of the
    // classification output layer is the last in the list (or the
    // only one, if no activation layers or hash layers were requested).
    std::vector<cv::Mat> out_mats;
    net.forward(out_mats, output_layers); // compute output
    assert(out_mats.size() == output_layers.size());
    cv::Mat prob = out_mats.back();
    out_mats.pop_back();

    LOG4CXX_DEBUG(logger_, "output prob mat rows = " << prob.rows << " cols = " << prob.cols);
    LOG4CXX_DEBUG(logger_, "output prob mat total: " << prob.total());
    LOG4CXX_DEBUG(logger_, "number of output mats = " << out_mats.size());

    int num_classes = DetectionComponentUtils::GetProperty<int>(job.job_properties, "NUMBER_OF_CLASSIFICATIONS", 1);

    // The number of classifications requested must be greater
    // than 0 and less than the total size of the output blob.
    if ((num_classes <= 0) || (num_classes > prob.total())) {
        LOG4CXX_ERROR(logger_, "Number of classifications requested: "
                << num_classes
                << " is invalid. It must be greater than 0, and less than the total returned by the net output layer = "
                << prob.total());
        return MPF_INVALID_PROPERTY;
    }

    double threshold = DetectionComponentUtils::GetProperty<double>(job.job_properties, "CONFIDENCE_THRESHOLD", 0.0);

    // The threshold must be greater than or equal to 0.0.
    if (threshold < 0.0) {
        LOG4CXX_ERROR(logger_, "The confidence threshold requested: "
                << threshold
                << " is invalid. It must be greater than 0.0.");
        return MPF_INVALID_PROPERTY;
    }

    getTopNClasses(prob, num_classes, threshold, classes);

    // Compute the spectral hash values
    if (!out_mats.empty()) {
        for (int i = 0; i < hashInfoList_.size(); i++) {
            spectral_hash_values.push_back(computeSpectralHash(out_mats.back(), hashInfoList_.back()));
            out_mats.pop_back();
            hashInfoList_.pop_back();
        }
        assert(hashInfoList_.size() == 0);
    }

    // Notify the user if any of the spectral hash input files could
    // not be read or contained invalid information

    if (!bad_hash_file_names.empty()) {
        std::string prop_val = boost::algorithm::join(bad_hash_file_names, "; ");
        spectral_hash_values.push_back(
            std::make_pair("INVALID SPECTRAL HASH FILENAME LIST", prop_val));
    }

    // Create the activation layers output
    if (!good_act_layer_names.empty()) {
        getActivationLayers(out_mats, good_act_layer_names, activation_layers);
    }

    if (!bad_act_layer_names.empty()) {
        std::string prop_val = boost::algorithm::join(bad_act_layer_names, "; ");
        activation_layers.push_back(
            std::make_pair("INVALID ACTIVATION LAYER LIST", prop_val));
    }
 
    return MPF_DETECTION_SUCCESS;
}

//-----------------------------------------------------------------------------
void CaffeDetection::getTopNClasses(cv::Mat &prob_blob,
                                    int num_classes, double threshold,
                                    std::vector< std::pair<int, float> > &classes) {

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

//-----------------------------------------------------------------------------
MPFDetectionError CaffeDetection::readClassNames(std::vector<std::string> &class_names) {

    std::ifstream fp(synset_file_);
    if (!fp.is_open())
    {
        LOG4CXX_ERROR(logger_, "File with class labels not found: " << synset_file_);
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

//-----------------------------------------------------------------------------
void CaffeDetection::getActivationLayers(const std::vector<cv::Mat> &mats,
                                         const std::vector<std::string> &good_names,
                                         std::vector<std::pair<std::string,std::string> > &activations)
{
    for (int i = 0; i < good_names.size(); i++) {

        cv::Mat act = mats[i];
        std::string name = good_names[i];
        // Create a JSON-formatted string to represent the activation
        // values matrix.
        std::string filename = name + ".json";
        cv::FileStorage actStore(filename, cv::FileStorage::WRITE | cv::FileStorage::MEMORY);
        actStore << "activation values" << act;
        std::string actString = actStore.releaseAndGetString();
        std::transform(name.begin(), name.end(), name.begin(), ::toupper);
        name += " ACTIVATION MATRIX";
        activations.push_back(std::make_pair(name, actString));

    }
}

//-----------------------------------------------------------------------------
std::pair<std::string,std::string>
CaffeDetection::computeSpectralHash(const cv::Mat &activations,
                                    const SpectralHashInfo &hash_info) {

    cv::Mat omega0;
    cv::Mat omegas;
    omega0 = CV_PI / (hash_info.mx - hash_info.mn);
    omegas = cv::repeat(omega0, hash_info.modes.rows, 1);
    omegas = omegas.mul(hash_info.modes);

    cv::Mat x = cv::repeat( ((activations * hash_info.pc) - hash_info.mn) ,omegas.rows, 1).mul(omegas);

    std::string bitset;
    float* xPtr = (float*) x.data;
    for(int r=0;r<x.rows;r++){
        int bit=1;
        for(int c=0;c<x.cols;c++){
            bit *= ((cos(x.at<float>(r,c)) > 0) ? 1 : -1);
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
    return(std::make_pair(name, bitset));
}

//-----------------------------------------------------------------------------

MPF_COMPONENT_CREATOR(CaffeDetection);
MPF_COMPONENT_DELETER();

