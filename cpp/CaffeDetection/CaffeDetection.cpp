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

#include <Utils.h>
#include <detectionComponentUtils.h>
#include <MPFImageReader.h>

#include "ModelFileParser.h"


using CONFIG4CPP_NAMESPACE::StringVector;
using namespace MPF::COMPONENT;

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
        std::string output_layer_name = DetectionComponentUtils::GetProperty<std::string>(job.job_properties, "MODEL_OUTPUT_LAYER", "prob");

        std::map<std::string, ModelFiles>::iterator iter = model_defs_.find(model_name);
        if (iter == model_defs_.end()) {
            LOG4CXX_ERROR(logger_, "Could not load specified model: " << model_name);
            return MPF_DETECTION_NOT_INITIALIZED;
        }
        LOG4CXX_INFO(logger_, "Get detections using model: " << model_name);

        ModelFiles model_files = model_defs_.find(model_name)->second;
        synset_file_ = model_files.synset_file;

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

        LOG4CXX_DEBUG(logger_, "original img mat rows = " << img.rows << " cols = " << img.cols);

        int resize_width = DetectionComponentUtils::GetProperty<int>(job.job_properties, "RESIZE_WIDTH", 224);
        int resize_height = DetectionComponentUtils::GetProperty<int>(job.job_properties, "RESIZE_HEIGHT", 224);

        cv::resize(img, img, cv::Size(resize_width, resize_height));
        LOG4CXX_DEBUG(logger_, "resized img mat rows = " << img.rows << " cols = " << img.cols);

        int left_and_right_crop = DetectionComponentUtils::GetProperty<int>(job.job_properties, "LEFT_AND_RIGHT_CROP", 0);
        int top_and_bottom_crop = DetectionComponentUtils::GetProperty<int>(job.job_properties, "TOP_AND_BOTTOM_CROP", 0);

        if (left_and_right_crop > 0 || top_and_bottom_crop > 0) {
            cv::Rect roi(left_and_right_crop, top_and_bottom_crop,
                         img.cols - (2 * left_and_right_crop), img.rows - (2 * top_and_bottom_crop));
            img = img(roi);
            LOG4CXX_DEBUG(logger_, "cropped img mat rows = " << img.rows << " cols = " << img.cols);
        }

        bool transpose = DetectionComponentUtils::GetProperty<bool>(job.job_properties, "TRANSPOSE", false);

        if (transpose) {
            cv::Mat transposed =  cv::Mat(img.cols, img.rows, img.type());
            cv::transpose(img, transposed);
            img = transposed;
        }

        int sub_blue = DetectionComponentUtils::GetProperty<int>(job.job_properties, "SUBTRACT_BLUE_VALUE", 0);
        int sub_green = DetectionComponentUtils::GetProperty<int>(job.job_properties, "SUBTRACT_GREEN_VALUE", 0);
        int sub_red = DetectionComponentUtils::GetProperty<int>(job.job_properties, "SUBTRACT_RED_VALUE", 0);

        if (sub_blue != 0 || sub_green != 0 || sub_red != 0) {
            cv::Mat sub_colors(img.cols, img.rows, img.type(), cv::Scalar(sub_blue, sub_green, sub_red)); // BGR
            img = img - sub_colors;
        }

        // convert Mat to batch of images
        cv::Mat input_blob = cv::dnn::blobFromImage(img, 1.0, false); // swapRB = false

        net.setBlob(".data", input_blob); // set the network input

        net.forward(); // compute output

        LOG4CXX_DEBUG(logger_, "Gather output of layer named \"" << output_layer_name << "\"");

        // gather output of last layer
        cv::Mat prob = net.getBlob(output_layer_name);

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

        LOG4CXX_DEBUG(logger_, "output prob mat rows = " << prob.rows << " cols = " << prob.cols);
        LOG4CXX_DEBUG(logger_, "output prob mat total: " << prob.total());

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

        std::vector< std::pair<int,float> > class_info;
        getTopNClasses(prob, num_classes, threshold, class_info);

        if (!class_info.empty()) {
            Properties det_prop;
            MPFImageLocation detection(0,0,0,0);

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

MPF_COMPONENT_CREATOR(CaffeDetection);
MPF_COMPONENT_DELETER();

