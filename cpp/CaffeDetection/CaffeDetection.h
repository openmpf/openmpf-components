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


#ifndef OPENMPF_COMPONENTS_CAFFEDETECTION_H
#define OPENMPF_COMPONENTS_CAFFEDETECTION_H

#include <log4cxx/logger.h>
#include <opencv2/dnn.hpp>

#include <adapters/MPFImageDetectionComponentAdapter.h>
#include <MPFDetectionComponent.h>


struct ModelFiles {
    cv::String model_txt;
    cv::String model_bin;
    std::string synset_file;
};

struct SpectralHashInfo {
    std::string model_name;
    std::string layer_name;
    int nbits;
    cv::Mat mx;
    cv::Mat mn;
    cv::Mat modes;
    cv::Mat pc;
};

class CaffeDetection : public MPF::COMPONENT::MPFImageDetectionComponentAdapter {
public:

    CaffeDetection();

    ~CaffeDetection();

    bool Init();

    bool Close();

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFImageJob &job,
            std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;

    std::string GetDetectionType();
private:

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFJob &job,
            cv::dnn::Net &net,
            cv::Mat &frame,
            std::vector< std::pair<int,float> > &classes,
            std::vector< std::pair<std::string, std::string> > &activation_layers,
            std::vector< std::pair<std::string, std::string> > &spectral_hash_values);

    void getTopNClasses(cv::Mat &prob_blob, int num_classes, double threshold,
                        std::vector< std::pair<int,float> > &classes);

    MPF::COMPONENT::MPFDetectionError readClassNames(std::vector<std::string> &class_names);

    void getLayerNameLists(const std::vector<cv::String> &names_to_search,
                           const std::string &model_name,
                           std::string &name_list,
                           std::vector<std::string> &good_names,
                           std::vector<std::string> &bad_names);

    // Return a vector of pairs of strings. The first string in each
    // pair contains the name of the activation layer, and the second
    // string contains a semi-colon separated list of activation
    // values.
    void getActivationLayers(const std::vector<cv::Mat> &mats,
                             const std::vector<std::string> &good_names,
                             std::vector<std::pair<std::string,std::string> > &activations);

    bool parseAndValidateHashInfo(const std::string filename,
                                  cv::FileStorage &spParams,
                                  SpectralHashInfo &hash_info);

    // Read the files containing information on spectral hash
    // computations. This list of file names is specified by the user
    // as a job property.
    MPF::COMPONENT::MPFDetectionError
    initHashInfoList(const std::vector<cv::String> &names_to_search,
                     const std::string &model_name,
                     std::string &hash_file_list,
                     std::vector<std::string> &good_names,
                     std::vector<std::string> &bad_file_names);

    // Computes the spectral hash for the activation values in a given
    // layer. Returns a pair containing the name of the layer, and a
    // string containing the spectral hash as a sequence of 1's and 0's.
    std::pair<std::string,std::string> computeSpectralHash(const cv::Mat &activations,
                                                           const SpectralHashInfo &hash_data);

    std::string synset_file_;
    std::map<std::string, ModelFiles> model_defs_;
    std::vector<SpectralHashInfo> hashInfoList_;

    log4cxx::LoggerPtr logger_;
};


#endif //OPENMPF_COMPONENTS_CAFFEDETECTION_H
