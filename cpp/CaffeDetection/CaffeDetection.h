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
            std::vector< std::pair<std::string, std::string> > &activation_layers);

    void getTopNClasses(cv::Mat &prob_blob, int num_classes, double threshold,
                        std::vector< std::pair<int,float> > &classes);

    MPF::COMPONENT::MPFDetectionError readClassNames(std::vector<std::string> &class_names);

    // Return a vector of pairs of strings. The first string in each
    // pair contains the name of the activation layer, and the second
    // string contains a semi-colon separated list of activation
    // values.
    void getActivationLayers(const std::vector<cv::Mat> &mats,
                             const std::vector<std::string> &good_names,
                             const std::vector<std::string> &bad_names,
                             std::vector<std::pair<std::string,std::string> > &activations);

    std::string synset_file_;
    std::map<std::string, ModelFiles> model_defs_;
    log4cxx::LoggerPtr logger_;
};


#endif //OPENMPF_COMPONENTS_CAFFEDETECTION_H
