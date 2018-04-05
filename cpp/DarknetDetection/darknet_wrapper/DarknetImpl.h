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


#ifndef OPENMPF_COMPONENTS_DARKNETIMPL_H
#define OPENMPF_COMPONENTS_DARKNETIMPL_H


#include <memory>
#include <vector>
#include <opencv2/core.hpp>

#include <MPFDetectionComponent.h>

#include "../include/DarknetInterface.h"

#include "darknet.h"


namespace DarknetHelpers {
    using network_ptr_t = std::unique_ptr<network, decltype(&free_network)>;


    // Darknet's network_detect function has a float** parameter that is used to store the probabilities.
    // The float** is used as a two dimensional array. The first level index is the id of a box,
    // and the second level index is the id of a classification.
    class ProbHolder {
    public:
        ProbHolder(int output_layer_size, int num_classes);

        float** Get();

        float* operator[](size_t box_idx);

        void Clear();

    private:
        int mat_size_;

        std::unique_ptr<float[]> prob_mat_;

        std::unique_ptr<float*[]> prob_row_ptrs_;
    };
}


template<typename ClassFilter>
class DarknetImpl : public DarknetInterface {

public:
    DarknetImpl(const MPF::COMPONENT::Properties &props, const ModelSettings &settings);

    std::vector<DarknetResult> Detect(const cv::Mat &cv_image) override;


private:
    DarknetHelpers::network_ptr_t network_;
    int output_layer_size_;
    int num_classes_;
    std::vector<std::string> names_;
    ClassFilter class_filter_;

    float confidence_threshold_;

    DarknetHelpers::ProbHolder probs_;

    std::unique_ptr<box[]> boxes_;
};


#endif //OPENMPF_COMPONENTS_DARKNETIMPL_H
