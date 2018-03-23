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


#ifndef OPENMPF_COMPONENTS_DARKNETDETECTION_H
#define OPENMPF_COMPONENTS_DARKNETDETECTION_H

#include <memory>

#include <log4cxx/logger.h>
#include <opencv2/core.hpp>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>

#include "Trackers.h"
#include "ModelSettings.h"
#include "ModelsIniParser.h"

extern "C" {
    #include "darknet.h"
};


class DarknetDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:
    bool Init() override;

    bool Close() override;

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks) override;

    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFImageJob &job,
            std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;


    std::string GetDetectionType() override;


private:
    log4cxx::LoggerPtr logger_;

    ModelsIniParser<ModelSettings> models_parser_;


    template <typename Tracker>
    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
            Tracker &tracker);

    static void ConvertResultsUsingPreprocessor(std::vector<DarknetResult> &darknet_results,
                                                std::vector<MPF::COMPONENT::MPFImageLocation> &locations);

    ModelSettings GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const;

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



    class Detector {
    public:

        explicit Detector(const MPF::COMPONENT::Properties &props, const ModelSettings &settings);

        std::vector<DarknetResult> Detect(const cv::Mat &cv_image);


    private:
        using network_ptr_t = std::unique_ptr<network, decltype(&free_network)>;

        network_ptr_t network_;
        int output_layer_size_;
        int num_classes_;
        std::vector<std::string> names_;

        float confidence_threshold_;

        ProbHolder probs_;

        std::unique_ptr<box[]> boxes_;


        static network_ptr_t LoadNetwork(const ModelSettings &model_settings);

        static int GetOutputLayerSize(const network &net);

        static int GetNumClasses(const network &net);

        static std::vector<std::string> LoadNames(const ModelSettings &model_settings, int expected_name_count);

        // Darknet functions that accept C style strings, accept a char* instead of a const char*.
        static std::unique_ptr<char[]> ToNonConstCStr(const std::string &str);

        static cv::Rect BoxToRect(const box &box, const cv::Size &image_size);
    };


    // Darknet uses its own image type, which is a C struct. This class adds two features to the Darknet image type.
    // One is that it adds a destructor, which calls the free_image function defined in the Darknet library.
    // The second feature is that it converts a cv::Mat to the Darknet image format.
    struct DarknetImageHolder {
        image darknet_image;

        explicit DarknetImageHolder(const cv::Mat &cv_image);

        ~DarknetImageHolder();

        DarknetImageHolder(const DarknetImageHolder&) = delete;
        DarknetImageHolder& operator=(const DarknetImageHolder&) = delete;
        DarknetImageHolder(DarknetImageHolder&&) = delete;
        DarknetImageHolder& operator=(DarknetImageHolder&&) = delete;
    };


};







#endif //OPENMPF_COMPONENTS_DARKNETDETECTION_H
