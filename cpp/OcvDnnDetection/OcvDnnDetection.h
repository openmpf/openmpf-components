/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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


#ifndef OPENMPF_COMPONENTS_OCVDNNDETECTION_H
#define OPENMPF_COMPONENTS_OCVDNNDETECTION_H

#include <memory>

#include <log4cxx/logger.h>
#include <opencv2/dnn.hpp>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <ModelsIniParser.h>

struct SpectralHashInfo {
    std::string file_name;
    std::string model_name;
    std::string layer_name;
    int nbits;
    cv::Mat mx;
    cv::Mat mn;
    cv::Mat modes;
    cv::Mat pc;
};

class OcvDnnDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:

    struct ModelSettings {
        std::string model_config_file;
        std::string model_binary_file;
        std::string synset_file;
    };

    bool Init() override;

    bool Close() override;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(const MPF::COMPONENT::MPFImageJob &job) override;

    std::string GetDetectionType() override;


private:

    log4cxx::LoggerPtr logger_;

    MPF::COMPONENT::ModelsIniParser<ModelSettings> models_parser_;

    struct OcvDnnJobConfig;

    void getTopNClasses(cv::Mat &prob_blob, int num_classes, double threshold,
                        std::vector< std::pair<int,float> > &classes) const;



    static void addActivationLayerInfo(
            const OcvDnnJobConfig &config,
            const std::vector<std::pair<std::string, cv::Mat>> &activation_layer_mats,
            MPF::COMPONENT::Properties &detection_properties);


    void addSpectralHashInfo(OcvDnnJobConfig &config,
                             const std::vector<std::pair<SpectralHashInfo, cv::Mat>> &spectral_hash_mats,
                             MPF::COMPONENT::Properties &detection_properties) const;


    // Computes the spectral hash for the activation values in a given
    // layer. Returns a pair containing the name of the layer, and a
    // string containing the spectral hash as a sequence of 1's and 0's.
    std::pair<std::string, std::string> computeSpectralHash(const cv::Mat &activations,
                                                            const SpectralHashInfo &hash_info) const;


    // Sets the location parameter to a MPFImageLocation if a detection is found in the input frame.
    std::unique_ptr<MPF::COMPONENT::MPFImageLocation> getDetection(
            OcvDnnDetection::OcvDnnJobConfig &config, const cv::Mat &input_frame) const;

    template <typename Tracker>
    std::vector<MPF::COMPONENT::MPFVideoTrack> getDetections(
            const MPF::COMPONENT::MPFVideoJob &job, Tracker tracker) const;


    static void getNetworkOutput(
            OcvDnnJobConfig &config,
            const cv::Mat &input_frame,
            cv::Mat &output_layer,
            std::vector<std::pair<std::string, cv::Mat>> &activation_layer_info,
            std::vector<std::pair<SpectralHashInfo, cv::Mat>> &spectral_hash_info);


    // struct to hold configuration options and data structures that change every job.
    struct OcvDnnJobConfig {
    public:
        std::vector<std::string> class_names;
        cv::dnn::Net net;

        cv::Size resize_size;
        cv::Size crop_size;
        cv::Scalar subtract_colors;


        // In order to get all the layers we need in one pass through the network, we need to add all the layer
        // names to a single collection. After getting the output layers we need to know whether it was requested
        // in order to get the classification, to get the activation layers, or to compute the spectral hash.
        // In order to keep track of which layer was retrieved for which purpose output_layers will contain
        // the layer names in a specific order.
        // The first element is the name of the classification layer.
        // The next region will contain the activation layer names.
        // The final region will contain the names of the layers for which we need to compute the spectral hash.
        std::vector<cv::String> output_layers;

        std::string model_output_layer;
        std::string model_input_name;

        std::vector<std::string> requested_activation_layer_names;
        std::vector<std::string> bad_activation_layer_names;

        std::vector<SpectralHashInfo> spectral_hash_info;
        std::vector<std::string> bad_hash_file_names;


        int number_of_classifications;
        double confidence_threshold;

        OcvDnnJobConfig(const MPF::COMPONENT::Properties &props,
                       const MPF::COMPONENT::ModelsIniParser<ModelSettings> &model_parser,
                       const log4cxx::LoggerPtr &logger);

    private:
        static std::vector<std::string> readClassNames(const std::string &synset_file);

        void validateLayerNames(
                std::string requested_activation_layers,
                const std::vector<cv::String> &net_layers,
                const std::string &model_name,
                const log4cxx::LoggerPtr &logger);


        void getSpectralHashInfo(
                std::string hash_file_list,
                const std::vector<cv::String> &net_layers,
                const std::string &model_name,
                const log4cxx::LoggerPtr &logger);

        static bool parseAndValidateHashInfo(const std::string &file_name, cv::FileStorage &sp_params,
                                             SpectralHashInfo &hash_info, const log4cxx::LoggerPtr &logger);
    };
};


#endif //OPENMPF_COMPONENTS_OCVDNNDETECTION_H
