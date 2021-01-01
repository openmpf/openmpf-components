/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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


#include <future>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <log4cxx/logger.h>

#include <BlockingQueue.h>
#include <MPFDetectionComponent.h>
#include <MPFDetectionObjects.h>

#include "../include/DarknetInterface.h"
#include "darknet.h"


namespace DarknetHelpers {
    using network_ptr_t = std::unique_ptr<network, void(*)(network*)>;

    struct DarknetImageHolder;
}


template<typename ClassFilter>
class DarknetImpl : public DarknetInterface {

public:
    DarknetImpl(const std::string &job_name, const MPF::COMPONENT::Properties &props,
                const ModelSettings &settings, log4cxx::LoggerPtr &logger);

    std::vector<DarknetResult> Detect(int frame_number, const cv::Mat &cv_image) override;

    void Detect(int frame_number, const cv::Mat &cv_image, std::vector<DarknetResult> &detections) override;

    void Detect(const DarknetHelpers::DarknetImageHolder &image_holder, std::vector<DarknetResult> &detections);

    cv::Size GetTargetFrameSize();

private:
    std::string log_prefix_;
    log4cxx::LoggerPtr logger_;
    DarknetHelpers::network_ptr_t network_;
    int output_layer_size_;
    int num_classes_;
    std::vector<std::string> names_;
    ClassFilter class_filter_;
    float confidence_threshold_;
};



class DarknetAsyncImpl : public DarknetAsyncInterface {
public:
    DarknetAsyncImpl(const std::string &job_name, const MPF::COMPONENT::Properties &props,
                     const ModelSettings &settings, log4cxx::LoggerPtr &logger);

    ~DarknetAsyncImpl() override;

    void Submit(int frame_number, const cv::Mat &cv_image) override;

    std::vector<DarknetResult> GetResults() override;

private:
    std::string log_prefix_;

    log4cxx::LoggerPtr logger_;

    using DarknetQueue = MPF::COMPONENT::BlockingQueue<std::unique_ptr<DarknetHelpers::DarknetImageHolder>>;
    DarknetQueue work_queue_;

    cv::Size target_frame_size_;

    std::future<std::vector<DarknetResult>> work_done_future_;

    bool get_results_called_ = false;

    template<typename ClassFilter>
    void Init(const std::string &job_name, const MPF::COMPONENT::Properties &props, const ModelSettings &settings);


    // Runs on a thread spawned by the call to std::async in the Init method.
    template<typename ClassFilter>
    static std::vector<DarknetResult> ProcessFrameQueue(DarknetImpl<ClassFilter> darknet_impl,
                                                        DarknetQueue &work_queue);
};


#endif //OPENMPF_COMPONENTS_DARKNETIMPL_H
