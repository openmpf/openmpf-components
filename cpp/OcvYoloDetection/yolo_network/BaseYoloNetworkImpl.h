/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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

#ifndef OPENMPF_COMPONENTS_BASEYOLONETWORKIMPL_H
#define OPENMPF_COMPONENTS_BASEYOLONETWORKIMPL_H

#include <functional>
#include <string>
#include <vector>

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>

#include "../Config.h"
#include "../DetectionLocation.h"
#include "../Frame.h"

#include "YoloNetwork.h"

class BaseYoloNetworkImpl {
public:
    BaseYoloNetworkImpl(ModelSettings modelSettings, const Config &config);

    virtual ~BaseYoloNetworkImpl();

    using ProcessFrameDetectionsCallback =
    std::function<void(std::vector<std::vector<DetectionLocation>> &&dets,
                       std::vector<Frame>::const_iterator begin,
                       std::vector<Frame>::const_iterator end)>;

    virtual void GetDetections(
            std::vector<Frame> &frames,
            const ProcessFrameDetectionsCallback &processFrameDetectionsCallback,
            const Config &config);

    virtual bool IsCompatible(const ModelSettings &modelSettings, const Config &config) const;

    virtual void Finish();

    virtual void Reset() noexcept;

protected:
    log4cxx::LoggerPtr log_ = log4cxx::Logger::getLogger("OcvYoloDetection");

    ModelSettings modelSettings_;
    int cudaDeviceId_;
    cv::dnn::Net net_;

    std::vector<std::string> names_;
    cv::Mat1f confusionMatrix_;
    std::string classAllowListPath_;
    std::function<bool(const std::string &)> classFilter_;

    std::vector<std::vector<DetectionLocation>> GetDetectionsCvdnn(
            const std::vector<Frame> &frames, const Config &config);

    std::vector<DetectionLocation> ExtractFrameDetectionsCvdnn(
            int frameIdx, const Frame &frame, const std::vector<cv::Mat> &layerOutputs,
            const Config &config) const;

    DetectionLocation CreateDetectionLocationCvdnn(
            const Frame &frame,
            const cv::Rect2d &boundingBox,
            const cv::Mat1f &scores,
            const Config &config) const;
};

#endif // OPENMPF_COMPONENTS_BASEYOLONETWORKIMPL_H
