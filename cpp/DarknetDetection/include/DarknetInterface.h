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


#ifndef OPENMPF_COMPONENTS_DARKNETINTERFACE_H
#define OPENMPF_COMPONENTS_DARKNETINTERFACE_H

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>



struct DarknetResult {
    int frame_number;
    cv::Rect detection_rect;
    std::vector<std::pair<float, std::string>> object_type_probs;

    DarknetResult(
            int frame_number,
            const cv::Rect &detection_rect,
            std::vector<std::pair<float, std::string>> &&object_type_probs = {})
        : frame_number(frame_number)
        , detection_rect(detection_rect)
        , object_type_probs(std::move(object_type_probs))
    {
    }
};

struct ModelSettings {
    std::string network_config_file;
    std::string names_file;
    std::string weights_file;
};


class DarknetInterface {
public:
    DarknetInterface(const std::map<std::string, std::string> &props, const ModelSettings &settings) { }

    virtual ~DarknetInterface() = default;

    virtual std::vector<DarknetResult> Detect(int frame_number, const cv::Mat &cv_image) = 0;

    virtual void Detect(int frame_number, const cv::Mat &cv_image, std::vector<DarknetResult> &detections) = 0;
};



class DarknetAsyncInterface {
public:
    DarknetAsyncInterface(const std::map<std::string, std::string> &props, const ModelSettings &settings) { }

    virtual ~DarknetAsyncInterface() = default;

    virtual void Submit(int frame_number, const cv::Mat &cv_image) = 0;

    virtual std::vector<DarknetResult> GetResults() = 0;
};


#endif //OPENMPF_COMPONENTS_DARKNETINTERFACE_H
