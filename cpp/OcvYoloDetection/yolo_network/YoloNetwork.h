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

#ifndef OPENMPF_COMPONENTS_YOLONETWORK_H
#define OPENMPF_COMPONENTS_YOLONETWORK_H

#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/dnn/dnn.hpp>

#include "../Config.h"
#include "../DetectionLocation.h"
#include "../Frame.h"

struct ModelSettings {
    std::string ocvDnnNetworkConfigFile;
    std::string ocvDnnWeightsFile;
    std::string namesFile;
    std::string confusionMatrixFile;
};

class YoloNetwork {
public:
    YoloNetwork(ModelSettings modelSettings, const Config &config);

    ~YoloNetwork();

    using ProcessFrameDetectionsCallback =
    std::function<void(std::vector<std::vector<DetectionLocation>> &&dets,
                       std::vector<Frame>::const_iterator begin,
                       std::vector<Frame>::const_iterator end)>;

    void GetDetections(
            std::vector<Frame> &frames,
            const ProcessFrameDetectionsCallback &processFrameDetectionsCallback,
            const Config &config);

    bool IsCompatible(const ModelSettings &modelSettings, const Config &config) const;

    void Finish();

    void Reset() noexcept;

private:
    class YoloNetworkImpl;

    std::unique_ptr<YoloNetworkImpl> pimpl_;
};

#endif // OPENMPF_COMPONENTS_YOLONETWORK_H
