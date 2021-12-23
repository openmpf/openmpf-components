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

#include <utility>

#include "BaseYoloNetworkImpl.h"

using namespace MPF::COMPONENT;

class YoloNetwork::YoloNetworkImpl : public BaseYoloNetworkImpl {
public:
    using BaseYoloNetworkImpl::BaseYoloNetworkImpl; // use base constructor

    ~YoloNetworkImpl() = default;
};

YoloNetwork::YoloNetwork(ModelSettings model_settings, const Config &config)
        : pimpl_(new YoloNetworkImpl(std::move(model_settings), config)) {}

YoloNetwork::~YoloNetwork() = default;

void YoloNetwork::GetDetections(
        std::vector<Frame> &frames,
        const ProcessFrameDetectionsCallback &processFrameDetectionsFun,
        const Config &config) {
    pimpl_->GetDetections(frames, processFrameDetectionsFun, config);
}

bool YoloNetwork::IsCompatible(const ModelSettings &modelSettings, const Config &config) const {
    return pimpl_->IsCompatible(modelSettings, config);
}

void YoloNetwork::Finish() {
    return pimpl_->Finish();
}

void YoloNetwork::Reset() noexcept {
    return pimpl_->Reset();
}
