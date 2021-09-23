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


#ifndef OPENMPF_COMPONENTS_OCVYOLODETECTION_H
#define OPENMPF_COMPONENTS_OCVYOLODETECTION_H

#include <memory>
#include <list>
#include <string>
#include <vector>

#include <log4cxx/logger.h>

#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>
#include <ModelsIniParser.h>

#include "Config.h"
#include "yolo_network/YoloNetwork.h"


class OcvYoloDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:
    bool Init() override;

    bool Close() override;

    std::string GetDetectionType() override;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job) override;

    std::vector<MPF::COMPONENT::MPFImageLocation> GetDetections(
            const MPF::COMPONENT::MPFImageJob &job) override;

private:
    log4cxx::LoggerPtr logger_;

    MPF::COMPONENT::ModelsIniParser<ModelSettings> modelsParser_;

    std::unique_ptr<YoloNetwork> cachedYoloNetwork_;

    YoloNetwork& GetYoloNetwork(const MPF::COMPONENT::Properties &jobProperties,
                                const Config &config);
};

#endif //OPENMPF_COMPONENTS_OCVYOLODETECTION_H
