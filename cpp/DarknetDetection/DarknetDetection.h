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
#include <ModelsIniParser.h>
#include <dlfcn.h>
#include <DlClassLoader.h>

#include "include/DarknetInterface.h"
#include "SPSCBoundedQueue.h"


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
    using DarknetDl = MPF::COMPONENT::DlClassLoader<DarknetInterface>;

    log4cxx::LoggerPtr logger_;
    
    std::string plugin_path_;

    MPF::COMPONENT::ModelsIniParser<ModelSettings> models_parser_;

    DarknetDl GetDarknetImpl(const MPF::COMPONENT::MPFJob &job);

    struct VideoFrame {
      int index;
      cv::Mat frame;

      VideoFrame() : index(0) {}
      VideoFrame(int i, cv::Mat &f) 
               : index(i), frame(std::move(f)) {}
    };

    template <typename Tracker>
    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
            Tracker &tracker);

  template<typename Tracker, typename Entry>
    void RunDetection(DarknetDl &detector, Tracker &tracker,
                      SPSCBoundedQueue<Entry> &queue);

    static void ConvertResultsUsingPreprocessor(std::vector<DarknetResult> &darknet_results,
                                                std::vector<MPF::COMPONENT::MPFImageLocation> &locations);

    ModelSettings GetModelSettings(const MPF::COMPONENT::Properties &job_properties) const;
};







#endif //OPENMPF_COMPONENTS_DARKNETDETECTION_H
