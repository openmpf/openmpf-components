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

#ifndef OPENMPF_COMPONENTS_TESTUTILS_H
#define OPENMPF_COMPONENTS_TESTUTILS_H

#include <string>
#include <vector>

#include "Track.h"
#include "OcvYoloDetection.h"


/** ***************************************************************************
*   macros for "pretty" gtest messages
**************************************************************************** */
#define GTEST_BOX "[          ] "
#define GOUT(MSG){                                                            \
  std::cout << GTEST_BOX << MSG << std::endl;                                 \
}


bool init_logging();

OcvYoloDetection initComponent();

template<typename T>
T findDetectionWithClass(const std::string &classification,
                         const std::vector<T> &detections) {
    for (const auto &detection: detections) {
        if (classification == detection.detection_properties.at("CLASSIFICATION")) {
            return detection;
        }
    }

    std::string message = "No detection with class: " + classification;
    throw std::runtime_error(message);
}

float iou(const cv::Rect &r1, const cv::Rect &r2);

float iou(const MPF::COMPONENT::MPFImageLocation &l1,
          const MPF::COMPONENT::MPFImageLocation &l2);

MPF::COMPONENT::Properties getTinyYoloConfig(float confidenceThreshold = 0.5);

MPF::COMPONENT::Properties getYoloConfig(float confidenceThreshold = 0.5);

MPF::COMPONENT::Properties getTritonYoloConfig(const std::string &tritonServer,
                                               float confidenceThreshold = 0.5);

bool same(MPF::COMPONENT::MPFImageLocation &l1,
          MPF::COMPONENT::MPFImageLocation &l2,
          float confidenceTolerance,
          float iouTolerance,
          float &confidenceDiff,
          float &iouValue);

bool same(MPF::COMPONENT::MPFImageLocation &l1,
          MPF::COMPONENT::MPFImageLocation &l2,
          float confidenceTolerance = 0.01,
          float iouTolerance = 0.1);

bool same(MPF::COMPONENT::MPFVideoTrack &t1,
          MPF::COMPONENT::MPFVideoTrack &t2,
          float confidenceTolerance,
          float iouTolerance,
          float &confidenceDiff,
          float &aveIou);

bool same(MPF::COMPONENT::MPFVideoTrack &t1,
          MPF::COMPONENT::MPFVideoTrack &t2,
          float confidenceTolerance = 0.01,
          float iouTolerance = 0.1);

void write_track_output(std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
                        const std::string& outTrackFileName,
                        MPF::COMPONENT::MPFVideoJob &videoJob);

std::vector<MPF::COMPONENT::MPFVideoTrack> read_track_output(const std::string& inTrackFileName);

void write_track_output_video(std::string inVideoFileName,
                              std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks,
                              const std::string& outVideoFileName,
                              MPF::COMPONENT::MPFVideoJob &videoJob);

bool objectFound(const std::string &expectedObjectName,
                 const MPF::COMPONENT::Properties &detectionProperties);

bool objectFound(const std::string &expectedObjectName,
                 int frameNumber,
                 const std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);

bool objectFound(const std::string &expected_object_name,
                 const std::vector<MPF::COMPONENT::MPFImageLocation> &detections);

#endif // OPENMPF_COMPONENTS_TESTUTILS_H
