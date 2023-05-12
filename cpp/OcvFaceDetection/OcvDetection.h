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


#ifndef OPENMPF_COMPONENTS_OCVDETECTION_H
#define OPENMPF_COMPONENTS_OCVDETECTION_H

#include <string>
#include <vector>
#include <utility>

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

class OcvDetection {
public:
    OcvDetection();
    virtual ~OcvDetection();

    std::vector<std::pair<cv::Rect, int>> DetectFaces(const cv::Mat &frame_gray, int min_face_size = 48);

    bool Init(std::string &run_directory);

private:
    std::string face_cascade_path;

    cv::CascadeClassifier face_cascade;

    bool initialized;

    log4cxx::LoggerPtr openFaceDetectionLogger;

    void GroupRectanglesMod(std::vector<cv::Rect>& rectList, int groupThreshold, double eps, std::vector<int>* weights, std::vector<double>* levelWeights);
};


#endif //OPENMPF_COMPONENTS_OCVDETECTION_H
