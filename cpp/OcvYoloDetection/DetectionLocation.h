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

#ifndef OPENMPF_COMPONENTS_DETECTIONLOCATION_H
#define OPENMPF_COMPONENTS_DETECTIONLOCATION_H

#include <log4cxx/logger.h>
#include <opencv2/core.hpp>

#include "Config.h"
#include "Frame.h"


// TODO Should something be done about circular dependency between DetectionLocation and Track?
class Track;


class DetectionLocation : public MPF::COMPONENT::MPFImageLocation {

public:
    /// frame associated with detection
    Frame frame;

    DetectionLocation(const Config &config,
                      Frame frame,
                      const cv::Rect2d &boundingBox,
                      float confidence,
                      cv::Mat classFeature,
                      cv::Mat dftFeature);

    /// get unit vector of scores
    cv::Mat getClassFeature() const;

    /// get dft for phase correlation
    // TODO Determine if this can be made const
    cv::Mat getDFTFeature();

    /// get location as an opencv rectangle
    cv::Rect2i getRect() const;

    /// set location from an opencv rectangle
    void setRect(const cv::Rect2i &rect);

    /// 1 - compute intersection over union
    float iouDist(const Track &track) const;

    /// compute normalized center to center distance
    float center2CenterDist(const Track &track) const;

    /// compute deep feature similarity distance
    // TODO: Determine if we can make this "float featureDist(const Track &track) const"
    float featureDist(Track &track);

    /// compute Kalman filter residual distance
    float kfResidualDist(const Track &track) const;

    /// get bbox alignment via phase correlation
    cv::Point2d phaseCorrelate(Track &tr);

private:
    int dftSize_;
    bool dftHanningWindowEnabled_;
    float edgeSnapDist_;

    /// unit vector of with elements proportional to scores for each classes
    cv::Mat classFeature_;

    /// dft for matching-up detections via phase correlation
    cv::Mat dftFeature_;

    /// get Hanning window of specified size
    cv::Mat1f getHanningWindow(const cv::Size &size) const;
};


#endif //OPENMPF_COMPONENTS_DETECTIONLOCATION_H
