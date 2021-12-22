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

#include "Frame.h"

using namespace MPF::COMPONENT;

cv::Mat Frame::getDataAsResizedFloat(
        const cv::Size2i targetSize,
        const int cvBorderType,
        const cv::Scalar &cvBorderValue) const {

    double targetAspect = targetSize.width / static_cast<double>(targetSize.height);
    double dataAspect = data.cols / static_cast<double>(data.rows);
    double scaleFactor;
    if (targetAspect > dataAspect) {
        // limited by target y
        scaleFactor = targetSize.height / static_cast<double>(data.rows);
    } else {
        // limited by target x
        scaleFactor = targetSize.width / static_cast<double>(data.cols);
    }

    cv::Mat resizedData;
    cv::resize(data, resizedData, cv::Size(), scaleFactor, scaleFactor);

    int leftPadding = (targetSize.width - resizedData.cols) / 2;
    int topPadding = (targetSize.height - resizedData.rows) / 2;

    // Convert rectangular image to square image by adding grey bars to the smaller dimensions.
    // Grey was chosen because that is what the Darknet library does.
    cv::copyMakeBorder(
            resizedData,
            resizedData,
            topPadding,
            targetSize.width - resizedData.rows - topPadding,
            leftPadding,
            targetSize.height - resizedData.cols - leftPadding,
            cvBorderType,
            cvBorderValue);
    assert(("Frame resize did not result in desired dimensions.",
            targetSize.width == resizedData.cols
            && targetSize.height == resizedData.rows));
    resizedData.convertTo(resizedData, CV_32F, 1 / 255.0);
    return resizedData;
}
