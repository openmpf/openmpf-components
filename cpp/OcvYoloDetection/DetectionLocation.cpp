/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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

#include "DetectionLocation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

#include "util.h"
#include "Track.h"
#include "ocv_phasecorr.h"

using namespace MPF::COMPONENT;


DetectionLocation::DetectionLocation(const Config &config,
                                     Frame frame,
                                     const cv::Rect2d &boundingBox,
                                     float confidence,
                                     cv::Mat classFeature,
                                     cv::Mat dftFeature)
        : frame(std::move(frame))
        , dftSize_(config.dftSize)
        , dftHanningWindowEnabled_(config.dftHannWindowEnabled)
        , edgeSnapDist_(config.edgeSnapDist)
        , classFeature_(std::move(classFeature))
        , dftFeature_(std::move(dftFeature)) {
    this->confidence = confidence;
    setRect(boundingBox & cv::Rect2d(0, 0, DetectionLocation::frame.data.cols,
                                     DetectionLocation::frame.data.rows));
}


/** **************************************************************************
* Compute (1 - Intersection Over Union) metric between a rectangle and detection
* comprised of 1 - the ratio of the area of the intersection of the detection
* rectangles divided by the area of the union of the detection rectangles.
*
* \param   track track to compute iou with
* \returns 1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::iouDist(const Track &track) const {
    cv::Rect2i detectionRect(x_left_upper, y_left_upper, width, height);
    cv::Rect2i predictedRect = track.predictedBox();

    int intersectionArea = (detectionRect & predictedRect).area();
    int unionArea = (detectionRect | predictedRect).area();
    return 1.0f - static_cast<float>(intersectionArea) / static_cast<float>(unionArea);
}

/** **************************************************************************
* Compute euclidean center to distance center distance from normalized centers
*
* \param   track track
* \returns   normalized center to center distance [0 ... Sqrt(2)]
*
*************************************************************************** */
float DetectionLocation::center2CenterDist(const Track &track) const {
    cv::Rect2f rect1 = getRect();
    auto center1 = rect1.tl() + cv::Point2f(rect1.size() / 2.0f);

    cv::Rect2f rect2 = track.back().getRect();
    auto center2 = rect2.tl() + cv::Point2f(rect2.size() / 2.0f);

    auto distPerAxis = center2 - center1;
    float relativeWidth = distPerAxis.x / static_cast<float>(frame.data.cols);
    float relativeHeight = distPerAxis.y / static_cast<float>(frame.data.rows);
    return std::sqrt(relativeWidth * relativeWidth + relativeHeight * relativeHeight);
}

/** **************************************************************************
* Compute offset required for pixel alignment between track's tail detection
* and location
*
* \param   track track
* \returns point giving the offset for pixel alignment
*
*************************************************************************** */
cv::Point2d DetectionLocation::phaseCorrelate(Track &track) {
    // TODO: Use words instead of single letters and explain what this does
    cv::Mat1f P, Pm, C;
    cv::mulSpectrums(getDFTFeature(), track.back().getDFTFeature(), P, 0, true);
    cv::magSpectrums(P, Pm);
    cv::divSpectrums(P, Pm, C, 0, false);
    cv::idft(C, C);
    cv::fftShift(C);
    cv::Point peakLoc;
    cv::minMaxLoc(C, nullptr, nullptr, nullptr, &peakLoc);
    double response = 0;
    return cv::Point2d(dftSize_ / 2.0, dftSize_ / 2.0) -
           cv::weightedCentroid(C, peakLoc, cv::Size(5, 5), &response);
}


/** **************************************************************************
* Compute feature distance (similarity) to track's tail detection's
* feature vectors
*
* \param   tr track
* \returns distance [0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::featureDist(Track &track) {
    // roi of track object in track's frame
    cv::Rect2d trackRoi(track.back().getRect());
    if (trackRoi.area() <= 0) {
        LOG_TRACE("tr tail " << trackRoi << " has zero area");
        return 1;
    }

    // center of shifted track roi
    cv::Point2d center
        = cv::Point2d(x_left_upper + 0.5 * width, y_left_upper + 0.5 * height)
                  - phaseCorrelate(track);

    // center not ok
    if (!cv::Rect2d({0, 0}, frame.data.size()).contains(center)) {
        LOG_TRACE("ctr " << center << " outside bgrFrame");
        return 1;
    }

    cv::Mat comp;
    // grab corresponding region from bgrFrame with border replication
    // TODO: Why use two different methods of selecting ROI? Here cv::getRectSubPix is used, but
    //       cv::Mat::operator()(cv::Rect) is used in cv::absdiff below.
    cv::getRectSubPix(frame.data, trackRoi.size(), center, comp);

    // compute pixel wise absolute diff (could do HSV transform 1st?!)
    cv::absdiff(track.back().frame.data(trackRoi), comp, comp);
    assert(frame.data.depth() == CV_8U);

    // get mean normalized BGR pixel difference
    cv::Scalar mean = cv::mean(comp) / 255.0;
    // combine BGR: d = BGR.BGR
    // TODO: Why mean.ddot(mean)? That is the vector's length squared.
    // TODO: Should this be (cv::mean(comp.reshape(1)) / 255.0)[0] ?
    return mean.ddot(mean) / frame.data.channels();
}


/** **************************************************************************
* Compute kalman distance (error from predicted) to track's tail detection's
*
* \param   track track
* \returns distance
*
*************************************************************************** */
float DetectionLocation::kfResidualDist(const Track &track) const {
    return track.testResidual(getRect(), edgeSnapDist_);
}


/** **************************************************************************
* get the location as an opencv rectangle
*************************************************************************** */
cv::Rect2i DetectionLocation::getRect() const {
    return { x_left_upper, y_left_upper, width, height };
}

/** **************************************************************************
* set the location from an opencv rectangle
*************************************************************************** */
void DetectionLocation::setRect(const cv::Rect2i &rec) {
    cv::Rect2i snapped = snapToEdges(rec, rec, frame.data.size(), edgeSnapDist_);
    x_left_upper = snapped.x;
    y_left_upper = snapped.y;
    width = snapped.width;
    height = snapped.height;
}


/** **************************************************************************
* get hanning window of specified size
*
* \param size  size of hanning windowing to return
*
*************************************************************************** */
cv::Mat1f DetectionLocation::getHanningWindow(const cv::Size &size) const {
    if (size.width == dftSize_ && size.height == dftSize_) {
        // most common size so cache it.
        static cv::Mat1f defaultWindow;

        if (defaultWindow.empty() || defaultWindow.rows != dftSize_) {
            //LOG_TRACE("Created default hanning window of size " << defWin.size());
            cv::createHanningWindow(defaultWindow, cv::Size(dftSize_, dftSize_), CV_32F);
        }
        return defaultWindow;
    }
    else {
        cv::Mat1f customWindow;
        //LOG_TRACE("Created custom hanning window of size " << customWin.size());
        cv::createHanningWindow(customWindow, size, CV_32F);
        return customWindow;
    }
}

/** **************************************************************************
* get class feature vector consisting of a unit vector of scores
*
* \returns unit magnitude class score feature vector
*
*************************************************************************** */
cv::Mat DetectionLocation::getClassFeature() const {
    return classFeature_;
}

/** **************************************************************************
* Lazy accessor method to get feature vector
*
* \returns unit magnitude feature vector
*
*************************************************************************** */
cv::Mat DetectionLocation::getDFTFeature() {
    if (!dftFeature_.empty()) {
        return dftFeature_;
    }

    // make normalized grayscale float image
    cv::Mat gray;
    // Convert to gray scale
    cv::cvtColor(frame.data(getRect()), gray, CV_BGR2GRAY);
    cv::Scalar mean;
    cv::Scalar stdDeviation;
    cv::meanStdDev(gray, mean, stdDeviation);
    stdDeviation[0] = std::max(stdDeviation[0], 1 / 255.0);
    gray.convertTo(gray,
                   CV_32FC1,
                   1.0 / stdDeviation[0],
                   -mean[0] / stdDeviation[0]);  // Convert to zero mean unit std

    // zero pad (or clip) center of grayscale image into center of dft buffer
    cv::Mat1f feature = cv::Mat1f::zeros(dftSize_, dftSize_);
    cv::Rect2i grayRoi((gray.size() - feature.size()) / 2, feature.size());
    // Image  source region
    grayRoi &= cv::Rect2i(cv::Point2i(0, 0), gray.size());

    // Buffer target region
    cv::Rect2i featureRoi((feature.size() - grayRoi.size()) / 2, grayRoi.size());

    // Apply hanning window
    if (dftHanningWindowEnabled_) {
        cv::Mat1f hann = getHanningWindow(grayRoi.size());
        cv::multiply(gray(grayRoi), hann, gray(grayRoi));
    }
    // add region from image to buffer
    gray(grayRoi).copyTo(feature(featureRoi));

    // take dft of buffer (CCS packed)
    cv::dft(feature,
            dftFeature_,
            cv::DFT_REAL_OUTPUT);  // created CCS packed dft

    assert(dftFeature_.type() == CV_32FC1);
    return dftFeature_;
}

