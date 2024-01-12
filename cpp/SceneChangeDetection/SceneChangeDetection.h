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


#ifndef OPENMPF_COMPONENTS_SceneChangeDetection_H
#define OPENMPF_COMPONENTS_SceneChangeDetection_H

#include <string>
#include <vector>

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>

#include <adapters/MPFVideoDetectionComponentAdapter.h>
#include <MPFDetectionObjects.h>
#include <MPFDetectionComponent.h>


class SceneChangeDetection : public MPF::COMPONENT::MPFVideoDetectionComponentAdapter {
public:
    bool Init() override;

    bool Close() override;

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job) override;

private:
    log4cxx::LoggerPtr logger_;
    cv::Mat dilateKernel;
    int numPixels;

    // Sets threshold for edge detection (range 0-255).
    // Fepresents cutoff score for fraction of mismatches between two frames.
    // Higher thresholds = less frame detections = lower sensitivity.
    double edge_thresh = 70;

    // Threshold for histogram detection.
    // Higher values result in more detections (higher sensitivity).
    // Range 0-1.
    double hist_thresh = 0.9;

    // Threshold for content detection.
    // Higher values result in less detections (lower sensitivity).
    // Range 0-1.
    double cont_thresh = 35;

    // Threshold for thrs detection.
    // Higher values result in more detections (higher sensitivity).
    // Range 0-1? (most likely 0-255).
    double thrs_thresh = 15;

    // Second threshold for thrs detection (combines with thrs_thres).
    // Higher values decrease sensitivity.
    // Range 0-1.
    double minPercent = 0.95;
    int channels[2] = {0,1};
    bool fadeOut;

    // Expected min number of frames between scene changes.
    int minScene = 15;

    // Toggles each type of detection (true = perform detection).
    bool do_hist = true, do_edge = true, do_cont = true, do_thrs = true;
    bool use_middle_frame = true;

    bool DetectChangeEdges(const cv::Mat &frameGray, cv::Mat &lastFrameEdgeFinal) const;
    bool DetectChangeHistogram(const cv::Mat &frame, cv::Mat &lastHist) const;
    bool DetectChangeContent(const cv::Mat &frame, cv::Mat &lastFrameHSV) const;
    bool DetectChangeThreshold(const cv::Mat &frame, cv::Mat &lastFrame);


    int histSize[2] = {30,32};
    // Hue varies from 0 to 179, see cvtColor.
    float hranges[2] = {0,180};
    // Saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color).
    float sranges[2] = {0,256};
    bool frameUnderThreshold(const cv::Mat &image, double threshold, double numPixels) const;
};


#endif
