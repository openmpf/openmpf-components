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


#ifndef OPENMPF_COMPONENTS_SceneChangeDetection_H
#define OPENMPF_COMPONENTS_SceneChangeDetection_H


#include <log4cxx/logger.h>
#include <QHash>
#include <QString>


#include <adapters/MPFVideoDetectionComponentAdapter.h>
#include <MPFDetectionComponent.h>
#include <MPFVideoCapture.h>
#include <MPFImageReader.h>


class SceneChangeDetection : public MPF::COMPONENT::MPFVideoDetectionComponentAdapter {
    QHash<QString, QString> parameters;
public:

    SceneChangeDetection();

    ~SceneChangeDetection();

    bool Init();

    bool Close();

    std::vector<MPF::COMPONENT::MPFVideoTrack> GetDetections(const MPF::COMPONENT::MPFVideoJob &job);

    std::string GetDetectionType();


private:

    log4cxx::LoggerPtr logger_;
    cv::Mat dilateKernel;
    int numPixels;
    double edge_thresh;
    double hist_thresh;
    double cont_thresh;
    double thrs_thresh;
    double minPercent;
    std::map<int,int> keyframes;
    int channels[2] = {0,1};
    bool fadeOut;
    int minScene;

    bool do_hist, do_edge, do_cont, do_thrs, use_middle_frame;

    bool DetectChangeEdges(const cv::Mat &frameGray, cv::Mat &lastFrameEdgeFinal);
    bool DetectChangeHistogram(const cv::Mat &frame, cv::Mat &lastHist);
    bool DetectChangeContent(const cv::Mat &frame, cv::Mat &lastFrameHSV);
    bool DetectChangeThreshold(const cv::Mat &frame, cv::Mat &lastFrame);


    int histSize[2] = {30,32};
    // Hue varies from 0 to 179, see cvtColor.
    float hranges[2] = {0,180};
    // Saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color).
    float sranges[2] = {0,256};
    bool frameUnderThreshold(const cv::Mat &image, double threshold, double numPixels);

    // Support for reading .ini files.
    void SetDefaultParameters();
    void SetReadConfigParameters();
};


#endif
