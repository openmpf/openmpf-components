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


#ifndef OPENMPF_COMPONENTS_LICENSEPLATETEXTDETECTION_H
#define OPENMPF_COMPONENTS_LICENSEPLATETEXTDETECTION_H


#include <string>
#include <vector>

#include <log4cxx/logger.h>
#include <alpr.h>
#include <QHash>
#include <QString>

#include <MPFDetectionComponent.h>
#include <MPFVideoCapture.h>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>

#include "alpr.h"

/**
 * The TextDetection class implements license plate text detection
 * and tracking capabilities for images and videos, and is based on
 * the OpenALPR license plate recognition library.
 */
class LicensePlateTextDetection : public MPF::COMPONENT::MPFImageAndVideoDetectionComponentAdapter {

public:

    /**
     * Constructor
     *
     */
    LicensePlateTextDetection();

    /**
     * Destructor
     */
    ~LicensePlateTextDetection();

    /**
     * The Init method instantiates and configures OpenALPR.  This method
     * must be called once after the constructor but before any other methods
     * are invoked.
     */
    bool Init();

    /**
     * The Close method frees resources used by OpenALPR.  This method
     * should be called prior to the destructor.
     */
    bool Close();

    /**
     * The GetDetections method returns license plate text detected in
     * a single image.
     * @param job   Info about current job
     * @param locations     A vector of type struct MPFImageLocation that holds detected text data upon return
     */
    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFImageJob &job,
            std::vector<MPF::COMPONENT::MPFImageLocation> &locations) override;

    /**
     * The GetDetections method returns license plate text detected in
     * a video.
     * @param job   Info about current job
     * @param tracks    A vector of type struct MPFVideoTrack that holds track data based on text and location upon return
     */
    MPF::COMPONENT::MPFDetectionError GetDetections(
            const MPF::COMPONENT::MPFVideoJob &job,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks) override;

    std::string GetDetectionType();

private:
    MPF::COMPONENT::MPFDetectionError GetDetectionsFromVideoCapture(
            const MPF::COMPONENT::MPFVideoJob &job,
            MPF::COMPONENT::MPFVideoCapture &video_capture,
            std::vector<MPF::COMPONENT::MPFVideoTrack> &tracks);
/**
     * The CompareKeys method determines whether a detected text string
     * has been detected previously in the given image or video, for
     * the purpose of forming tracks based on text.  It uses a string
     * similarity method based on computing the Levenshtein method to
     * compensate for imperfect text detection by OpenALPR.
     * @param key_1 A currently detected license plate text string
     * @param key_2 A previously detected license plate text string
     */
    bool CompareKeys(const std::string &key_1, const std::string &key_2);

    /**
     * The LevenshteinScore method computes a string similarity metric
     * based on determining the Levenshtein distance between two input
     * strings.
     * @param s1 A currently detected license plate text string
     * @param s2 A previously detected license plate text string
     */
    float LevenshteinScore(const std::string &s1, const std::string &s2);

    std::vector<alpr::AlprPlateResult> alprRecognize(const cv::Mat &frame);

    static void SetText(MPF::COMPONENT::MPFImageLocation &detection,
                        const std::string &text);

    static std::string GetText(const MPF::COMPONENT::MPFImageLocation &detection);

    log4cxx::LoggerPtr td_logger_;
    /**< log4cxx Logger pointer passed in from the calling program */
    QHash <QString, QString> parameters;
    /**< Qt hash table that holds input config parameters */
    alpr::Alpr *alpr_;
    /**< Pointer to the created OpenALPR instance */
    float rectangle_intersection_min_;
    /**< minimum amount of license plate area overlap from frame to frame, for location based tracking */
    float levenshtein_score_min_;        /**< minimum string similarity value that should be used to associate detected text with an existing track */

};



#endif //OPENMPF_COMPONENTS_LICENSEPLATETEXTDETECTION_H
