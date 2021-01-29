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
#ifndef OCVYOLODETECTION_FRAME_H
#define OCVYOLODETECTION_FRAME_H

#include <utility>

#include <opencv2/opencv.hpp>


/* **************************************************************************
*  Represent a frame with time stamp
*************************************************************************** */
class Frame {
public:
    /// index of frame
    size_t idx;

    /// time of current frame in sec
    double time;

    /// time interval between frames in sec
    double timeStep;

    /// bgr image frame
    cv::Mat data;

    cv::Rect getRect() const {
        return {cv::Point(0, 0), data.size()};
    }

    Frame(size_t idx, double time, double timeStep, cv::Mat data)
        : idx(idx)
        , time(time)
        , timeStep(timeStep)
        , data(std::move(data)) { };


    explicit Frame(cv::Mat data)
        : Frame(0, 0, 0, std::move(data)) {
    }
};


#endif
