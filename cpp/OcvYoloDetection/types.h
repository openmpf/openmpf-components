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

#ifndef OCVYOLODETECTION_TYPES_H
#define OCVYOLODETECTION_TYPES_H

#include <type_traits>
#include <memory>
#include <list>
#include <iomanip>
#include <opencv2/dnn.hpp>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>


//#define KFDUMP_STATE  //enable csv debug output for kalman filter states

// make function pointer calls more readable
#define CALL_MEMBER_FUNC(object,ptrToMember)  ((object).*(ptrToMember))
#define CALL_FUNC(ptrToFunc) (*(ptrToFunc))

namespace MPF{
 namespace COMPONENT{

    using namespace std;

    class Frame;
    class Config;
    class DetectionLocation;
    class Track;

    using stringVec                = vector<string>;                     ///< vector of strings
    using floatVec                 = vector<float>;                      ///< vector of floats
    using intVec                   = vector<int>;                        ///< vector of integers

    using MPFVideoTrackVec         = vector<MPFVideoTrack>;              ///< vector of MPFVideoTracks
    using MPFImageLocationVec      = vector<MPFImageLocation>;           ///< vector of MPFImageLocations
    using MPFImageLocationVecVec   = vector<MPFImageLocationVec>;        ///< vector of MPFImageLocations vectors

    using cvPixel                  = cv::Point3_<uint8_t>;               ///< image pixel type used by images
    using cvMatVec                 = vector<cv::Mat>;                    ///< vector of OpenCV matrices/images
    using cvMatVecVec              = vector<cvMatVec>;                   ///< vector of OpenCV matrices vectors
    using cvRect2dVec              = vector<cv::Rect2d>;                 ///< vector of OpenCV rectangles
    using cvPoint2iVec             = vector<cv::Point2i>;                ///< vector of OpenCV points
    using cvPoint2fVec             = vector<cv::Point2f>;                ///< vector of OpenCV 2D float points
    using cvPoint2fVecVec          = vector<cvPoint2fVec>;               ///< vector of vectors of OpenCV 2D float points

    using FrameVec                 = vector<Frame>;                      ///< vector for Frames
    using DetectionLocationList    = list<DetectionLocation>;            ///< list of detection locations
    using DetectionLocationListVec = vector<DetectionLocationList>;      ///< vector of DetectionLocation Lists
    using DetectionLocationVec     = vector<DetectionLocation>;          ///< vector of DetectionLocations
    using TrackList                = list<Track>;                        ///< list of tracks
  }
}
#endif