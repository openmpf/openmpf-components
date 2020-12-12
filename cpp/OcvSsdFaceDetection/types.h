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

#ifndef OCVSSDFACEDETECTION_TYPES_H
#define OCVSSDFACEDETECTION_TYPES_H

#include <type_traits>
#include <typeinfo>
#include <memory>
#include <list>
#include <iomanip>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>

#include "OrientationType.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  class DetectionLocation;
  class Track;

  using MPFVideoTrackVec        = vector<MPFVideoTrack>;           ///< vector of MPFVideoTracks
  using MPFImageLocationVec     = vector<MPFImageLocation>;        ///< vector of MPFImageLocations
  using cvMatVec                = vector<cv::Mat>;                 ///< vector of OpenCV matrices/images
  using cvRectVec               = vector<cv::Rect>;                ///< vector of OpenCV rectangles
  using cvPointVec              = vector<cv::Point>;               ///< vector of OpenCV points
  using cvPoint2fVec            = vector<cv::Point2f>;             ///< vector of OpenCV 2D float points
  using cvPoint2fVecVec         = vector<cvPoint2fVec>;            ///< vector of vectors of OpenCV 2D float points
  using DetectionLocationPtr    = unique_ptr<DetectionLocation>;   ///< DetectionLocation pointers
  using DetectionLocationPtrVec = vector<DetectionLocationPtr>;    ///< vector of DetectionLocation pointers
  using TrackPtr                = unique_ptr<Track>;               ///< pointer to a track
  using TrackPtrList            = list<TrackPtr>;                  ///< list of track pointers

 }
}

#endif