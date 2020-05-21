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

#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  typedef vector<MPFVideoTrack>    MPFVideoTrackVec;     ///< vector of MPFVideoTracks
  typedef vector<MPFImageLocation> MPFImageLocationVec;  ///< vector of MPFImageLocations
  typedef vector<cv::Mat>          cvMatVec;             ///< vector of OpenCV matrices/images
  typedef vector<cv::Rect>         cvRectVec;            ///< vector of OpenCV rectangles
  typedef vector<cv::Point>        cvPointVec;           ///< vector of OpenCV points
  typedef vector<cv::Point2f>      cvPoint2fVec;         ///< vector of OpenCV 2D float points
  typedef vector<cvPoint2fVec>     cvPoint2fVecVec;      ///< vector of vectors of OpenCV 2D float points

  /** **************************************************************************
  *   Dump MPFLocation to a stream
  *************************************************************************** */
  inline
  ostream& operator<< (ostream& os, const MPFImageLocation& l) {
    os << "[" << l.x_left_upper << "," << l.y_left_upper << "]-("
               << l.width        << "," << l.height       << "):"
               << l.confidence;
    return os;
  }

  /** **************************************************************************
  *   Dump MPFTrack to a stream
  *************************************************************************** */
  inline
  ostream& operator<< (ostream& os, const MPFVideoTrack& t) {
    os << t.start_frame << endl;
    os << t.stop_frame  << endl;
    for(auto& p:t.frame_locations){
      os << p.second.x_left_upper << "," << p.second.y_left_upper << ","
          << p.second.width << "," << p.second.height << endl;
    }
    return os;
  }

  /** ****************************************************************************
  *   Dump vectors to a stream
  ***************************************************************************** */
  template<typename T> inline
  ostream& operator<< (ostream& os, const vector<T>& v) {
    os << "{";
    size_t last = v.size() - 1;
    for(size_t i = 0; i < v.size(); ++i){
      os << v[i];
      if(i != last) os << ", ";
    }
    os << "}";
    return os;
  }
 }
}

#endif