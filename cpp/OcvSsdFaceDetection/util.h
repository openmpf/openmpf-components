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
#ifndef OCVYOLODETECTION_UTIL_H
#define OCVYOLODETECTION_UTIL_H

#include <opencv2/opencv.hpp>
#include <dlib/matrix.h>

#include "detectionComponentUtils.h"

#include "types.h"

namespace MPF{
  namespace COMPONENT{

    using namespace std;

    #define THROW_EXCEPTION(MSG){                                    \
      string path(__FILE__);                                         \
      string f(path.substr(path.find_last_of("/\\") + 1));           \
      throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
    }                                                                          ///< exception macro so we can see where in the code it happened

    cv::Rect2i snapToEdges(const cv::Rect2i& rt,
                           const cv::Rect2i& rm,
                           const cv::Size2i& frameSize,
                          const float edgeSnapDist);                           ///< snap a rectangle to frame edges if close

    inline
    float cosDist(const cv::Mat &f1, const cv::Mat &f2){                       ///< cosine distance between two unit vectors
      return 1.0f - max( 0.0f, min( 1.0f, static_cast<float>(f1.dot(f2))));
    }

    string format(cv::Mat1f m);                                                ///< output cv matrix on single line

    template<typename T>
    string dformat(dlib::matrix<T> m);                                         ///< output dlib matrix on single line

    template<typename T>
    vector<T> fromString(const string data);                                   ///< read vector from string

    cv::Mat fromString(const string data,
                       const int    rows,
                       const int    cols,
                       const string dt);                                       ///< read opencv matrix from string

    template<typename T>
    T get(const Properties &p, const string &k, const T def);                  ///< get MPF properties of various types

    template<typename T>
    T getEnv(const Properties &p, const string &k, const T def);               ///< get MPF properties of various types with fallback to environment variables

    template<typename T>
    ostream& operator<< (ostream& os, const vector<T>& v);                     ///< output vector to stream
    ostream& operator<< (ostream& os, const MPFImageLocation& l);              ///< output MPFImageLocation to stream
    ostream& operator<< (ostream& os, const MPFVideoTrack& t);                 ///< output MPFVideoTrack to stream

  }
}

namespace cv{
  std::ostream& operator<< (std::ostream& os, const cv::Rect& r);              ///< reformat cv::rect output to stream
}

#endif