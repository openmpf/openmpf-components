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
#ifndef OCVYOLODETECTION_UTIL_H
#define OCVYOLODETECTION_UTIL_H

#include <ostream>
#include <string>
#include <sstream>
#include <vector>

#include <dlib/matrix.h>

#include <opencv2/core.hpp>

#include <MPFDetectionObjects.h>



/// snap a rectangle to frame edges if close
cv::Rect2i snapToEdges(const cv::Rect2i &rt,
                       const cv::Rect2i &rm,
                       const cv::Size2i &frameSize,
                       float edgeSnapDist);


/// cosine distance between two unit vectors
inline float cosDist(const cv::Mat &f1, const cv::Mat &f2) {
    return 1.0f - std::max(0.0f, std::min(1.0f, static_cast<float>(f1.dot(f2))));
}

/// output cv matrix on single line
std::string format(const cv::Mat1f &m);


/** ****************************************************************************
 *  print out dlib matrix on a single line
 *
 * \param   m matrix to serialize to single line string
 * \returns single line string representation of matrix
 *
***************************************************************************** */
template<typename T>
std::string dformat(const dlib::matrix<T> &m) {
    std::stringstream ss;
    ss << "{";
    for (size_t r = 0; r < m.nr(); r++) {
        for (size_t c = 0; c < m.nc(); c++) {
            ss << m(r, c);
            if (c != m.nc() - 1) { ss << ","; }
        }
        if (r != m.nr() - 1) { ss << "; "; }
    }
    ss << "}";
    return ss.str();
}

/// read opencv matrix from string
cv::Mat fromString(const std::string &data,
                   int rows,
                   int cols,
                   const std::string& dt = "f");



/// output vector to stream
template<typename T>
std::ostream & operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "{";
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i != last) { os << ", "; }
    }
    os << "}";
    return os;
}


namespace MPF {
    namespace COMPONENT {
        /// output MPFImageLocation to stream
        std::ostream &operator<<(std::ostream &os, const MPF::COMPONENT::MPFImageLocation &l);

        /// output MPFVideoTrack to stream
        std::ostream &operator<<(std::ostream &os, const MPF::COMPONENT::MPFVideoTrack &t);
    }
}

namespace cv {
    /// reformat cv::rect output to stream
    std::ostream &operator<<(std::ostream &os, const cv::Rect &r);
}

#endif
