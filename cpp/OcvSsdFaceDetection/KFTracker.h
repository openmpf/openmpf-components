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

#ifndef KF_TRACKER_H
#define KF_TRACKER_H

#include <ostream>

#include <log4cxx/logger.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#include "types.h"

namespace MPF{
  namespace COMPONENT{

    using namespace std;

    class KFTracker{

      public:

        static bool Init(log4cxx::LoggerPtr log, const string plugin_path);  ///< setup class shared members

        static cv::Mat_<float> measurementFromBBox(const cv::Rect2i& r);
        static cv::Rect2i      bboxFromState(const cv::Mat_<float> state);


        const cv::Rect2i predictedBBox() const {return bboxFromState(_kf.statePre)  & _roi;};
        const cv::Rect2i correctedBBox() const {return bboxFromState(_kf.statePost) & _roi;};

        void predict(const float t);         ///< advance Kalman state to time t and get predicted bbox
        void correct(const cv::Rect2i &rec); ///< correct current filter state with measurement rec

        KFTracker(const float t,
                  const float dt,
                  const cv::Rect2i &rec0,
                  const cv::Rect2i &roi,
                  const cv::Mat_<float> &rn,
                  const cv::Mat_<float> &qn);

        #ifndef NDEBUG
          static size_t _objId;
          size_t _myId;
          void dump();
          stringstream _state_trace;
          friend ostream& operator<< (ostream& out, const KFTracker& kft);
        #endif

      private:
        static log4cxx::LoggerPtr  _log;       ///< shared log object
        cv::KalmanFilter           _kf;        ///< Kalman filter for bounding box
        float                      _t;         ///< time corresponding to Kalman filter state
        float                      _dt;        ///< time step to use for filter updates
        const cv::Rect2i           _roi;       ///< canvas clipping limits for bboxes returned by filter
        const cv::Mat_<float>      _qn;        ///< Kalman filter process noise variances (i.e. unknown accelerations) [ax,ay,aw,ah]

        void _setTimeStep(float dt);           ///< update model variables Q F for time step size dt

    };


  }
}

#endif