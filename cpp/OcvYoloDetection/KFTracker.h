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
#include "Config.h"

namespace MPF{
  namespace COMPONENT{

    using namespace std;

    class KFTracker{

      public:
        void             setStatePreFromBBox (const cv::Rect2i& r);
        void             setStatePostFromBBox(const cv::Rect2i& r);
        const cv::Rect2i predictedBBox() const {return _bboxFromState(_kf.statePre)  & _roi;};
        const cv::Rect2i correctedBBox() const {return _bboxFromState(_kf.statePost) & _roi;};

        void predict(const float t);                ///< advance Kalman state to time t and get predicted bbox
        void correct(const cv::Rect2i &rec);        ///< correct current filter state with measurement rec
        float testResidual(const cv::Rect2i &rec, const int snapDist) const;  ///< return a normalized error if rec is assigned
        KFTracker(const float t,
                  const float dt,
                  const cv::Rect2i &rec0,
                  const cv::Rect2i &roi,
                  const cv::Mat1f &rn,
                  const cv::Mat1f &qn);

        KFTracker(KFTracker &&t):_kf(move(t._kf)),
                                 _t(t._t),
                                 _dt(t._dt),
                                 _roi(move(t._roi)),
                                 _qn(move(t._qn)),
                                 _state_trace(t._state_trace.str()){}

        // diagnostic output function for debug/tuning
        void dump(string filename);
        friend ostream& operator<< (ostream& out, const KFTracker& kft);

      private:
        mutable cv::KalmanFilter _kf;          ///< kalman filter for bounding box
        float                    _t;           ///< time corresponding to kalman filter state
        float                    _dt;          ///< time step to use for filter updates
        cv::Rect2i               _roi;         ///< canvas clipping limits for bboxes returned by filter
        cv::Mat1f                _qn;          ///< kalman filter process noise variances (i.e. unknown accelerations) [ax,ay,aw,ah]
        stringstream             _state_trace; ///< time series of states for csv file output supporting debug/tuning

        static cv::Mat1f      _measurementFromBBox(const cv::Rect2i& r);
        static cv::Rect2i     _bboxFromState(const cv::Mat1f& state);
        void                  _setTimeStep(float dt);           ///< update model variables Q F for time step size dt

    };


  }
}

#endif