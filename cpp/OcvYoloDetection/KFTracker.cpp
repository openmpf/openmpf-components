/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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

#include "DetectionLocation.h"
#include "KFTracker.h"


#define PROCESS_NOISE CONTINUOUS_WHITE
// TODO: Consider exposing a property to use piecewise white noise instead of continuous white noise.
//#define PROCESS_NOISE PIECEWISE_WHITE

using namespace std;
using namespace MPF::COMPONENT;

// Kalman Filter Dimensions (4x constant acceleration model)
const int KF_STATE_DIM = 12; ///< dimensionality of Kalman state vector       [x, vx, ax, y, vy, ay, w, vw, aw, h, vh, ah]
const int KF_MEAS_DIM = 4;   ///< dimensionality of Kalman measurement vector [x, y, w, h]
const int KF_CTRL_DIM = 0;   ///< dimensionality of Kalman control input      []


/** **************************************************************************
 * Update the model state transition matrix F and the model noise covariance
 * matrix Q to be suitable of a timestep of dt
 *
 * \param dt new time step in sec
 *
 * \note F and Q are block diagonal with 4 blocks, one for each x, y, w, h
 *
*************************************************************************** */
void KFTracker::_setTimeStep(float dt) {

    if (fabs(_dt - dt) > 2 * numeric_limits<float>::epsilon()) {
        _dt = dt;                                                                  //LOG_TRACE("kd dt:" << _dt);

        float dt2 = dt * dt;
        float dt3 = dt2 * dt;
        float dt4 = dt2 * dt2;

        float half_dt2 = 0.5 * dt2;

#if PROCESS_NOISE == CONTINUOUS_WHITE
        float third_dt3 = dt3 / 3.0;
        float sixth_dt3 = dt3 / 6.0;
        float eighth_dt4 = dt4 / 8.0;
        float twentieth_dt5 = dt2 * dt3 / 20.0;
#elif PROCESS_NOISE == PIECEWISE_WHITE
        float    half_dt3 = dt3 / 2.0;
        float quarter_dt4 = dt4 / 4.0;
#endif

        for (int b = 0; b < 4; b++) {
            int i = 3 * b;
            // update state transition matrix F
            //    | 0  1    2   3  4    5    6  7    8    9 10   11
            //--------------------------------------------------------
            //  0 | 1 dt .5dt^2 0  0    0    0  0    0    0  0    0   |   | x|
            //  1 | 0  1   dt   0  0    0    0  0    0    0  0    0   |   |vx|
            //  2 | 0  0    1   0  0    0    0  0    0    0  0    0   |   |ax|
            //  3 | 0  0    0   1 dt .5dt^2  0  0    0    0  0    0   |   | y|
            //  4 | 0  0    0   0  1   dt    0  0    0    0  0    0   |   |vy|
            //  5 | 0  0    0   0  0    1    0  0    0    0  0    0   |   |ay|
            //  6 | 0  0    0   0  0    0    1 dt .5dt^2  0  0    0   |   | w|
            //  7 | 0  0    0   0  0    0    0  1   dt    0  0    0   |   |vw|
            //  8 | 0  0    0   0  0    0    0  0    1    0  0    0   |   |aw|
            //  9 | 0  0    0   0  0    0    0  0    0    1 dt .5dt^2 |   | h|
            // 10 | 0  0    0   0  0    0    0  0    0    0  1   dt   |   |vh|
            // 11 | 0  0    0   0  0    0    0  0    0    0  0    1   |   |ah|
            _kf.transitionMatrix.at<float>(i, 1 + i) =
            _kf.transitionMatrix.at<float>(1 + i, 2 + i) = dt;
            _kf.transitionMatrix.at<float>(i, 2 + i) = half_dt2;

            // update process noise matrix Q
#if PROCESS_NOISE == CONTINUOUS_WHITE
            // See "Out[4]" here:
            // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
            //    | 0       1      2      3       4      5      6       7      8      9      10     11
            //----------------------------------------------------------------------------------------------
            //  0 | dt^5/20 dt^4/8 dt^3/6 0       0      0      0       0      0      0       0      0      |   | x|
            //  1 | dt^4/8  dt^3/3 dt^2/2 0       0      0      0       0      0      0       0      0      |   |vx|
            //  2 | dt^3/6  dt^2/2 dt     0       0      0      0       0      0      0       0      0      |   |ax|
            //  3 | 0       0      0      dt^5/20 dt^4/8 dt^3/6 0       0      0      0       0      0      |   | y|
            //  4 | 0       0      0      dt^4/8  dt^3/3 dt^2/2 0       0      0      0       0      0      |   |vy|
            //  5 | 0       0      0      dt^3/6  dt^2/2 dt     0       0      0      0       0      0      |   |ay|
            //  6 | 0       0      0      0       0      0      dt^5/20 dt^4/8 dt^3/6 0       0      0      |   | w|
            //  7 | 0       0      0      0       0      0      dt^4/8  dt^3/3 dt^2/2 0       0      0      |   |vw|
            //  8 | 0       0      0      0       0      0      dt^3/6  dt^2/2 dt     0       0      0      |   |aw|
            //  9 | 0       0      0      0       0      0      0       0      0      dt^5/20 dt^4/8 dt^3/6 |   | h|
            // 10 | 0       0      0      0       0      0      0       0      0      dt^4/8  dt^3/3 dt^2/2 |   |vh|
            // 11 | 0       0      0      0       0      0      0       0      0      dt^3/6  dt^2/2 dt     |   |ah|
            _kf.processNoiseCov.at<float>(i, i) = _qn.at<float>(b, 0) * twentieth_dt5;
            _kf.processNoiseCov.at<float>(1 + i, i) =
            _kf.processNoiseCov.at<float>(i, 1 + i) = _qn.at<float>(b, 0) * eighth_dt4;
            _kf.processNoiseCov.at<float>(2 + i, i) =
            _kf.processNoiseCov.at<float>(i, 2 + i) = _qn.at<float>(b, 0) * sixth_dt3;
            _kf.processNoiseCov.at<float>(1 + i, 1 + i) = _qn.at<float>(b, 0) * third_dt3;
            _kf.processNoiseCov.at<float>(1 + i, 2 + i) =
            _kf.processNoiseCov.at<float>(2 + i, 1 + i) = _qn.at<float>(b, 0) * half_dt2;
            _kf.processNoiseCov.at<float>(2 + i, 2 + i) = _qn.at<float>(b, 0) * dt;
#elif PROCESS_NOISE == PIECEWISE_WHITE
            // See "Out[8]" here:
            // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
            //    | 0      1      2      3      4      5      6      7      8      9     10     11
            //------------------------------------------------------------------------------------------
            //  0 | dt^4/4 dt^3/2 dt^2/2 0      0      0      0      0      0      0      0      0      |   | x|
            //  1 | dt^3/2 dt^2   dt     0      0      0      0      0      0      0      0      0      |   |vx|
            //  2 | dt^2/2 dt     1      0      0      0      0      0      0      0      0      0      |   |ax|
            //  3 | 0       0     0      dt^4/4 dt^3/2 dt^2/2 0      0      0      0      0      0      |   | y|
            //  4 | 0       0     0      dt^3/2 dt^2   dt     0      0      0      0      0      0      |   |vy|
            //  5 | 0       0     0      dt^2/2 dt     1      0      0      0      0      0      0      |   |ay|
            //  6 | 0       0     0      0      0      0      dt^4/4 dt^3/2 dt^2/2 0      0      0      |   | w|
            //  7 | 0       0     0      0      0      0      dt^3/2 dt^2   dt     0      0      0      |   |vw|
            //  8 | 0       0     0      0      0      0      dt^2/2 dt     1      0      0      0      |   |aw|
            //  9 | 0       0     0      0      0      0      0      0      0      dt^4/4 dt^3/2 dt^2/2 |   | h|
            // 10 | 0       0     0      0      0      0      0      0      0      dt^3/2 dt^2   dt     |   |vh|
            // 11 | 0       0     0      0      0      0      0      0      0      dt^2/2 dt     1      |   |ah|
            _kf.processNoiseCov.at<float>(  i,  i) = _qn.at<float>(b,0) * quarter_dt4;
            _kf.processNoiseCov.at<float>(1+i,  i) =
            _kf.processNoiseCov.at<float>(  i,1+i) = _qn.at<float>(b,0) * half_dt3;
            _kf.processNoiseCov.at<float>(2+i,  i) =
            _kf.processNoiseCov.at<float>(  i,2+i) = _qn.at<float>(b,0) * half_dt2;
            _kf.processNoiseCov.at<float>(1+i,1+i) = _qn.at<float>(b,0) * dt2;
            _kf.processNoiseCov.at<float>(1+i,2+i) =
            _kf.processNoiseCov.at<float>(2+i,1+i) = _qn.at<float>(b,0) * dt;
            _kf.processNoiseCov.at<float>(2+i,2+i) = _qn.at<float>(b,0);
#endif
        }

    }
}

/** **************************************************************************
 * Convert bbox to a measurement vector consisting of center coordinate and
 * width and height dimensions
 *
 * \param r bounding box to convert
 *
 * \returns measurement vector
 *
*************************************************************************** */
cv::Mat1f KFTracker::_measurementFromBBox(const cv::Rect2i &r) {
    cv::Mat1f z(KF_MEAS_DIM, 1);
    z.at<float>(0) = r.x + r.width / 2.0f;
    z.at<float>(1) = r.y + r.height / 2.0f;
    z.at<float>(2) = r.width;
    z.at<float>(3) = r.height;
    return z;
}

/** **************************************************************************
 * "Paste" a rectangle   [ x  y  w  h]
 *  into a filter state  [ x,vx,ax, y,vy,ay, w,wv,aw, h,vh,ah]
 *                         0  1  2  3  4  5  6  7  8  9 10 11
 *
 * \param r        rectangle to paste into state
 * \param [in,out] state state to paste measurement into
 *
 * \note "hidden" (not affected by measurement) states will not be changed
 *
*************************************************************************** */
void KFTracker::setStatePreFromBBox(const cv::Rect2i &r) {

    cv::Mat1f ns = _kf.measurementMatrix.t() * _measurementFromBBox(r);

    for (int i = 0; i < ns.rows; i++) {
        if (fabs(ns.at<float>(i)) > 2.0 * FLT_EPSILON) {
            _kf.statePre.at<float>(i) = ns.at<float>(i);
        }
    }
}

/** **************************************************************************
 * "Paste" a rectangle   [ x  y  w  h]
 *  into a filter state  [ x,vx,ax, y,vy,ay, w,wv,aw, h,vh,ah]
 *                         0  1  2  3  4  5  6  7  8  9 10 11
 *
 * \param r        rectangle to paste into state
 *
 * \note "hidden" (not affected by measurement) states will not be changed
 *
*************************************************************************** */
void KFTracker::setStatePostFromBBox(const cv::Rect2i &r) {

    cv::Mat1f ns = _kf.measurementMatrix.t() * _measurementFromBBox(r);

    for (int i = 0; i < ns.rows; i++) {
        if (fabs(ns.at<float>(i)) > 2.0 * FLT_EPSILON) {
            _kf.statePost.at<float>(i) = ns.at<float>(i);
        }
    }
}

/** **************************************************************************
 * Convert a filter state [ x,vx,ax, y,vy,ay, w,wv,aw, h,vh,ah] to roi(x,y,w,h)
 *                          0  1  2  3  4  5  6  7  8  9 10 11
 *
 * \param state filter state vector to convert
 *
 * \returns bounding box corresponding to state
 *
*************************************************************************** */
cv::Rect2i KFTracker::_bboxFromState(const cv::Mat1f &state) {
    return cv::Rect2i(static_cast<int>(state.at<float>(0) - state.at<float>(6) / 2.0f + 0.5f),
                      static_cast<int>(state.at<float>(3) - state.at<float>(9) / 2.0f + 0.5f),
                      static_cast<int>(state.at<float>(6) + 0.5f),
                      static_cast<int>(state.at<float>(9) + 0.5f));
}

/** **************************************************************************
 * Get bounding box prediction by advancing filter state to time t
 * and update filter time to t
 *
 * \param t time in sec to advance filter to with prediction
 *
*************************************************************************** */
void KFTracker::predict(float t) {
    _setTimeStep(t - _t);
    _t = t;
    _kf.predict();
    _kf.errorCovPre += _kf.errorCovPre.t();   // guarantee cov symmetry to help stability
    _kf.errorCovPre /= 2.0f;
    // TODO: Consider exposing a property to disable width/height velocity/acceleration to reduce "bouncy" tracks.
    /*_kf.statePre.at<float>(8) =
    _kf.statePre.at<float>(11) = 0.0;  // kill width & height acceleration
    _kf.statePre.at<float>(7) =
    _kf.statePre.at<float>(10) = 0.0;  // kill width & height velocity */
}

/** **************************************************************************
 * Get bounding box from state after corrected by a measurement at filter time t
 *
 * \param rec measurement to use for correcting filter state at current filter
 *            time
 *
 * \note filter will record and dump error statistics in _state_trace if
 *       compiled for debug
 *
*************************************************************************** */
void KFTracker::correct(const cv::Rect2i &rec) {
    _kf.correct(_measurementFromBBox(rec));
    _kf.errorCovPost += _kf.errorCovPost.t();   // guarantee cov symmetry to help stability
    _kf.errorCovPost /= 2.0f;
    // TODO: Consider exposing a property to disable width/height velocity/acceleration to reduce "bouncy" tracks.
    /*_kf.statePost.at<float>(8) =
    _kf.statePost.at<float>(11) = 0.0;  // kill width & height acceleration
    _kf.statePost.at<float>(7) =
    _kf.statePost.at<float>(10) = 0.0;  // kill width & height velocity */
#ifdef KFDUMP_STATE
    _state_trace << (*this) << endl;
#endif
}

/** **************************************************************************
 * [ x,vx,ax, y,vy,ay, w,wv,aw, h,vh,ah] to roi(x,y,w,h)
 *   0  1  2  3  4  5  6  7  8  9 10 11
*************************************************************************** */
float KFTracker::testResidual(const cv::Rect2i &rec, const int edgeSnapDist) const {
    // backup variables
    cv::Mat1f gain = _kf.gain.clone();
    cv::Mat1f statePost = _kf.statePost.clone();
    cv::Mat1f errorCovPost = _kf.errorCovPost.clone();

    // perform trial correction and get error squared
    _kf.correct(_measurementFromBBox(rec));
    //_kf.errorCovPost += _kf.errorCovPost.t();   // guarantee cov symmetry to help stability
    //_kf.errorCovPost /= 2.0f;

    cv::Mat1f err_sq = _kf.statePre - _kf.statePost;
    err_sq = err_sq.mul(err_sq);

    // be permissive knock out errors due to frame edges
    int border_x = static_cast<float>(edgeSnapDist * _roi.width);
    cv::Rect2i corrBBox = correctedBBox();
    if (corrBBox.x <= border_x
        || corrBBox.x >= _roi.width - border_x
        || rec.x <= border_x
        || rec.x >= _roi.width - border_x) {
        //err_sq.at<float>(0) =
        //err_sq.at<float>(1) =
        //err_sq.at<float>(2) =
        err_sq.at<float>(6) =
        err_sq.at<float>(7) =
        err_sq.at<float>(8) = 0.0f;
    }
    int border_y = static_cast<float>(edgeSnapDist * _roi.height);
    if (corrBBox.y <= border_y
        || corrBBox.y >= _roi.height - border_y
        || rec.y <= border_y
        || rec.y >= _roi.height - border_y) {
        //err_sq.at<float>( 3) =
        //err_sq.at<float>( 4) =
        //err_sq.at<float>( 5) =
        err_sq.at<float>(9) =
        err_sq.at<float>(10) =
        err_sq.at<float>(11) = 0.0f;
    }

    // restore variables changed by correction
    _kf.gain = gain;
    _kf.statePost = statePost;
    _kf.errorCovPost = errorCovPost;

    // get max maximum normalized error
    err_sq = err_sq / _kf.errorCovPre.diag();  // div by 0.0 is handled by returning 0.0
    float maxErr_sq = 0.0;
    for (int i = 0; i < err_sq.rows; i++) maxErr_sq = max(maxErr_sq, err_sq.at<float>(i));
    float maxErr = sqrt(maxErr_sq);
    return maxErr;
}

/** **************************************************************************
 * Construct and initialize Kalman filter motion tracker
 *
 * \param t    time corresponding to initial bounding box rec0
 * \param dt   timestep to use for the filter (sec)
 * \param rec0 initial bounding box measurement
 * \param roi  clipping constraints so BBoxes don't wonder off frame
 * \param rn   4x1 column vector of variances for measurement noise, var([x,y,w,h])
 * \param qn   4x1 column vector of variances for model noise, var([ax,ay,aw,ah])
 *
*************************************************************************** */
KFTracker::KFTracker(const float t,
                     const float dt,
                     const cv::Rect2i &rec0,
                     const cv::Rect2i &roi,
                     const cv::Mat1f &rn,
                     const cv::Mat1f &qn) :
        _t(t),
        _dt(-1.0f),
        _roi(roi),
        _qn(qn),
        _kf(KF_STATE_DIM, KF_MEAS_DIM, KF_CTRL_DIM, CV_32F)       // Kalman(state_dim,meas_dim,contr_dim)
{
    assert(rn.rows == KF_MEAS_DIM && rn.cols == 1);
    assert(qn.rows == KF_MEAS_DIM && qn.cols == 1);  // only accelerations model noise should be specified

    assert(_roi.x == 0);
    assert(_roi.y == 0);
    assert(_roi.width > 0);
    assert(_roi.height > 0);

    // measurement matrix H
    //      0  1  2  3  4  5  6  7  8  9 10 11
    //-----------------------------------------
    //  0 | 1  0  0  0  0  0  0  0  0  0  0  0 |       | x |
    //  1 | 0  0  0  1  0  0  0  0  0  0  0  0 |       | y |
    //  2 | 0  0  0  0  0  0  1  0  0  0  0  0 | * A = | w |
    //  3 | 0  0  0  0  0  0  0  0  0  1  0  0 |       | h |
    _kf.measurementMatrix.at<float>(0, 0) =
    _kf.measurementMatrix.at<float>(1, 3) =
    _kf.measurementMatrix.at<float>(2, 6) =
    _kf.measurementMatrix.at<float>(3, 9) = 1.0f;

    // measurement noise covariance matrix R
    //    | 0   1  2  3
    //------------------
    //  0 | xx  0  0  0
    //  1 |  0 yy  0  0
    //  2 |  0  0 ww  0
    //  3 |  0  0  0  hh
    _kf.measurementNoiseCov.at<float>(0, 0) = rn.at<float>(0);
    _kf.measurementNoiseCov.at<float>(1, 1) = rn.at<float>(1);
    _kf.measurementNoiseCov.at<float>(2, 2) = rn.at<float>(2);
    _kf.measurementNoiseCov.at<float>(3, 3) = rn.at<float>(3);

    //adjust F and setup process noise covariance matrix Q
    _setTimeStep(dt);

    //initialize filter state
    cv::Mat1f z0 = _measurementFromBBox(rec0);
    _kf.statePost.at<float>(0) = z0.at<float>(0);
    _kf.statePost.at<float>(3) = z0.at<float>(1);
    _kf.statePost.at<float>(6) = z0.at<float>(2);
    _kf.statePost.at<float>(9) = z0.at<float>(3);

    //initialize error covariance matrix P
    // See "Design the Measurement Noise Matrix" here:
    // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/08-Designing-Kalman-Filters.ipynb
    // z: [ x, y, w, h]
    // x: [ x,vx,ax, y,vy,ay, w,wv,aw, h,vh,ah]
    //      0  1  2  3  4  5  6  7  8  9 10 11
    _kf.errorCovPost.at<float>(0, 0) = rn.at<float>(0);
    _kf.errorCovPost.at<float>(1, 1) =
            (z0.at<float>(2) / dt) * (z0.at<float>(2) / dt);    // guess max vx as one width/dt
    _kf.errorCovPost.at<float>(2, 2) = 10.0 * _kf.processNoiseCov.at<float>(2, 2);      // guess ~3 sigma ?!

    _kf.errorCovPost.at<float>(3, 3) = rn.at<float>(1);
    _kf.errorCovPost.at<float>(4, 4) =
            (z0.at<float>(3) / dt) * (z0.at<float>(3) / dt);   // guess max vy as one height/dt
    _kf.errorCovPost.at<float>(5, 5) = 10.0 * _kf.processNoiseCov.at<float>(5, 5);     // guess ~3 sigma ?!

    _kf.errorCovPost.at<float>(6, 6) = rn.at<float>(2);
    _kf.errorCovPost.at<float>(7, 7) = 10.0 * _kf.processNoiseCov.at<float>(7, 7);     // guess ~3 sigma
    _kf.errorCovPost.at<float>(8, 8) = 10.0 * _kf.processNoiseCov.at<float>(8, 8);     // guess ~3 sigma

    _kf.errorCovPost.at<float>(9, 9) = rn.at<float>(3);
    _kf.errorCovPost.at<float>(10, 10) = 10.0 * _kf.processNoiseCov.at<float>(10, 10); // guess ~3 sigma
    _kf.errorCovPost.at<float>(11, 11) = 10.0 * _kf.processNoiseCov.at<float>(11, 11); // guess ~3 sigma

#ifdef KFDUMP_STATE
    _state_trace << (*this) << endl;                    // trace filter initial error stats
#endif
}

#include <fstream>

/** **************************************************************************
*  Write out error statistics over time to csv file for filter tuning
*************************************************************************** */
void KFTracker::dump(const string& filename) {
    ofstream dump;
    dump.open(filename, ofstream::out | ofstream::trunc);
    dump << "t,";
    dump << "px,pvx,pax, py,pvy,pay, pw,pvw,paw, ph,pvh,pah, ";
    dump << "cx,cvx,cax, cy,cvy,cay, cw,cvw,caw, ch,cvh,cah, ";
    dump << "err_x, err_y, err_w, err_h,";
    for (int r = 0; r < 12; r++) {
        dump << "P" << setfill('0') << setw(2) << r << "_" << r;
        dump << ",";
    }
    dump << endl;
    dump << _state_trace.rdbuf();
    dump.close();
}

/** **************************************************************************
*   Dump Diagnostics for MPF::COMPONENT::KFTracker to a stream
*************************************************************************** */
ostream &operator<<(ostream &out, const KFTracker &kft) {
    out << kft._t << ",";

    for (int i = 0; i < kft._kf.statePre.rows; i++) {
        out << kft._kf.statePre.at<float>(i) << ",";
    }
    out << " ";

    for (int i = 0; i < kft._kf.statePost.rows; i++) {
        out << kft._kf.statePost.at<float>(i) << ",";
    }
    out << " ";

    for (int i = 0; i < kft._kf.temp5.rows; i++) {
        out << kft._kf.temp5.at<float>(i) << ",";
    }
    out << " ";

    for (int r = 0; r < kft._kf.errorCovPost.rows; r++) {
        //for(int c=0; c<kft._kf.errorCovPost.cols; c++){
        out << sqrt(kft._kf.errorCovPost.at<float>(r, r)) << ",";
        //}
    }
    return out;
}
