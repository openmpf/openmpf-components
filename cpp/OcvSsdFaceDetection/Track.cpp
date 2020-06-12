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

#include "Track.h"

using namespace MPF::COMPONENT;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr Track::_log;


string format(cv::Mat m);

/** ***************************************************************************
*  Constructor
*
* \param detPtr a 1st detection to add to the new track
* \param cfg    job struct to initialize kalman filter params from
*
**************************************************************************** */
Track::Track(unique_ptr<DetectionLocation> detPtr, const JobConfig &cfg):
  _kfDisabled(cfg.kfDisabled),
  _kf(KF_STATE_DIM,KF_MEAS_DIM,KF_CTRL_DIM,CV_32F)                 // kalman(state_dim,meas_dim,contr_dim)
{
  if(! _kfDisabled){
    // State Transition Matrix A
    //    | 1  0 dt  0  0  0  0  0|   | x|
    //    | 0  1  0 dt  0  0  0  0|   | y|
    //    | 0  0  1  0  0  0  0  0|   |vx|
    //    | 0  0  0  1  0  0  0  0| * |vy|
    //    | 0  0  0  0  1  0 dt  0|   | w|
    //    | 0  0  0  0  0  1  0 dt|   | h|
    //    | 0  0  0  0  0  0  1  0|   |vw|
    //    | 0  0  0  0  0  0  0  1|   |vh|
    _kf.transitionMatrix = cv::Mat_<float>::eye(KF_STATE_DIM,KF_STATE_DIM); // add dt later

    _kf.measurementMatrix   = cfg.H;
    _kf.processNoiseCov     = cfg.Q;
    _kf.measurementNoiseCov = cfg.R;
    _kf.errorCovPre = cv::Mat::eye(KF_STATE_DIM,KF_STATE_DIM,CV_32F);  // 1st measurements taken truth as no noise

    // initialize state vector
    _kf.statePost = cv::Mat_<float>::zeros(KF_STATE_DIM,1);
    _kf.statePost.at<float>(0,0) = detPtr->x_left_upper + detPtr->width /2.0f; // x
    _kf.statePost.at<float>(1,0) = detPtr->y_left_upper + detPtr->height/2.0f; // y
    _kf.statePost.at<float>(4,0) = detPtr->width;                              // width
    _kf.statePost.at<float>(5,0) = detPtr->height;                             // height
                                                                               LOG4CXX_TRACE(_log,"kf initial state:" << format(_kf.statePost));
  }
  _locationPtrs.push_back(move(detPtr));
}

/** **************************************************************************
* Advance Kalman filter system state to frameIdx time step
*
* \param frameIdx to which to advance the filter state to
*
*************************************************************************** */
void Track::kalmanPredict(const size_t frameIdx){

  if(_kfDisabled) return;

  float dt = frameIdx - _locationPtrs.back()->frameIdx;
  // set st in state transition matrix
  _kf.transitionMatrix.at<float>(0,2) = dt;
  _kf.transitionMatrix.at<float>(1,3) = dt;
  _kf.transitionMatrix.at<float>(4,6) = dt;
  _kf.transitionMatrix.at<float>(5,7) = dt;
  _kf.predict();                                                               LOG4CXX_TRACE(_log,"kf predicted:" << getKalmanPredictedLocation());

}

/** **************************************************************************
* Correct last measurement and update gains using Kalman filter
*************************************************************************** */
void Track::kalmanCorrect(){

  if(_kfDisabled) return;

  cv::Mat meas = cv::Mat_<float>(KF_MEAS_DIM,1);
  meas.at<float>(0,0) = _locationPtrs.back()->x_left_upper + _locationPtrs.back()->width /2.0f; // x
  meas.at<float>(1,0) = _locationPtrs.back()->y_left_upper + _locationPtrs.back()->height/2.0f; // y
  meas.at<float>(2,0) = _locationPtrs.back()->width;   // width
  meas.at<float>(3,0) = _locationPtrs.back()->height;  // height
                                                                               LOG4CXX_TRACE(_log, "kf predicted:" << format(_kf.statePre));
                                                                               LOG4CXX_TRACE(_log, "kf meas:" << format(meas));
  _kf.correct(meas);                                                           LOG4CXX_TRACE(_log, "kf corrected:" << format(_kf.statePost));

  cv::Rect2i cloc = getKalmanCorrectedLocation();
  _locationPtrs.back()->x_left_upper = cloc.x;
  _locationPtrs.back()->y_left_upper = cloc.y;
  _locationPtrs.back()->width        = cloc.width;
  _locationPtrs.back()->height       = cloc.height;                            LOG4CXX_TRACE(_log,"corrected detection: f"<<_locationPtrs.back()->frameIdx<<*_locationPtrs.back());
}

/** **************************************************************************
* Retrieve the current predicted position rectangle from Kalman filter state
*
* \return rectangle corresponding to filter predicted state (without measuremnt)
*
*************************************************************************** */
const cv::Rect2i  Track::getKalmanPredictedLocation() const{
  if(_kfDisabled){
    return cv::Rect2i(_locationPtrs.back()->x_left_upper,
                      _locationPtrs.back()->y_left_upper,
                      _locationPtrs.back()->width,
                      _locationPtrs.back()->height);
  }else{
    return cv::Rect2i(static_cast<int>(_kf.statePre.at<float>(0)
                                    - _kf.statePre.at<float>(4)/2.0f + 0.5f),
                      static_cast<int>(_kf.statePre.at<float>(1)
                                    - _kf.statePre.at<float>(5)/2.0f + 0.5f),
                      static_cast<int>(_kf.statePre.at<float>(4)      + 0.5f),
                      static_cast<int>(_kf.statePre.at<float>(5)      + 0.5f));
  }
}

/** **************************************************************************
* Retrieve the current corrected position rectangle from Kalman filter state
*  corrected states are after a new position measurement has been taken
*
* \return rectangle corresponding to filter corrected state (post measuremnt)
*
*************************************************************************** */
const cv::Rect2i  Track::getKalmanCorrectedLocation() const{
  if(_kfDisabled){
    return cv::Rect2i(_locationPtrs.back()->x_left_upper,
                      _locationPtrs.back()->y_left_upper,
                      _locationPtrs.back()->width,
                      _locationPtrs.back()->height);
  }else{
    return cv::Rect2i(static_cast<int>(_kf.statePost.at<float>(0)
                                     - _kf.statePost.at<float>(4)/2.0f + 0.5f),
                      static_cast<int>(_kf.statePost.at<float>(1)
                                     - _kf.statePost.at<float>(5)/2.0f + 0.5f),
                      static_cast<int>(_kf.statePost.at<float>(4)      + 0.5f),
                      static_cast<int>(_kf.statePost.at<float>(5)      + 0.5f));
  }
}

/** **************************************************************************
*  Get a new DetectionLocation from an existing one based on a frame
*
* \param cfg job struct to access current image frame from
*
* \returns ptr to new location based on tracker's estimation
*
* \note    tracker is passed on to the new location on success
*
**************************************************************************** */
DetectionLocationPtr Track::ocvTrackerPredict(const JobConfig &cfg){

  if(_trackerPtr.empty()){   // initialize a new tracker if we don't have one already
    _trackerPtr = cv::TrackerMOSSE::create();  // could try different trackers here. e.g. cv::TrackerKCF::create();
    _trackerPtr->init(_locationPtrs.back()->getBGRFrame(),
           cv::Rect2d(_locationPtrs.back()->x_left_upper,
                      _locationPtrs.back()->y_left_upper,
                      _locationPtrs.back()->width,
                      _locationPtrs.back()->height));                                    LOG4CXX_TRACE(_log,"tracker created for " << (MPFImageLocation)*_locationPtrs.back());
    _trackerStartFrameIdx = cfg.frameIdx;
  }

  cv::Rect2d p;
  DetectionLocationPtr detPtr;
  if(cfg.frameIdx - _trackerStartFrameIdx <= cfg.maxFrameGap){
    if(_trackerPtr->update(cfg.bgrFrame,p)){
      detPtr = DetectionLocationPtr(new DetectionLocation(
        p.x, p.y, p.width, p.height, 0.0,
        cv::Point2f((p.x + p.width/2.0f )/static_cast<float>(cfg.bgrFrame.cols),
                    (p.y + p.height/2.0f)/static_cast<float>(cfg.bgrFrame.rows)),
        cfg.frameIdx,
        cfg.bgrFrame));                                                                 LOG4CXX_TRACE(_log,"tracking " << (MPFImageLocation)*_locationPtrs.back() << " to " << (MPFImageLocation)*detPtr);

      detPtr->copyFeature(*(_locationPtrs.back()));  // clone feature of prior detection
    }else{
                                                                                        LOG4CXX_TRACE(_log,"could not track " << (MPFImageLocation)*_locationPtrs.back() << " to new location");
    }
  }else{
                                                                                        LOG4CXX_TRACE(_log,"extrapolation tracking stopped" << (MPFImageLocation)*_locationPtrs.back() << " frame gap = " << cfg.frameIdx - _trackerStartFrameIdx << " > " <<  cfg.maxFrameGap);
  }
  return detPtr;
}

/** **************************************************************************
* Add detection pointer to the end of the track
*
* \param d pointer to the detection to append to the track
*
* \note the previous track tail will have its image frame released
*************************************************************************** */
void Track::push_back(DetectionLocationPtr d){

  assert(_locationPtrs.size() > 0);
  _locationPtrs.back()->releaseBGRFrame();
  _locationPtrs.push_back(move(d));
}

/** **************************************************************************
* Setup class shared static configurations and initialize
*
* \param log         logger object for logging
* \param plugin_path plugin directory
*
* \return true if everything was properly initialized, false otherwise
*************************************************************************** */
bool Track::Init(log4cxx::LoggerPtr log, string plugin_path=""){

  _log = log;

  return true;
}