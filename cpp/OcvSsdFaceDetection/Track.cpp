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
Track::Track(unique_ptr<DetectionLocation> detPtr, const JobConfig &cfg){
  if(! cfg.kfDisabled){
    _kfPtr = unique_ptr<KFTracker>(
      new KFTracker(cfg.frameTimeInSec,
                    cfg.frameTimeStep,
                    detPtr->getRect(),
                    cv::Rect2i(0,0,cfg.bgrFrame.cols-1,cfg.bgrFrame.rows-1),
                    cfg.RN,
                    cfg.QN));
  }
  _locationPtrs.push_back(move(detPtr));
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
    cv::Rect2i bbox    = _locationPtrs.back()->getRect();
    cv::Rect2i overlap =  bbox & cv::Rect2i(0,0,_locationPtrs.back()->getBGRFrame().cols -1,
                                                _locationPtrs.back()->getBGRFrame().rows -1);
     if(overlap.width > 1 && overlap.height > 1){
       _trackerPtr = cv::TrackerMOSSE::create();                               // could try different trackers here. e.g. cv::TrackerKCF::create();
       _trackerPtr->init(_locationPtrs.back()->getBGRFrame(), bbox);           LOG4CXX_TRACE(_log,"tracker created for " << (MPFImageLocation)*_locationPtrs.back());
       _trackerStartFrameIdx = cfg.frameIdx;
    }else{                                                                     LOG4CXX_TRACE(_log,"can't create tracker created for " << (MPFImageLocation)*_locationPtrs.back());
      return nullptr;
    }
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
        cfg.frameTimeInSec,
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
*************************************************************************** */
void Track::kalmanPredict(float t){
  _kfPtr->predict(t);                                                          LOG4CXX_TRACE(_log,"kf pred:" << _locationPtrs.back()->getRect() << " => " << _kfPtr->predictedBBox());
}

/** **************************************************************************
 * apply kalman correction to tail detection
*************************************************************************** */
void Track::kalmanCorrect(){
  if(_kfPtr){                                                                  LOG4CXX_TRACE(_log,"kf meas:" << _locationPtrs.back()->getRect());
    _kfPtr->correct(_locationPtrs.back()->getRect());                          LOG4CXX_TRACE(_log,"kf corr:" << _locationPtrs.back()->getRect());
    back()->setRect(_kfPtr->correctedBBox());
  }
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