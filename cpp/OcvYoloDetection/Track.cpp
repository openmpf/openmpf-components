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

#include "util.h"
#include "Track.h"

using namespace MPF::COMPONENT;

string format(cv::Mat m);

/** ***************************************************************************
 *
**************************************************************************** */
void Track::kalmanInit(const float t,
                       const float dt,
                       const cv::Rect2i &rec0,
                       const cv::Rect2i &roi,
                       const cv::Mat1f &rn,
                       const cv::Mat1f &qn){
  _kfPtr = unique_ptr<KFTracker>(new KFTracker(t,dt,rec0,roi,rn,qn));
}

/** **************************************************************************
*  Get a new DetectionLocation from an existing one based on a frame
*
* \param framePtr pointer to the frame with new tracking info
*
* \returns ptr to new location based on tracker's estimation
*
* \note    tracker is passed on to the new location on success
*
**************************************************************************** */
 bool Track::ocvTrackerPredict(const Frame &frame, const long maxFrameGap, cv::Rect2i &prediction){

  if(_ocvTrackerPtr.empty()){   // initialize a new tracker if we don't have one already
    cv::Rect2i bbox    = locations.back().getRect();
    cv::Rect2i overlap =  bbox & cv::Rect2i(0,0,locations.back().frame.bgr.cols -1,
                                                locations.back().frame.bgr.rows -1);
     if(overlap.width > 1 && overlap.height > 1){
       _ocvTrackerPtr = cv::TrackerMOSSE::create();                            // could try different trackers here. e.g. cv::TrackerKCF::create();
       _ocvTrackerPtr->init(locations.back().frame.bgr, bbox);                 LOG_TRACE("tracker created for " << (MPFImageLocation)locations.back());
       _ocvTrackerStartFrameIdx = frame.idx;
    }else{                                                                     LOG_TRACE("can't create tracker created for " << (MPFImageLocation)locations.back());
      return false;
    }
  }

  if(frame.idx -_ocvTrackerStartFrameIdx <= maxFrameGap){
    cv::Rect2d pred;
    if(_ocvTrackerPtr->update(frame.bgr, pred)){
      prediction.x      = static_cast<int>(pred.x      + 0.5);                 // handle rounding
      prediction.y      = static_cast<int>(pred.y      + 0.5);
      prediction.width  = static_cast<int>(pred.width  + 0.5);
      prediction.height = static_cast<int>(pred.height + 0.5);                 LOG_TRACE("tracking " << (MPFImageLocation)locations.back() << " to " << prediction);
      return true;
    }else{                                                                     LOG_TRACE("could not track " << (MPFImageLocation)locations.back() << " to new location");
    }
  }                                                                            LOG_TRACE("extrapolation tracking stopped" << (MPFImageLocation)locations.back() << " frame gap = " << frame.idx - _ocvTrackerStartFrameIdx << " > " <<  maxFrameGap);
  return false;
}

/** **************************************************************************

*************************************************************************** */
const cv::Rect2i Track::predictedBox() const {
  if(_kfPtr){
    return _kfPtr->predictedBBox();
  }else{
    return locations.back().getRect();
  }
}

/** **************************************************************************
 * Advance Kalman filter state to predict next bbox at time t
 *
 * \param t time in sec to which kalamn filter state is advanced to
 *
*************************************************************************** */
void Track::kalmanPredict(const float t, const float edgeSnap){
  if(_kfPtr){
    _kfPtr->predict(t);

    // make frame edges "sticky"
    _kfPtr->setStatePreFromBBox(
      snapToEdges(locations.back().getRect(),_kfPtr->predictedBBox(),
                  locations.back().frame.bgr.size(), edgeSnap));           LOG_TRACE("kf pred:" << locations.back().getRect() << " => " << _kfPtr->predictedBBox());
  }
}

/** **************************************************************************
 * apply kalman correction to tail detection using tail's measurement
*************************************************************************** */
void Track::kalmanCorrect(const float edgeSnap){
  if(_kfPtr){                                                                  LOG_TRACE("kf meas:" << locations.back().getRect());
    _kfPtr->correct(locations.back().getRect());

    _kfPtr->setStatePostFromBBox(
      snapToEdges(locations.back().getRect(),_kfPtr->correctedBBox(),
                  locations.back().frame.bgr.size(), edgeSnap));

    locations.back().setRect(_kfPtr->correctedBBox());                         LOG_TRACE("kf corr:" << locations.back().getRect());
  }
}

/** **************************************************************************
*************************************************************************** */
float Track::testResidual(const cv::Rect2i bbox, const float edgeSnap) const{
  if(_kfPtr){
    return _kfPtr->testResidual(bbox, edgeSnap);
  }else{
    return 0.0;
  }
}
