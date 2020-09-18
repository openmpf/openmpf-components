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


/** **************************************************************************
*   Dump MPF::COMPONENT::Track to a stream
*************************************************************************** */
ostream& MPF::COMPONENT::operator<< (ostream& out, const Track& t) {
  out << "<f"   << t.locations.front().frame.idx << (MPFImageLocation)(t.locations.front())
      << "...f" << t.locations.back().frame.idx  << (MPFImageLocation)(t.locations.back())
      << ">("<<t.locations.size()<<")";
  return out;
}

/** ****************************************************************************
*   Compute cost vector and solve it to give detection to track assignment matrix
*
* \param COST_FUNC     cost function to use
* \param TrackPtrList  list of existing tracks to consider for assignment
* \param detections    vector of detections that need assigned to tracks
* \param maxCost       maximum assignment cost, if exceeded the particular
*                      detection to track assignment will be removed from result
* \param maxClassDist  maximum class distance for track to detection
*

* \returns assignment vector am[track,detection] with dim (# tracks x # detections)
*          if am[x,y]==0 then detection[y] should be assigned to track[x]
*
***************************************************************************** */
template<assignmentCostFunc COST_FUNC>
void Track::assignDetections(TrackList                &tracks,
                              DetectionLocationList    &detections,
                              TrackList                &assignedTracks,
                              const float               maxCost,
                              const float               maxKFResidual,
                              const float               edgeSnap){

  if(tracks.empty() || detections.empty() || maxCost <=0.0 ) return;  // nothing to do

  // rows -> tracks, cols -> detections, but matrix has to be square!
  size_t n = max(tracks.size(),detections.size());
  dlib::matrix<long> costs = dlib::zeros_matrix<long>(n,n);

  // fill in actual costs for non-dummy entries
  size_t r = 0;
  intVec uniqueCosts;
  for(auto &track:tracks){
    size_t c = 0;
    for(auto &det:detections){
      if(track.locations.back().frame.idx < det.frame.idx){
        if(det.kfResidualDist(track) <= maxKFResidual ){               // must produce a reasonable normalized residual
          float cost = CALL_MEMBER_FUNC(det,COST_FUNC)(track);
          int iCost = ((cost <= maxCost) ? (INT_MAX - static_cast<int>(1.0E9 * cost)) : 0); //r*detections.size()+c);
          if(iCost != 0 ){
            while(find(uniqueCosts.begin(),uniqueCosts.end(),iCost) != uniqueCosts.end()) iCost--;  // cost have to be unique or dlib::max_cost_assignment() hangs...
            uniqueCosts.push_back(iCost);
          }
          costs(r,c) = iCost;                                                  //LOG_TRACE("C[" << r << "," << c << "]=" << cost << ":" << iCost);
        }
      }else{                                                                   LOG_TRACE("track back idx(" << track.locations.back().frame.idx <<") < detection idx(" << det.frame.idx << ")");
      }
      c++;
    }
    r++;
  }                                                                            LOG_TRACE("cost matrix[tr=" << costs.nr() << ",det=" << costs.nc() << "]: " << dformat(costs));

  // solve cost matrix, track av[track] get assigned detection[av[track]]
  vector<long> av = dlib::max_cost_assignment(costs);                          LOG_TRACE( "solved assignment vec["<< av.size() << "] = "  << av);

  // build association mapping
  vector<DetectionLocationList::iterator> detItrs;
  for(DetectionLocationList::iterator detItr=detections.begin(); detItr != detections.end(); detItr++){
    detItrs.push_back(detItr);
  }
  long t = 0;
  TrackList::iterator trItr = tracks.begin();
  while( trItr != tracks.end() ){
    if(costs(t,av[t]) != 0 ){                                         // don't do assignments that are too costly (i.e. new track needed)
                                                                      LOG_TRACE("assigning det " << *detItrs.at(av[t]) << " to track " << *trItr << " with residual:" << detItrs.at(av[t])->kfResidualDist(*trItr) << " cost:" << (INT_MAX - costs(t,av[t]))/1.0E9);
      trItr->releaseOCVTracker();
      const_cast<cv::Mat*>(&(trItr->locations.back().frame.bgr))->release();  // old tail's image no longer needec
      trItr->locations.splice(trItr->locations.end(), detections, detItrs.at(av[t]));
      trItr->kalmanCorrect(edgeSnap);
      assignedTracks.splice(assignedTracks.end(), tracks, trItr++);

    }else{
      trItr++;
    }
    t++;
  }
}

/** ***************************************************************************
* Some explicit template instantiations used elsewhere in project
* **************************************************************************** */
template void Track::assignDetections<&DetectionLocation::featureDist>      (TrackList&, DetectionLocationList&, TrackList&, const float, const float, const float);
template void Track::assignDetections<&DetectionLocation::iouDist>          (TrackList&, DetectionLocationList&, TrackList&, const float, const float, const float);
template void Track::assignDetections<&DetectionLocation::center2CenterDist>(TrackList&, DetectionLocationList&, TrackList&, const float, const float, const float);


/** ****************************************************************************
*   Compute cost vector and solve it to give detection to track assignment matrix
*
*

* \returns assignment vector am[track,detection] with dim (# tracks x # detections)
*          if am[x,y]==0 then detection[y] should be assigned to track[x]
*
***************************************************************************** */
template<typename TrackClusterListType, typename DetectionClusterListType, assignmentCostFunc COST_FUNC>
void Track::assignDetections(TrackClusterListType         &trkClusterList,
                             DetectionClusterListType     &detClusterList,
                             TrackList                    &assignedTracks,
                             const float                   maxCost,
                             const float                   maxClassDist,
                             const float                   maxKFResidual,
                             const float                   edgeSnap){
  for(auto& trCl : trkClusterList){
    for(auto& dtCl : detClusterList){
      if(   !trCl.members.empty()
          && !dtCl.members.empty()
          && cosDist(trCl.aveFeature,dtCl.aveFeature) < maxClassDist){
        Track::assignDetections<COST_FUNC>(trCl.members,dtCl.members,assignedTracks,maxCost,maxKFResidual,edgeSnap);
      }
    }
  }
}

/** ***************************************************************************
* Some explicit template instantiations used elsewhere in project
* **************************************************************************** */
template void Track::assignDetections<TrackClusterList, DetectionClusterList, &DetectionLocation::featureDist>      (TrackClusterList&, DetectionClusterList&, TrackList&, const float, const float, const float, const float);
template void Track::assignDetections<TrackClusterList, DetectionClusterList, &DetectionLocation::iouDist>          (TrackClusterList&, DetectionClusterList&, TrackList&, const float, const float, const float, const float);
template void Track::assignDetections<TrackClusterList, DetectionClusterList, &DetectionLocation::center2CenterDist>(TrackClusterList&, DetectionClusterList&, TrackList&, const float, const float, const float, const float);
