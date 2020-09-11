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

#ifndef OCVYOLODETECTION_TRACK_H
#define OCVYOLODETECTION_TRACK_H

#include <log4cxx/logger.h>
#include <opencv2/tracking.hpp>

// 3rd party code for solving assignment problem
#include <dlib/matrix.h>
#include <dlib/optimization/max_cost_assignment.h>

#include "types.h"
#include "Config.h"
#include "DetectionLocation.h"
#include "KFTracker.h"

namespace MPF{
  namespace COMPONENT{

    using namespace std;

    class Track{

      public:
        DetectionLocationList   locations;                    ///< vector of locations making up track

        bool ocvTrackerPredict(const Frame    &frame,
                               const long      maxFrameGap,
                               cv::Rect2i     &prediction);   ///< predict a new detection from an exiting one using a tracker
        void releaseOCVTracker() {_ocvTrackerPtr.release();}  ///< release tracker so it can be reinitialized

        void             kalmanPredict(const float t, const float edgeSnap);
        void             kalmanCorrect(const float edgeSnap);
        float            testResidual(const cv::Rect2i bbox, const float edgeSnap) const;
        const cv::Rect2i predictedBox() const;
        void             kalmanInit(const float t,
                                    const float dt,
                                    const cv::Rect2i &rec0,
                                    const cv::Rect2i &roi,
                                    const cv::Mat1f &rn,
                                    const cv::Mat1f &qn);

      typedef float (DetectionLocation::*DetectionLocationCostFunc)(const Track &tr) const; ///< cost member-function pointer type

      template<DetectionLocationCostFunc COST_FUNC>
      static void  assignDetections(TrackList               &tracks,
                                    DetectionLocationList   &detections,
                                    TrackList               &assignedTracks,
                                    const float              maxCost,
                                    const float              maxClassDist,
                                    const float              maxKFResidual,
                                    const float              edgeSnap); ///< determine assign detections to tracks

        #ifdef KFDUMP_STATE
          void             kalmanDump(string filename){if(_kfPtr) _kfPtr->dump(filename);}
        #endif

        Track(){};
        Track(Track &&t):locations(move(t.locations)),
                         _ocvTrackerPtr(move(t._ocvTrackerPtr)),
                         _ocvTrackerStartFrameIdx(t._ocvTrackerStartFrameIdx),
                         _kfPtr(move(t._kfPtr)){}                    ///< move constructor

        friend ostream& operator<< (ostream& out, const Track& t);

      private:

        cv::Ptr<cv::Tracker>    _ocvTrackerPtr;                ///< openCV tracker to help bridge gaps when detector fails
        size_t                  _ocvTrackerStartFrameIdx;      ///< frame index at which the tracker was initialized

        unique_ptr<KFTracker>   _kfPtr;                        ///< kalman filter tracker
    };

    /** **************************************************************************
    *   Dump MPF::COMPONENT::Track to a stream
    *************************************************************************** */
    inline
    ostream& operator<< (ostream& out, const Track& t) {
      out << "<f"   << t.locations.front().frame.idx << (MPFImageLocation)(t.locations.front())
          << "...f" << t.locations.back().frame.idx  << (MPFImageLocation)(t.locations.back())
          << ">("<<t.locations.size()<<")";
      return out;
    }

    /** ****************************************************************************
     *  print out dlib matrix on a single line
     *
     * \param   m matrix to serialize to single line string
     * \returns single line string representation of matrix
     *
    ***************************************************************************** */
    template<typename T>
    string dformat(dlib::matrix<T> m){
      stringstream ss;
      ss << "{";
      for(size_t r=0;r<m.nr();r++){
        for(size_t c=0;c<m.nc();c++){
          ss << m(r,c);
          if(c != m.nc()-1) ss << ",";
        }
        if(r != m.nr()-1) ss << "; ";
      }
      ss << "}";
      string str = ss.str();
      return str;
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
    template<DetectionLocationCostFunc COST_FUNC>
    void Track::assignDetections(TrackList                &tracks,
                                DetectionLocationList    &detections,
                                TrackList                &assignedTracks,
                                const float               maxCost,
                                const float               maxClassDist,
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
            if(    det.classDist(track)   <= maxClassDist                      // must be of a class similar to track's tail to be considered
                && det.kfResidualDist(track) <= maxKFResidual ){               // must produce a reasonable normalized residual
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
          trItr->releaseOCVTracker();
         // trItr->locations.back().frame.bgr.release();
          trItr->locations.splice(trItr->locations.end(), detections, detItrs.at(av[t]));
          trItr->kalmanCorrect(edgeSnap);
          assignedTracks.splice(assignedTracks.end(), tracks, trItr++);

        }else{
          trItr++;
        }
        t++;
      }
    }
  }
}

#endif