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
#include "Cluster.h"

namespace MPF{
  namespace COMPONENT{

    using namespace std;

    using assignmentCostFunc = float (DetectionLocation::*)(const Track &tr) const; ///< cost member-function pointer type

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

        template<assignmentCostFunc COST_FUNC>
        static void  assignDetections(TrackList               &tracks,
                                      DetectionLocationList   &detections,
                                      TrackList               &assignedTracks,
                                      const float              maxCost,
                                      const float              maxKFResidual,
                                      const float              edgeSnap); ///< determine assign detections to tracks

        template<assignmentCostFunc COST_FUNC, typename TrackClusterListType, typename DetectionClusterListType>
        static void  assignDetections(TrackClusterListType     &trkClusterList,
                                      DetectionClusterListType &detClusterList,
                                      TrackList                &assignedTracks,
                                      const float               maxCost,
                                      const float               maxClassDist,
                                      const float               maxKFResidual,
                                      const float               edgeSnap);  ///< determine assign detections in cluster to tracks in clusters
        #ifdef KFDUMP_STATE
          void             kalmanDump(string filename){if(_kfPtr) _kfPtr->dump(filename);}
        #endif

        const cv::Mat&       getClassFeature() const {return locations.back().getClassFeature();}      ///< get class feature vector for last detection

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

    ostream& operator<< (ostream& out, const Track& t);

    using TrackCluster          = Cluster<Track, &Track::getClassFeature, &cosDist>;
    using DetectionCluster      = Cluster<DetectionLocation, &DetectionLocation::getClassFeature, &cosDist>;
    using TrackClusterList      = list<TrackCluster>;
    using DetectionClusterList  = list<DetectionCluster>;

  }
}





#endif