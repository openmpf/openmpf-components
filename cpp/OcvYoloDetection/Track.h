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

#include <algorithm>
#include <list>
#include <memory>
#include <ostream>
#include <utility>
#include <unordered_set>
#include <vector>

#include <log4cxx/logger.h>

#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>

// 3rd party code for solving assignment problem
#include <dlib/matrix.h>
#include <dlib/optimization/max_cost_assignment.h>

#include <MPFDetectionObjects.h>

#include "Config.h"
#include "DetectionLocation.h"
#include "KFTracker.h"
#include "Cluster.h"


class Track {

public:
    DetectionLocation& front();

    const DetectionLocation& front() const;

    DetectionLocation& back();

    const DetectionLocation& back() const;

    void add(DetectionLocation detectionLocation);

    // Not a member function so we have the ability to move (rather than copy) parts of track in
    // to the MPFVideoTrack
    static MPF::COMPONENT::MPFVideoTrack toMpfTrack(Track track);


    /// predict a new detection from an exiting one using a tracker
    bool ocvTrackerPredict(const Frame &frame, long maxFrameGap, cv::Rect2i &prediction);

    /// release tracker so it can be reinitialized
    void releaseOCVTracker() { ocvTracker_.release(); }

    void kalmanPredict(float t, float edgeSnap);

    void kalmanCorrect(float edgeSnap);

    float testResidual(const cv::Rect2i &bbox, float edgeSnap) const;

    cv::Rect2i predictedBox() const;

    // TODO use words for parameters
    void kalmanInit(float t,
                    float dt,
                    const cv::Rect2i &rec0,
                    const cv::Rect2i &roi,
                    const cv::Mat1f &rn,
                    const cv::Mat1f &qn);


    // TODO comment was never accurate

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
    template<typename TCostFunc>
    static void assignDetections(std::vector<Track> &tracks,
                                 std::vector<DetectionLocation> &detections,
                                 std::vector<Track> &assignedTracks,
                                 const float maxCost,
                                 const float maxKFResidual,
                                 const float edgeSnap,
                                 TCostFunc&& costFunc) {

        if (tracks.empty() || detections.empty() || maxCost <= 0.0) {
            // nothing to do
            return;
        }

        dlib::matrix<long> costs = getCostMatrix(tracks, detections, maxCost, maxKFResidual,
                                                 std::forward<TCostFunc>(costFunc));

        // solve cost matrix, track av[track] get assigned detection[av[track]]
        // Track i's assignment is assignments[i]
        std::vector<long> assignments = dlib::max_cost_assignment(costs);
        LOG_TRACE("solved assignment vec[" << assignments.size() << "] = " << assignments);

        std::vector<Track> unassignedTracks;
        std::unordered_set<int> assignedDetectionIdxs;

        for (int trackIdx = 0; trackIdx < tracks.size(); ++trackIdx) {
            auto& track = tracks[trackIdx];
            int assignedDetectionIdx = assignments.at(trackIdx);
            // dlib required the cost matrix to be square, so columns may not refer to actual
            // detections.
            bool isValidDetection = assignedDetectionIdx < detections.size();
            // don't do assignments that are too costly (i.e. new track needed)
            if (!isValidDetection || costs(trackIdx, assignedDetectionIdx) == 0) {
                unassignedTracks.push_back(std::move(track));
                continue;
            }
            LOG_TRACE("assigning det "
                  << detections[assignedDetectionIdx] << " to track " << track
                  << " with residual:"
                  << detections.at(assignedDetectionIdx).kfResidualDist(track) << " cost:"
                  << (INT_MAX - costs(trackIdx, assignedDetectionIdx)) / 1.0E9);
            track.releaseOCVTracker();
            assignedDetectionIdxs.insert(assignedDetectionIdx);
            track.add(std::move(detections[assignedDetectionIdx]));

            track.kalmanCorrect(edgeSnap);
            assignedTracks.push_back(std::move(track));
        }
        tracks = std::move(unassignedTracks);

        // Shift unassigned detections to the front of the vector before erasing.
        int detectionIdx = 0;
        for (int i = 0; i < detections.size(); ++i) {
            if (assignedDetectionIdxs.count(i) == 0) {
                if (detectionIdx != i) {
                    detections[detectionIdx] = std::move(detections[i]);
                }
                detectionIdx++;
            }
        }
        detections.erase(detections.begin() + detectionIdx, detections.end());
    }


    // TODO comment was never accurate

    /** ****************************************************************************
    *   Compute cost vector and solve it to give detection to track assignment matrix
    *
    *

    * \returns assignment vector am[track,detection] with dim (# tracks x # detections)
    *          if am[x,y]==0 then detection[y] should be assigned to track[x]
    *
    ***************************************************************************** */
    template<typename TCostFunc>
    static void assignDetections(std::vector<Cluster<Track>> &trackClusterList,
                                 std::vector<Cluster<DetectionLocation>> &detectionClusterList,
                                 std::vector<Track> &assignedTracks,
                                 const float maxCost,
                                 const float maxClassDist,
                                 const float maxKFResidual,
                                 const float edgeSnap,
                                 TCostFunc&& costFunc) {
        for (auto &trackCluster : trackClusterList) {
            if (trackCluster.members.empty()) {
                continue;
            }
            for (auto &detectionCluster : detectionClusterList) {
                if (detectionCluster.members.empty()) {
                    continue;
                }
                float distance = cosDist(trackCluster.averageFeature,
                                         detectionCluster.averageFeature);
                if (distance < maxClassDist) {
                    Track::assignDetections(trackCluster.members,
                                            detectionCluster.members,
                                            assignedTracks,
                                            maxCost,
                                            maxKFResidual,
                                            edgeSnap,
                                            std::forward<TCostFunc>(costFunc));
                }
            }
        }
    }

#ifdef KFDUMP_STATE
    void             kalmanDump(string filename){if(_kfPtr) _kfPtr->dump(filename);}
#endif

    /// get class feature vector for last detection
    cv::Mat getClassFeature() const { return back().getClassFeature(); }


    friend std::ostream &operator<<(std::ostream &out, const Track &t);

private:
    /// list of locations making up track
    std::vector<DetectionLocation> locations_;

    /// openCV tracker to help bridge gaps when detector fails
    cv::Ptr<cv::Tracker> ocvTracker_;

    /// frame index at which the tracker was initialized
    size_t ocvTrackerStartFrameIdx_ = 0;

    std::unique_ptr<KFTracker> kalmanFilterTracker_;


    template<typename TCostFunc>
    static dlib::matrix<long> getCostMatrix(std::vector<Track> &tracks,
                                            std::vector<DetectionLocation> &detections,
                                            float maxCost,
                                            float maxKFResidual,
                                            TCostFunc&& costFunc) {
        int matSize = std::max(tracks.size(), detections.size());
        // dlib::max_cost_assignment requires a square matrix, so some entries will be zero'ed out.
        // Each row is a track and each column is detection.
        // costs(trackIdx, detectionIdx) = cost of assigning detection to track
        dlib::matrix<long> costs = dlib::zeros_matrix<long>(matSize, matSize);

        // fill in actual costs for non-dummy entries
        for (int trackIdx = 0; trackIdx < tracks.size(); ++trackIdx) {
            auto& track = tracks[trackIdx];

            for (int detectionIdx = 0; detectionIdx < detections.size(); ++detectionIdx) {
                auto& detection = detections[detectionIdx];

                if (track.back().frame.idx < detection.frame.idx
                        // must produce a reasonable normalized residual
                        && detection.kfResidualDist(track) <= maxKFResidual) {
                    float cost = costFunc(detection, track);
                    // dlib::max_cost_assignment requires the matrix either be ints or longs.
                    // We also need a way to invert the costs because there is no min version of
                    // dlib::max_cost_assignment.
                    long longCost = cost <= maxCost
                            ? (INT_MAX - static_cast<long>(1.0E9 * cost))
                            : 0;
                    costs(trackIdx, detectionIdx) = longCost;
                }
            }
        }
        LOG_TRACE("cost matrix[tr=" << costs.nr() << ",det=" << costs.nc() << "]: "
                                    << dformat(costs));
        return costs;
    }
};

std::ostream &operator<<(std::ostream &out, const Track &t);


#endif
