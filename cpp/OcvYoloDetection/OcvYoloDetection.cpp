/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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

#include <functional>
#include <list>
#include <stdexcept>
#include <vector>
#include <utility>

#include <opencv2/core.hpp>

// MPF-SDK header files
#include <MPFAsyncVideoCapture.h>
#include <MPFImageReader.h>
#include <Utils.h>

#include "Cluster.h"
#include "Config.h"
#include "DetectionLocation.h"
#include "Track.h"
#include "OcvYoloDetection.h"

using namespace MPF::COMPONENT;
using DetectionComponentUtils::GetProperty;

namespace {
    std::vector<Frame> GetVideoFrames(MPFAsyncVideoCapture &videoCapture, int numFrames) {
        double fps = videoCapture.GetFrameRate();
        std::vector<Frame> frames;
        frames.reserve(numFrames);
        for (int i = 0; i < numFrames; i++) {
            if (MPFFrame mpfFrame = videoCapture.Read()) {
                double time = mpfFrame.index / fps;
                frames.emplace_back(mpfFrame.index, time, 1 / fps,
                                    std::move(mpfFrame.data));
            }
            else {
                break;
            }
        }
        return frames;
    }


    std::vector<Track> AssignDetections(
            std::vector<Cluster<Track>> &trackClusters,
            std::vector<Cluster<DetectionLocation>> &detectionClusters,
            const Config &config) {

        // tracks that were assigned a detection in the current frame
        std::vector<Track> assignedTracks;
        if (config.maxIOUDist > 0) {
            // intersection over union tracking and assignment
            Track::assignDetections(trackClusters,
                                    detectionClusters,
                                    assignedTracks,
                                    config.maxIOUDist,
                                    config.maxClassDist,
                                    config.maxKFResidual,
                                    config.edgeSnapDist,
                                    std::mem_fn(&DetectionLocation::iouDist),
                                    "IoU",
                                    config.enableDebug);
            LOG_TRACE("IOU assignment complete");
        }

        if (config.maxFeatureDist > 0) {
            // feature-based tracking tracking and assignment
            Track::assignDetections(trackClusters,
                                    detectionClusters,
                                    assignedTracks,
                                    config.maxFeatureDist,
                                    config.maxClassDist,
                                    config.maxKFResidual,
                                    config.edgeSnapDist,
                                    std::mem_fn(&DetectionLocation::featureDist),
                                    "DFT",
                                    config.enableDebug);
            LOG_TRACE("Feature assignment complete");
        }

        if (config.maxCenterDist > 0) {
            // center-to-center distance tracking and assignment
            Track::assignDetections(trackClusters,
                                    detectionClusters,
                                    assignedTracks,
                                    config.maxCenterDist,
                                    config.maxClassDist,
                                    config.maxKFResidual,
                                    config.edgeSnapDist,
                                    std::mem_fn(&DetectionLocation::center2CenterDist),
                                    "C2C",
                                    config.enableDebug);
            LOG_TRACE("Center2Center assignment complete");
        }
        return assignedTracks;
    }


    void ExtendWithOcvTracker(const Config& config, const Frame &frame,
                              Cluster<Track> &trackCluster) {
        if (config.mosseTrackerDisabled) {
            return;
        }
        // tracks leftover have no detections in current frame, try tracking
        for (auto &track : trackCluster.members) {
            cv::Rect2i predictedRect;
            bool ocvTrackingSuccessful = track.ocvTrackerPredict(frame, config.maxFrameGap,
                                                                 predictedRect);
            if (!ocvTrackingSuccessful ||
                    track.testResidual(predictedRect, config.edgeSnapDist) > config.maxKFResidual) {
                continue;
            }

            float previousConfidence = track.back().confidence;
            auto previousProps = track.back().detection_properties;
            constexpr float gapFillPenalty = 0.00001;
            track.add({
                  config, frame, predictedRect,
                  // Slightly lower confidence to make sure this detection is never chosen as the exemplar.
                  previousConfidence - gapFillPenalty,
                  track.back().getClassFeature(),
                  track.back().getDFTFeature()});
            track.back().detection_properties = std::move(previousProps);
            track.back().detection_properties.emplace("FILLED_GAP", "TRUE");
            track.kalmanCorrect(config.edgeSnapDist);
        }
    }


    void ProcessFrameDetections(
            const Config &config,
            const Frame &frame,
            std::vector<DetectionLocation> &&detections,
            std::vector<Track> &inProgressTracks,
            std::vector<MPFVideoTrack> &completedTracks) {

        LOG_TRACE(detections.size() << " detections to be matched to " << inProgressTracks.size()
                    << " tracks");

        // group detections according to class features
        std::vector<Cluster<DetectionLocation>> detectionClusterList
                = clusterItems(std::move(detections), config.maxClassDist);

        //group tracks according to class features
        std::vector<Cluster<Track>> trackClusterList
                = clusterItems(std::move(inProgressTracks), config.maxClassDist);
        inProgressTracks.clear();

        // tracks that were assigned a detection in the current frame
        std::vector<Track> assignedTracks
                = AssignDetections(trackClusterList, detectionClusterList, config);

        // LOG_TRACE( detectionsVec[i].size() <<" detections left for new tracks");
        // any detection not assigned up to this point becomes a new track
        for (auto &detectionCluster : detectionClusterList) {
            // make any unassigned detections into new tracks
            for (auto& detection : detectionCluster.members) {
                // start of tracks always get feature calculated
                detection.getDFTFeature();
                // create new track
                assignedTracks.emplace_back();
                if (!config.kfDisabled) {
                    // initialize Kalman tracker
                    assignedTracks.back().kalmanInit(detection.frame.time,
                                                     detection.frame.timeStep,
                                                     detection.getRect(),
                                                     detection.frame.getRect(),
                                                     config.RN, config.QN);
                }
                // add first detection to track
                assignedTracks.back().add(std::move(detection));
                LOG_TRACE("Created new track " << assignedTracks.back());

            }
        }

        for (auto &trackCluster : trackClusterList) {
            // check any tracks that didn't get a detection and use tracker to continue them if possible
            ExtendWithOcvTracker(config, frame, trackCluster);
            // move tracks with no new detections to assigned tracks
            assignedTracks.insert(assignedTracks.end(),
                                  std::make_move_iterator(trackCluster.members.begin()),
                                  std::make_move_iterator(trackCluster.members.end()));
        }

        for (auto& track : assignedTracks) {
            auto gapSize = frame.idx - track.back().frame.idx;
            if (gapSize > config.maxFrameGap) {
                // remove and convert any tracks too far in the past from the active list
                completedTracks.push_back(Track::toMpfTrack(std::move(track)));
            }
            else {
                inProgressTracks.push_back(std::move(track));
            }
        }

        // advance Kalman predictions
        if (!config.kfDisabled) {
            for (auto &track : inProgressTracks) {
                track.kalmanPredict(frame.time, config.edgeSnapDist);
            }
        }
    }
} // end anonymous namespace


bool OcvYoloDetection::Init() {
    logger_ = log4cxx::Logger::getLogger("OcvYoloDetection");
    LOG_DEBUG("Initializing OcvYoloDetector");

    auto pluginPath = GetRunDirectory() + "/OcvYoloDetection";
    modelsParser_.Init(pluginPath + "/models")
            .RegisterPathField("ocvdnn_network_config", &ModelSettings::ocvDnnNetworkConfigFile)
            .RegisterPathField("ocvdnn_weights", &ModelSettings::ocvDnnWeightsFile)
            .RegisterPathField("names", &ModelSettings::namesFile)
            .RegisterOptionalPathField("confusion_matrix", &ModelSettings::confusionMatrixFile);

    LOG4CXX_INFO(logger_, "Initialized OcvYoloDetection component.");
    return true;
}

bool OcvYoloDetection::Close() {
    return true;
}


std::string OcvYoloDetection::GetDetectionType() {
   return "CLASS";
}


YoloNetwork& OcvYoloDetection::GetYoloNetwork(const Properties &jobProperties,
                                              const Config &config) {
    auto modelName = GetProperty(jobProperties, "MODEL_NAME", "tiny yolo");
    auto modelsDirPath = GetProperty(jobProperties, "MODELS_DIR_PATH", ".");
    auto modelSettings = modelsParser_.ParseIni(modelName, modelsDirPath + "/OcvYoloDetection");

    if (cachedYoloNetwork_) {
        if (cachedYoloNetwork_->IsCompatible(modelSettings, config)) {
            LOG4CXX_INFO(logger_, "Reusing cached model.");
            return *cachedYoloNetwork_;
        }
        else {
            // Reset to remove current network from memory before loading the new one.
            cachedYoloNetwork_.reset();
        }
    }
    cachedYoloNetwork_.reset(new YoloNetwork(std::move(modelSettings), config));
    return *cachedYoloNetwork_;
}


/** ****************************************************************************
* Read an image and get object detections and features
*
* \param          job     MPF Image job
*
* \returns  locations collection to which detections have been added
*
***************************************************************************** */
std::vector<MPFImageLocation> OcvYoloDetection::GetDetections(const MPFImageJob &job) {
    try {
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");
        Config config(job.job_properties);
        auto& yoloNetwork = GetYoloNetwork(job.job_properties, config);

        MPFImageReader imageReader(job);
        std::vector<MPFImageLocation> results;
        std::vector<Frame> frameBatch = { Frame(imageReader.GetImage()) };
        frameBatch.front().idx = 0;
        yoloNetwork.GetDetections(
          frameBatch,
          [&imageReader, &results]
          (std::vector<std::vector<DetectionLocation>> detectionsVec,
            std::vector<Frame>::const_iterator,
            std::vector<Frame>::const_iterator){
              for (std::vector<DetectionLocation> &locationList : detectionsVec) {
                  for (DetectionLocation &location : locationList) {
                      results.emplace_back(
                              location.x_left_upper,
                              location.y_left_upper,
                              location.width,
                              location.height,
                              location.confidence,
                              std::move(location.detection_properties));
                      imageReader.ReverseTransform(results.back());
                  }
              }
          },
          config);

        yoloNetwork.Cleanup(config);

        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << results.size()
                                << " detections.");
        return results;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }
}


/** ****************************************************************************
* Read frames from a video, get object detections and make tracks
*
* \param          job     MPF Video job
*
* \returns   Tracks collection to which detections have been added
*
***************************************************************************** */
std::vector<MPFVideoTrack> OcvYoloDetection::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Starting job");
        std::vector<MPFVideoTrack> completedTracks;
        std::vector<Track> inProgressTracks;

        Config config(job.job_properties);
        if(config.tritonEnabled && !config.mosseTrackerDisabled){
          config.mosseTrackerDisabled = true;
          LOG_WARN("MOSSE tracker is not supported with Triton, and has been disabled for this job");
        }
        auto& yoloNetwork = GetYoloNetwork(job.job_properties, config);

        MPFAsyncVideoCapture videoCapture(job);

       // place to hold frames till callbacks are done
       std::unordered_map<int, std::vector<Frame>> frameBatches;

        while (true) {
            auto tmp = GetVideoFrames(videoCapture, config.frameBatchSize);

            if (tmp.empty()) {
                yoloNetwork.Cleanup(config);
                break;
            }

            int frameBatchKey = tmp.back().idx;
            frameBatches.insert(std::make_pair(frameBatchKey, std::move(tmp)));

            LOG_TRACE("Processing frames [" << frameBatches.at(frameBatchKey).front().idx << "..."
                                            << frameBatches.at(frameBatchKey).back().idx << "]");
            yoloNetwork.GetDetections(
              frameBatches.at(frameBatchKey),

              // this callback gets called MULTIPLE times
              [&config, &frameBatches, &inProgressTracks, &completedTracks, frameBatchKey]
              (std::vector<std::vector<DetectionLocation>>&& detectionsVec,
               std::vector<Frame>::const_iterator begin,
               std::vector<Frame>::const_iterator end) {

                int backFrameIdx = (end - 1)->idx;
                int i = 0;
                for(std::vector<Frame>::const_iterator it = begin; it != end; ++it,++i){
                  ProcessFrameDetections(config, *it,
                                         std::move(detectionsVec.at(i)),
                                         inProgressTracks,
                                         completedTracks);
                }

                // last frame in batch, release frame batch
                if(frameBatchKey == backFrameIdx){
                  frameBatches.erase(frameBatchKey);
                }
              },

              config);
        }

        assert(("All frame batches should have been processed.", frameBatches.empty()));

        LOG_TRACE("Converting remaining active tracks to MPF tracks");
        // Convert any remaining active tracks to MPFVideoTracks. Remove any detections
        // that are below the confidence threshold, and drop empty tracks, if any.
        for (Track &track : inProgressTracks) {
            completedTracks.push_back(Track::toMpfTrack(std::move(track)));
        }

        std::vector<std::vector<MPFVideoTrack>::iterator> tracks_to_erase;
        for (std::vector<MPFVideoTrack>::iterator it = completedTracks.begin();
             it != completedTracks.end();
             it++) {
            MPFVideoTrack &mpfTrack = *it;
            // Remove detections below the confidence threshold.
            std::vector<int> detections_to_erase;
            for (const auto &loc : mpfTrack.frame_locations) {
                if (loc.second.confidence < config.confidenceThreshold) {
                    detections_to_erase.push_back(loc.first);
                }
            }
            for (int idx : detections_to_erase) {
                mpfTrack.frame_locations.erase(idx);
            }

            if (!mpfTrack.frame_locations.empty()) {
                // Adjust start and stop frames in case detections were removed at
                // the beginning or end of the track.
                mpfTrack.start_frame = mpfTrack.frame_locations.begin()->first;
                mpfTrack.stop_frame = mpfTrack.frame_locations.rbegin()->first;
                videoCapture.ReverseTransform(mpfTrack);
            }
            else {
                // The frame locations map is empty, so discard this track.
                // This is unlikely to happen, but we need to handle it just in case.
                tracks_to_erase.push_back(it);
            }
        }
        for (auto it : tracks_to_erase) {
            completedTracks.erase(it);
        }


        LOG4CXX_INFO(logger_, "[" << job.job_name << "] Found " << completedTracks.size()
                     << " tracks.");

        return completedTracks;
    }
    catch (...) {
        Utils::LogAndReThrowException(job, logger_);
    }
}


MPF_COMPONENT_CREATOR(OcvYoloDetection);
MPF_COMPONENT_DELETER();
