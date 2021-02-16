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

#include "OcvYoloDetection.h"

#include <functional>
#include <list>
#include <stdexcept>
#include <vector>
#include <utility>
#include <tuple>

#include <opencv2/core.hpp>

#include <log4cxx/xml/domconfigurator.h>

// MPF-SDK header files
#include <MPFAsyncVideoCapture.h>
#include <MPFImageReader.h>
#include <Utils.h>

#include "Cluster.h"
#include "Config.h"
#include "DetectionLocation.h"
#include "Track.h"


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
        // intersection over union tracking and assignment
        Track::assignDetections(trackClusters,
                                detectionClusters,
                                assignedTracks,
                                config.maxIOUDist,
                                config.maxClassDist,
                                config.maxKFResidual,
                                config.edgeSnapDist,
                                std::mem_fn(&DetectionLocation::iouDist));
        LOG_TRACE("IOU assignment complete");

        // feature-based tracking tracking and assignment
        Track::assignDetections(trackClusters,
                                detectionClusters,
                                assignedTracks,
                                config.maxFeatureDist,
                                config.maxClassDist,
                                config.maxKFResidual,
                                config.edgeSnapDist,
                                std::mem_fn(&DetectionLocation::featureDist));
        LOG_TRACE("Feature assignment complete");

        // center-to-center distance tracking and assignment
        Track::assignDetections(trackClusters,
                                detectionClusters,
                                assignedTracks,
                                config.maxCenterDist,
                                config.maxClassDist,
                                config.maxKFResidual,
                                config.edgeSnapDist,
                                std::mem_fn(&DetectionLocation::center2CenterDist));
        LOG_TRACE("Center2Center assignment complete");
        return assignedTracks;
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
                // TODO: Ask why this isn't assigned to anything
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
                // add 1st detection to track
                assignedTracks.back().add(std::move(detection));
                LOG_TRACE("created new track " << assignedTracks.back());

            }
        }

        for (auto &trackCluster : trackClusterList) {
            // check any tracks that didn't get a detection and use tracker to continue them if possible
            if (!config.mosseTrackerDisabled) {
                // track leftover have no detections in current frame, try tracking
                for (auto &track : trackCluster.members) {
                    cv::Rect2i predictedRect;
                    if (track.ocvTrackerPredict(frame, config.maxFrameGap, predictedRect)) {
                        // tracker returned something
                        if (track.testResidual(predictedRect, config.edgeSnapDist)
                                <= config.maxKFResidual) {
                            // TODO fix empty detection_properties and 0 confidence
                            track.add({
                                    config, frame, predictedRect, 0.0,
                                    track.back().getClassFeature(),
                                    track.back().getDFTFeature()});
                            track.kalmanCorrect(config.edgeSnapDist);
                        }
                    }
                }
            }
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
    auto pluginPath = GetRunDirectory() + "/OcvYoloDetection";

    log4cxx::xml::DOMConfigurator::configure(pluginPath + "/config/Log4cxxConfig.xml");
    logger_ = log4cxx::Logger::getLogger("OcvYoloDetection");
    LOG_DEBUG("Initializing OcvYoloDetector");

    modelsParser_.Init(pluginPath + "/models")
            .RegisterPathField("network_config", &ModelSettings::networkConfigFile)
            .RegisterPathField("names", &ModelSettings::namesFile)
            .RegisterPathField("weights", &ModelSettings::weightsFile)
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
        std::vector<std::vector<DetectionLocation>> detections = yoloNetwork.GetDetections(
                { Frame(imageReader.GetImage()) }, config);

        std::vector<MPFImageLocation> results;
        for (std::vector<DetectionLocation> &locationList : detections) {
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
        auto& yoloNetwork = GetYoloNetwork(job.job_properties, config);
        MPFAsyncVideoCapture videoCapture(job);

        while (true) {
            auto frameBatch = GetVideoFrames(videoCapture, config.frameBatchSize);
            if (frameBatch.empty()) {
                break;
            }
            LOG_TRACE("processing frames [" << frameBatch.front().idx << "..."
                                            << frameBatch.back().idx << "]");

            // look for new detections
            std::vector<std::vector<DetectionLocation>> detectionsVec
                    = yoloNetwork.GetDetections(frameBatch, config);

            assert(frameBatch.size() == detectionsVec.size());
            for (int i = 0; i < frameBatch.size(); i++) {
                ProcessFrameDetections(config, frameBatch.at(i), std::move(detectionsVec.at(i)),
                                       inProgressTracks,
                                       completedTracks);
            }
        }

        // convert any remaining active tracks to MPFVideoTracks
        for (Track &track : inProgressTracks) {
            completedTracks.push_back(Track::toMpfTrack(std::move(track)));
        }

        for (MPFVideoTrack &mpfTrack : completedTracks) {
            videoCapture.ReverseTransform(mpfTrack);
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
