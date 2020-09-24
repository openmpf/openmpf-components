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

#include "OcvYoloDetection.h"

#include <algorithm>
#include <stdexcept>
#include <limits.h>
#include <chrono>

#include <log4cxx/xml/domconfigurator.h>

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QString>

// MPF-SDK header files
#include "Utils.h"
#include "MPFSimpleConfigLoader.h"
#include "detectionComponentUtils.h"

#include "types.h"
#include "Config.h"
#include "Cluster.h"

using namespace MPF::COMPONENT;

/** ****************************************************************************
*  Initialize Yolo detector module by setting up paths and reading configs
*  configuration variables are turned into environment variables for late
*  reference
*
* \returns   true on success
***************************************************************************** */
bool OcvYoloDetection::Init() {
    Config::pluginPath = GetRunDirectory() + "/OcvYoloDetection";
    Config::configPath = Config::pluginPath + "/config";

    log4cxx::xml::DOMConfigurator::configure(Config::configPath + "/Log4cxxConfig.xml");
    Config::log = log4cxx::Logger::getLogger("OcvYoloDetection");              LOG_DEBUG("Initializing OcvYoloDetector");

    // read config file and create or update any missing env variables
    string configParamsPath = Config::configPath + "/mpfOcvYoloDetection.ini";
    QHash<QString,QString> params;
    if(LoadConfig(configParamsPath, params)) {                                 LOG_ERROR( "Failed to load the OcvYoloDetection config from: " << configParamsPath);
        return false;
    }                                                                          LOG_TRACE("read config file:" << configParamsPath);

    Properties props;
    for(auto p = params.begin(); p != params.end(); p++){
      const string key = p.key().toStdString();
      const string val = p.value().toStdString();                              LOG_TRACE("Config    Vars:" << key << "=" << val);
      const char* env_val = getenv(key.c_str());                               LOG_TRACE("Verifying ENVs:" << key << "=" << env_val);
      if(env_val == NULL){
        if(setenv(key.c_str(), val.c_str(), 0) !=0){                           LOG_ERROR("Failed to convert config to env variable: " << key << "=" << val);
          return false;
        }
      }else if(string(env_val) != val){                                        LOG_INFO("Keeping existing env variable:" << key << "=" << string(env_val));
      }
    }

    bool detectionLocationInitalizedOK = DetectionLocation::Init();

    //Properties fromEnv;
    int cudaDeviceId   = getEnv<int> ({},"CUDA_DEVICE_ID", -1);
    bool fallbackToCPU = getEnv<bool>({},"FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", true);
    bool defaultCudaDeviceOK = DetectionLocation::loadNetToCudaDevice(cudaDeviceId) || fallbackToCPU;

    return    detectionLocationInitalizedOK
           && defaultCudaDeviceOK;


}

/** ****************************************************************************
* Clean up and release any created detector objects
*
* \returns   true on success
***************************************************************************** */
bool OcvYoloDetection::Close() {
    return true;
}

/** ****************************************************************************
* Read an image and get object detections and features
*
* \param          job     MPF Image job
*
* \returns  locations collection to which detections have been added
*
***************************************************************************** */
MPFImageLocationVec OcvYoloDetection::GetDetections(const MPFImageJob   &job) {

  try {                                                                        LOG_DEBUG( "[" << job.job_name << "Data URI = " << job.data_uri);
    Config cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS){
      throw MPFDetectionException(cfg.lastError,"failed to parse image job configuration parameters");
    }
    DetectionLocation::loadNetToCudaDevice(cfg.cudaDeviceId);

    FrameVec framesVec = {cfg.getImageFrame()};  // only do one frame at a time for now
    DetectionLocationListVec detections = DetectionLocation::createDetections(cfg, framesVec);
    assert(framesVec.size() == detections.size());

    MPFImageLocationVecVec locationsVec;
    for(unsigned i=0; i<framesVec.size(); i++){                                   LOG_DEBUG( "[" << job.job_name << "] Number of detections = " << detections[i].size());
      MPFImageLocationVec locations;
      for(auto &det:detections[i]){
        MPFImageLocation loc = det;
        cfg.ReverseTransform(loc);
        locations.push_back(loc);
      }
      detections[i].clear();
      locationsVec.push_back(locations);
    }
    return locationsVec[0];   // only return the one for now

  }catch(const runtime_error& re){
    LOG_FATAL( "[" << job.job_name << "] runtime error: " << re.what());
    throw MPFDetectionException(re.what());
  }catch(const exception& ex){
    LOG_FATAL( "[" << job.job_name << "] exception: " << ex.what());
    throw MPFDetectionException(ex.what());
  }catch (...){
    LOG_FATAL( "[" << job.job_name << "] unknown error");
    throw MPFDetectionException(" Unknown error processing image job");
  }
                                                                               LOG_DEBUG("[" << job.job_name << "] complete.");
}

/** ****************************************************************************
* Convert track (list of detection ptrs) to an MPFVideoTrack object
*
* \param [in,out]  tracks  Tracks collection to which detections will be added
*
* \returns MPFVideoTrack object resulting from conversion
*
* \note detection pts are released on conversion and confidence is assigned
*       as the average of the detection confidences
*
***************************************************************************** */
MPFVideoTrack OcvYoloDetection::_convert_track(Track &track){

  MPFVideoTrack mpf_track;
  mpf_track.start_frame = track.locations.front().frame.idx;
  mpf_track.stop_frame  = track.locations.back().frame.idx;

  #ifdef KFDUMP_STATE
    stringstream ss;
    ss << &track << ".csv";
    string filename = ss.str();
    track.kalmanDump(filename);
    mpf_track.detection_properties["kf_id"] = filename;
  #endif

  for(auto &det : track.locations){
    mpf_track.confidence += det.confidence;
    mpf_track.frame_locations.insert(mpf_track.frame_locations.end(),{det.frame.idx,det});
  }
  mpf_track.confidence /= static_cast<float>(track.locations.size());
  track.locations.clear();

  return mpf_track;
}

/** ****************************************************************************
* Read frames from a video, get object detections and make tracks
*
* \param          job     MPF Video job
*
* \returns   Tracks collection to which detections have been added
*
***************************************************************************** */
MPFVideoTrackVec OcvYoloDetection::GetDetections(const MPFVideoJob &job){

  MPFVideoTrackVec  mpf_tracks;
  TrackList         tracks;

  try{
    Config cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS){
      throw MPFDetectionException(cfg.lastError,"failed to parse video job configuration parameters");
    }

    DetectionLocation::loadNetToCudaDevice(cfg.cudaDeviceId);

    FrameVec framesVec = cfg.getVideoFrames(cfg.frameBatchSize);            // get initial batch of frames
    do {                                                                       LOG_TRACE(".");
                                                                               LOG_TRACE("processing frames [" << framesVec.front().idx << "..." << framesVec.back().idx << "]");
      // do something clever with frame skipping here ?
      // detectTrigger++;
      // detectTrigger = detectTrigger % (cfg.detFrameInterval + 1);

      #ifndef NDEBUG
      auto start_time = chrono::high_resolution_clock::now();
      #endif
      DetectionLocationListVec detectionsVec = DetectionLocation::createDetections(cfg, framesVec); // look for new detections
      #ifndef NDEBUG
      auto end_time = chrono::high_resolution_clock::now();
      double time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() * 1e-9;
      LOG_DEBUG("Detection time:  " << fixed << setprecision(5) << time_taken << "[sec] for " << framesVec.size() << " frames");
      LOG_DEBUG("Detection  speed: " << framesVec.size() / time_taken << "[fps]");
      #endif

      #ifndef NDEBUG
      start_time = chrono::high_resolution_clock::now();
      #endif
      assert(framesVec.size() == detectionsVec.size());
      for(unsigned i=0; i<framesVec.size(); i++){

                                                                          LOG_TRACE( detectionsVec[i].size() <<" detections to be matched to " << tracks.size() << " tracks");
        TrackList assignedTracks;          // tracks that were assigned a detection in the current frame

        // group detections according to class features
        auto detClusterList = DetectionCluster::cluster(detectionsVec[i],cfg.maxClassDist);
        assert(detectionsVec[i].empty());

        //group tracks according to class features
        auto trackClusterList = TrackCluster::cluster(tracks,cfg.maxClassDist);
        assert(tracks.empty());

        // intersection over union tracking and assignment
        Track::assignDetections<&DetectionLocation::iouDist>(trackClusterList,detClusterList,assignedTracks,cfg.maxIOUDist,cfg.maxClassDist,cfg.maxKFResidual,cfg.edgeSnapDist);
        LOG_TRACE("IOU assignment complete");

        // feature-based tracking tracking and assignment
        Track::assignDetections<&DetectionLocation::featureDist>(trackClusterList,detClusterList,assignedTracks,cfg.maxFeatureDist,cfg.maxClassDist,cfg.maxKFResidual,cfg.edgeSnapDist);
        LOG_TRACE("Feature assignment complete");

        // center-to-center distance tracking and assignment
        Track::assignDetections<&DetectionLocation::center2CenterDist>(trackClusterList,detClusterList,assignedTracks,cfg.maxCenterDist,cfg.maxClassDist,cfg.maxKFResidual,cfg.edgeSnapDist);
        LOG_TRACE("Center2Center assignment complete");

                                                                                    // LOG_TRACE( detectionsVec[i].size() <<" detections left for new tracks");
        // any detection not assigned up to this point becomes a new track
        for(auto& detCluster : detClusterList){
          auto detItr = detCluster.members.begin();
          while(detItr != detCluster.members.end()){             // make any unassigned detections into new tracks
            detItr->getDFTFeature();                             // start of tracks always get feature calculated
            assignedTracks.push_back(Track());                                      // create new track
            if(!cfg.kfDisabled){                                                    // initialize kalman tracker
              assignedTracks.back().kalmanInit(detItr->frame.time,
                                              detItr->frame.timeStep,
                                              detItr->getRect(),
                                              detItr->frame.getRect(),
                                              cfg.RN,cfg.QN);
            }
            assignedTracks.back().locations.splice(assignedTracks.back().locations.end(), detCluster.members, detItr++); // add 1st detection to track

                                                        LOG_TRACE("created new track " << assignedTracks.back());
          }
          assert(detCluster.members.empty());                                  // all detections should be removed at this point
        }

        for(auto& trackCluster : trackClusterList){
          if(!cfg.mosseTrackerDisabled){                                         // check any tracks that didn't get a detection and use tracker to continue them if possible
            cv::Rect2i pred;
            for(auto &track : trackCluster.members){                                           // track leftover have no detections in current frame, try tracking
              if(track.ocvTrackerPredict(framesVec[i],cfg.maxFrameGap, pred)){   // tracker returned something
                if(track.testResidual(pred, cfg.edgeSnapDist) <= cfg.maxKFResidual){
                  auto& prevTailDet = track.locations.back();
                  track.locations.push_back(
                    DetectionLocation(cfg, framesVec[i], pred, 0.0,
                                      track.locations.back().getClassFeature(),
                                      track.locations.back().getDFTFeature()));
                  track.kalmanCorrect(cfg.edgeSnapDist);
                }
              }
            }
          }
          assignedTracks.splice(assignedTracks.end(), trackCluster.members);
          assert(trackCluster.members.empty());
        }
        tracks.splice(tracks.end(), assignedTracks);
        assert(assignedTracks.empty());

        // remove and convert any tracks too far in the past from the active list
        tracks.remove_if([&](Track& track){
          if(framesVec[i].idx - track.locations.back().frame.idx > cfg.maxFrameGap){       LOG_TRACE("dropping old track: " << track);
            mpf_tracks.push_back(_convert_track(track));
            return true;
          }
          return false;
        });

        // advance kalman predictions
        if(! cfg.kfDisabled){
          for(auto &track:tracks){
            track.kalmanPredict(framesVec[i].time, cfg.edgeSnapDist);
          }
        }


      }                                                                        LOG_DEBUG( "[" << job.job_name << "] Number of tracks detected = " << tracks.size());
      #ifndef NDEBUG
      end_time = chrono::high_resolution_clock::now();
      time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() * 1e-9;
      LOG_DEBUG("Tracking time:  " << fixed << setprecision(5) << time_taken << "[sec] for " << framesVec.size() << " frames");
      LOG_DEBUG("Tracking  speed: " << framesVec.size() / time_taken << "[fps]");
      #endif

      #ifndef NDEBUG
      start_time = chrono::high_resolution_clock::now();
      #endif
      // grab next batch of frames
      framesVec = cfg.getVideoFrames(cfg.frameBatchSize);
      #ifndef NDEBUG
      end_time = chrono::high_resolution_clock::now();
      time_taken = chrono::duration_cast<chrono::nanoseconds>(end_time - start_time).count() * 1e-9;
      LOG_DEBUG("Frame read time:  " << fixed << setprecision(5) << time_taken << "[sec] for " << framesVec.size() << " frames");
      LOG_DEBUG("Read speed: " << framesVec.size() / time_taken << "[fps]");
      #endif

    } while(framesVec.size() > 0);

    // convert any remaining active tracks to MPFVideoTracks
    for(auto &track:tracks){
      mpf_tracks.push_back(_convert_track(track));
    }

    // reverse transform all mpf tracks
    for(auto &mpf_track:mpf_tracks){
      cfg.ReverseTransform(mpf_track);
    }

    // sort tracks by start frame, just to be nice :)
    sort(mpf_tracks.begin(),mpf_tracks.end(),
      [](const MPFVideoTrack &a, const MPFVideoTrack &b){
        return a.start_frame < b.start_frame;
      });

    #ifdef KFDUMP_STATE    // rename kalman filter debug outputs
    for(size_t i=0; i < mpf_tracks.size(); i++){
      stringstream ssOld, ssNew;
      ssOld << mpf_tracks[i].detection_properties["kf_id"];
      mpf_tracks[i].detection_properties.erase("kf_id");
      ssNew << setw(4) << setfill('0') << i << ".csv";
      rename(ssOld.str().c_str(), ssNew.str().c_str());                        LOG_TRACE(i << ":" << ssOld.str());
    }
    #endif

    return mpf_tracks;

  }catch(const runtime_error& re){                                             LOG_FATAL( "[" << job.job_name << "] runtime error: " << re.what());
    throw MPFDetectionException(re.what());
  }catch(const exception& ex){                                                 LOG_FATAL( "[" << job.job_name << "] exception: " << ex.what());
    throw MPFDetectionException(ex.what());
  }catch (...) {                                                               LOG_FATAL( "[" << job.job_name << "] unknown error");
    throw MPFDetectionException("Unknown error processing video job");
  }
                                                                               LOG_DEBUG("[" << job.job_name << "] complete.");
}


MPF_COMPONENT_CREATOR(OcvYoloDetection);
MPF_COMPONENT_DELETER();
