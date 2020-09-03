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

#include <log4cxx/xml/domconfigurator.h>

#include <QFile>
#include <QFileInfo>
#include <QHash>
#include <QString>

// 3rd party code for solving assignment problem
#include <dlib/matrix.h>
#include <dlib/optimization/max_cost_assignment.h>

// MPF-SDK header files
#include "Utils.h"
#include "MPFSimpleConfigLoader.h"
#include "detectionComponentUtils.h"

#include "types.h"
#include "Config.h"

using namespace MPF::COMPONENT;

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
vector<long> OcvYoloDetection::_calcAssignmentVector(const TrackList               &tracks,
                                                     const DetectionLocationPtrVec &detections,
                                                     const float                    maxCost = INT_MAX,
                                                     const float                    maxClassDist = 0.5,
                                                     const float                    maxKFResidual = 4.0){
  vector<long> av; //assignment vector to return

  if(tracks.empty() || detections.empty()) return av;  // nothing to do

  // rows -> tracks, cols -> detections, but matrix has to be square!
  size_t n = max(tracks.size(),detections.size());
  dlib::matrix<long> costs = dlib::zeros_matrix<long>(n,n);

  // fill in actual costs for non-dummy entries
  size_t r = 0;
  intVec uniqueCosts;
  for(auto &track:tracks){
    for(size_t c=0; c<detections.size(); c++){
      if(track.back()->framePtr->idx < detections[c]->framePtr->idx){
        if(    detections[c]->classDist(track)   <= maxClassDist                      // must be of a class similar to track's tail to be considered
            && detections[c]->kfResidualDist(track) <= maxKFResidual ){               // must produce a reasonable normalized residual
          float cost = CALL_MEMBER_FUNC(*detections[c],COST_FUNC)(track);
          int iCost = ((cost <= maxCost) ? (INT_MAX - static_cast<int>(1.0E9 * cost)) : 0); //r*detections.size()+c);
          if(iCost != 0 ){
            while(find(uniqueCosts.begin(),uniqueCosts.end(),iCost) != uniqueCosts.end()) iCost--;  // cost have to be unique or dlib::max_cost_assignment() hangs...
            uniqueCosts.push_back(iCost);
          }
          costs(r,c) = iCost;                                                  //LOG_TRACE("C[" << r << "," << c << "]=" << cost << ":" << iCost);
        }
      }else{                                                                   LOG_TRACE("track back idx(" << track.back()->framePtr->idx <<") < detection idx(" << detections[c]->framePtr->idx << ")");
      }
    }
    r++;
  }                                                                            LOG_TRACE("cost matrix[tr=" << costs.nr() << ",det=" << costs.nc() << "]: " << dformat(costs));

  // solve cost matrix, track av[track] get assigned detection[av[track]]
  av = dlib::max_cost_assignment(costs);                                       LOG_TRACE( "solved assignment vec["<< av.size() << "] = "  << av);

  // drop dummy tracks used to pad cost matrix
  av.resize(tracks.size());

  // knock out assignments that are too costly (i.e. new track needed)
  for(long t=0; t<av.size(); t++){
    if(costs(t,av[t]) == 0 ){
      av[t] = -1;
    }
  }                                                                            LOG_TRACE( "modified assignment vec["<< av.size() << "] = "  << av);

  return av;
}

/** ****************************************************************************
*   Move detections to tails of tracks according to assignment matrix
*
* \param [in,out] tracks           list of tracks
* \param [in,out] detections       vector of detections
* \param          assignmentVector (v[track] <=> detection[v[track]]) with with -1 in skip assignment
* \param [in,out] assignedTracks   list of tracks that received a detection assignment
*
* \note detections that are assigned are removed from detections
*       and tracks that were assigned a detection are moved to assignedTracks
*
***************************************************************************** */
void OcvYoloDetection::_assignDetections2Tracks(TrackList                &tracks,
                                                vector<unique_ptr<DetectionLocation>>  &detections,
                                                const vector<long>       &assignmentVector,
                                                TrackList                &assignedTracks){


  assert(tracks.size() == assignmentVector.size());

  auto trIter=tracks.begin();
  for(long detIdx:assignmentVector){
    if(detIdx != -1){                                                  LOG_TRACE("assigning det: " << "f" << detections[detIdx]->framePtr->idx << " " <<  ((MPFImageLocation)*(detections[detIdx])) << " to track " << *trIter);
      trIter->releaseOCVTracker();
      trIter->push_back(move(detections.at(detIdx)));
      trIter->kalmanCorrect();
      assignedTracks.splice(assignedTracks.end(),tracks,trIter++);
    }else{
      trIter++;
    }
  }
  // remove detections that were assigned to tracks
  detections.erase(remove_if(detections.begin(),
                             detections.end(),
                             [](DetectionLocationPtr const& d){return !d;}),
                   detections.end());

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

    FramePtrVec framePtrs = cfg.getImageFrames(1);  // only do one frame at a time for now
    DetectionLocationPtrVecVec detectionsVec = DetectionLocation::createDetections(cfg, framePtrs);
    assert(framePtrs.size() == detectionsVec.size());

    MPFImageLocationVecVec locationsVec;
    for(unsigned i=0; i<framePtrs.size(); i++){                                LOG_DEBUG( "[" << job.job_name << "] Number of detections = " << detectionsVec[i].size());
      MPFImageLocationVec locations;
      for(auto &det:detectionsVec[i]){
        MPFImageLocation loc = *det;
        det.reset();                    // release frame object
        cfg.ReverseTransform(loc);
        locations.push_back(loc);
      }
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
  mpf_track.start_frame = track.front()->framePtr->idx;
  mpf_track.stop_frame  = track.back()->framePtr->idx;

  //stringstream start_feature;
  //stringstream stop_feature;
  //start_feature << track.front()->getDFTFeature();  // make sure we have computed features to serialize
  //stop_feature << track.back()->getDFTFeature();    //  for the start and end detections.
  //mpf_track.detection_properties["START_FEATURE"] = start_feature.str();
  //mpf_track.detection_properties["STOP_FEATURE"]  = stop_feature.str();

  #ifdef KFDUMP_STATE
    stringstream ss;
    ss << &track << ".csv";
    string filename = ss.str();
    track.kalmanDump(filename);
    mpf_track.detection_properties["kf_id"] = filename;
  #endif

  for(auto &det:track){
    mpf_track.confidence += det->confidence;
    mpf_track.frame_locations.insert(mpf_track.frame_locations.end(),{det->framePtr->idx,*det});
    det.reset();
  }
  mpf_track.confidence /= static_cast<float>(track.size());

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

    FramePtrVec framePtrs = cfg.getVideoFrames(cfg.frameBatchSize);            // get initial batch of frames
    do {                                                                       LOG_TRACE(".");
                                                                               LOG_TRACE("processing frames [" << framePtrs.front()->idx << "..." << framePtrs.back()->idx << "]");
      // do something clever with frame skipping here ?
      // detectTrigger++;
      // detectTrigger = detectTrigger % (cfg.detFrameInterval + 1);


      DetectionLocationPtrVecVec detectionsVec = DetectionLocation::createDetections(cfg, framePtrs); // look for new detections

      assert(framePtrs.size() == detectionsVec.size());
      for(unsigned i=0; i<framePtrs.size(); i++){

        // remove and archive any tracks too far in the past from the active list
        tracks.remove_if([&](Track& track){
          if(framePtrs[i]->idx - track.back()->framePtr->idx > cfg.maxFrameGap){       LOG_TRACE("dropping old track: " << track);
            mpf_tracks.push_back(_convert_track(track));
            return true;
          }
          return false;
        });

        DetectionLocationCostFunc iouCostFunc;
        // advance kalman predictions
        if(! cfg.kfDisabled){
          for(auto &track:tracks){
            track.kalmanPredict(framePtrs[i]->time);
          }
        }

        TrackList assignedTracks;          // tracks that were assigned a detection in the current frame

        if(!tracks.empty()){             // not all tracks were dropped
                                                                                          LOG_TRACE( detectionsVec[i].size() <<" detections to be matched to " << tracks.size() << " tracks");

          // calculate forbidden assignments based on classes


            vector<long> av;  // assignment vector i.e. av[track_idx] = det_idx

          // intersection over union tracking and assignment
          if(!detectionsVec[i].empty() && cfg.maxIOUDist > 0){
            if(! cfg.kfDisabled){
              av = _calcAssignmentVector<&DetectionLocation::kfIouDist>(tracks,detectionsVec[i],cfg.maxIOUDist,cfg.maxClassDist,cfg.maxKFResidual);
            }else{
              av = _calcAssignmentVector<&DetectionLocation::iouDist>(tracks,detectionsVec[i],cfg.maxIOUDist,cfg.maxClassDist,cfg.maxKFResidual);
            }
            _assignDetections2Tracks(tracks, detectionsVec[i], av, assignedTracks);     LOG_TRACE("IOU assignment complete");
          }

          // feature-based tracking tracking and assignment
          if(!detectionsVec[i].empty() && cfg.maxFeatureDist > 0){                                              LOG_TRACE( detectionsVec[i].size() <<" detections to be matched to " << tracks.size() << " tracks");
            av = _calcAssignmentVector<&DetectionLocation::featureDist>(tracks,detectionsVec[i],cfg.maxFeatureDist,cfg.maxClassDist,cfg.maxKFResidual);
            _assignDetections2Tracks(tracks, detectionsVec[i], av, assignedTracks);     LOG_TRACE("Feature assignment complete");
          }

          // center-to-center distance tracking and assignment
          if(!detectionsVec[i].empty() && cfg.maxCenterDist > 0){                                              LOG_TRACE( detectionsVec[i].size() <<" detections to be matched to " << tracks.size() << " tracks");
            av = _calcAssignmentVector<&DetectionLocation::center2CenterDist>(tracks,detectionsVec[i],cfg.maxCenterDist,cfg.maxClassDist,cfg.maxKFResidual);
            _assignDetections2Tracks(tracks, detectionsVec[i], av, assignedTracks);     LOG_TRACE("Center2Center assignment complete");
          }

        }

                                                                                    LOG_TRACE( detectionsVec[i].size() <<" detections left for new tracks");
        // any detection not assigned up to this point becomes a new track
        for(DetectionLocationPtr &det:detectionsVec[i]){                           // make any unassigned detections into new tracks
          det->getDFTFeature();                                       // start of tracks always get feature calculated

          assignedTracks.push_back(Track(cfg, move(det)));                         LOG_TRACE("created new track " << assignedTracks.back());
        }


        if(!cfg.mosseTrackerDisabled){
          // check any tracks that didn't get a detection and use tracker to continue them if possible
          for(auto &track:tracks){  // track left have no detections in current frame, try tracking
            DetectionLocationPtr detPtr = track.ocvTrackerPredict(framePtrs[i]);
            if(detPtr){  // tracker returned something
              if(detPtr->kfResidualDist(track) <= cfg.maxKFResidual){
                track.push_back(move(detPtr));              // add new location as tracks's tail
                track.kalmanCorrect();
              }
            }
          }
        }

        // put all the tracks that were assigned a detection back into active tracks list
        tracks.splice(tracks.end(),assignedTracks);

      }                                                                        LOG_DEBUG( "[" << job.job_name << "] Number of tracks detected = " << tracks.size());

      // grab next batch of frames
      framePtrs = cfg.getVideoFrames(cfg.frameBatchSize);
    } while(framePtrs.size() > 0);

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
