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
    bool defaultCudaDeviceOK = DetectionLocation::trySetCudaDevice(cudaDeviceId) || fallbackToCPU;

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
* \returns assignment vector am[track,detection] with dim (# tracks x # detections)
*          if am[x,y]==0 then detection[y] should be assigned to track[x]
*
***************************************************************************** */
template<DetectionLocationCostFunc COST_FUNC>
vector<long> OcvYoloDetection::_calcAssignmentVector(const TrackPtrList            &tracks,
                                                        const DetectionLocationPtrVec &detections,
                                                        const float                    maxCost = INT_MAX){
  vector<long> av; //assignment vector to return

  if(tracks.size() == 0 || detections.size() ==0) return av;  // nothing to do

  // rows -> tracks, cols -> detections, but matrix has to be square!
  size_t n = max(tracks.size(),detections.size());
  dlib::matrix<int> costs = dlib::zeros_matrix<int>(n,n);

  // fill in actual costs for non-dummy entries
  size_t r = 0;
  for(auto &track:tracks){
    for(size_t c=0; c<detections.size(); c++){
      if(track->back()->frameIdx < detections[c]->frameIdx){
        //float cost = CALL_MEMBER_FUNC(*(track->back()),COST_FUNC)(*detections[c]);
        float cost = CALL_MEMBER_FUNC(*detections[c],COST_FUNC)(*track);
        costs(r,c) = ((cost <= maxCost) ? (INT_MAX - static_cast<int>(1000.0f * cost)) : 0);
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
    if(costs(t,av[t]) == 0){
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
* \param          assignmentVector (v[track] <=> detection[v[track]]) with with -1 in
*                                   skip assignment
*
* \note detections that are assigned are removed from detections
*
***************************************************************************** */
void OcvYoloDetection::_assignDetections2Tracks(TrackPtrList             &tracks,
                                                   DetectionLocationPtrVec  &detections,
                                                   const vector<long>       &assignmentVector){

  long t=0;
  for(auto &track:tracks){
    if(assignmentVector[t] != -1){                                              LOG_TRACE("assigning det: " << "f" << detections[assignmentVector[t]]->frameIdx << " " <<  ((MPFImageLocation)*(detections[assignmentVector[t]])) << " to track " << *track);
      track->releaseTracker();
      track->push_back(move(detections.at(assignmentVector[t])));
      track->kalmanCorrect();
    }
    t++;
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
    auto cfgPtr = make_shared<Config>(job);
    if(cfgPtr->lastError != MPF_DETECTION_SUCCESS){
      throw MPFDetectionException(cfgPtr->lastError,"failed to parse image job configuration parameters");
    }
    DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfgPtr);
                                                                               LOG_DEBUG( "[" << job.job_name << "] Number of detections = " << detections.size());
    MPFImageLocationVec locations;
    for(auto &det:detections){
      MPFImageLocation loc = *det;
      det.reset();                    // release frame object
      cfgPtr->ReverseTransform(loc);
      locations.push_back(loc);
    }
    return locations;

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
* \param[in,out]  tracks  Tracks collection to which detections will be added
*
* \returns MPFVideoTrack object resulting from conversion
*
* \note detection pts are released on conversion and confidence is assigned
*       as the average of the detection confidences
*
***************************************************************************** */
MPFVideoTrack OcvYoloDetection::_convert_track(Track &track){

  MPFVideoTrack mpf_track;
  mpf_track.start_frame = track.front()->frameIdx;
  mpf_track.stop_frame  = track.back()->frameIdx;

  stringstream start_feature;
  stringstream stop_feature;
  start_feature << track.front()->getFeature();  // make sure we have computed features to serialize
  stop_feature << track.back()->getFeature();    //  for the start and end detections.
  mpf_track.detection_properties["START_FEATURE"] = start_feature.str();
  mpf_track.detection_properties["STOP_FEATURE"]  = stop_feature.str();

  #ifndef NDEBUG
    track.kalmanDump();
  #endif

  for(auto &det:track){
    mpf_track.confidence += det->confidence;
    mpf_track.frame_locations.insert(mpf_track.frame_locations.end(),{det->frameIdx,*det});
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
  TrackPtrList      trackPtrs;

  try{
    auto cfgPtr = make_shared<Config>(job);
    if(cfgPtr->lastError != MPF_DETECTION_SUCCESS){
      throw MPFDetectionException(cfgPtr->lastError,"failed to parse video job configuration parameters");
    }
    size_t detectTrigger = 0;
    while(cfgPtr->nextFrame()) {                                                   LOG_TRACE( ".");
                                                                                   LOG_TRACE( "processing frame " << cfgPtr->frameIdx);
      // remove any tracks too far in the past
      trackPtrs.remove_if([&](unique_ptr<Track>& tPtr){
        if(cfgPtr->frameIdx - tPtr->back()->frameIdx > cfgPtr->maxFrameGap){           LOG_TRACE("dropping old track: " << *tPtr);
          mpf_tracks.push_back(_convert_track(*tPtr));
          return true;
        }
        return false;
      });

      // advance kalman predictions
      if(! cfgPtr->kfDisabled){
        for(auto &trackPtr:trackPtrs){
          trackPtr->kalmanPredict(cfgPtr->frameTimeInSec);
        }
      }

      if(detectTrigger == 0){                                                  LOG_TRACE("checking for new detections");
        DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfgPtr); // look for new detections

        if(detections.size() > 0){   // found some detections in current frame
          if(trackPtrs.size() >= 0 ){  // not all tracks were dropped
            vector<long> av;                                                   LOG_TRACE( detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
            // intersection over union tracking and assignment
            if(! cfgPtr->kfDisabled){
              av = _calcAssignmentVector<&DetectionLocation::kfIouDist>(trackPtrs,detections,cfgPtr->maxIOUDist);
            }else{
              av = _calcAssignmentVector<&DetectionLocation::iouDist>(trackPtrs,detections,cfgPtr->maxIOUDist);
            }
            _assignDetections2Tracks(trackPtrs, detections, av);               LOG_TRACE("IOU assignment complete");

            // feature-based tracking tracking and assignment
            if(detections.size() > 0){                                         LOG_TRACE( detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
              av = _calcAssignmentVector<&DetectionLocation::featureDist>(trackPtrs,detections,cfgPtr->maxFeatureDist);
              _assignDetections2Tracks(trackPtrs, detections, av);             LOG_TRACE("Feature assignment complete");
            }

            // center-to-center distance tracking and assignment
            if(detections.size() > 0){                                         LOG_TRACE( detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
              av = _calcAssignmentVector<&DetectionLocation::center2CenterDist>(trackPtrs,detections,cfgPtr->maxCenterDist);
              _assignDetections2Tracks(trackPtrs, detections, av);             LOG_TRACE("Center2Center assignment complete");
            }

          }
                                                                               LOG_TRACE( detections.size() <<" detections left for new tracks");
          // any detection not assigned up to this point becomes a new track
          for(auto &det:detections){                                           // make any unassigned detections into new tracks
            det->getFeature();                                                 // start of tracks always get feature calculated
            trackPtrs.push_back(unique_ptr<Track>(new Track(cfgPtr,move(det))));  LOG_TRACE("created new track " << *(trackPtrs.back()));
          }
        }
      }

      // check any tracks that didn't get a detection and use tracker to continue them if possible
      for(auto &track:trackPtrs){
        if(track->back()->frameIdx < cfgPtr->frameIdx){  // no detections for track in current frame, try tracking
          DetectionLocationPtr detPtr = track->ocvTrackerPredict();
          if(detPtr){  // tracker returned something
            track->push_back(move(detPtr));              // add new location as tracks's tail
            track->kalmanCorrect();
          }
        }
      }

      detectTrigger++;
      detectTrigger = detectTrigger % (cfgPtr->detFrameInterval + 1);

    }                                                                          LOG_DEBUG( "[" << job.job_name << "] Number of tracks detected = " << trackPtrs.size());

    // convert any remaining active tracks to MPFVideoTracks
    for(auto &trackPtr:trackPtrs){
      mpf_tracks.push_back(_convert_track(*trackPtr));
    }

    // reverse transform all mpf tracks
    for(auto &mpf_track:mpf_tracks){
      cfgPtr->ReverseTransform(mpf_track);
    }

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
