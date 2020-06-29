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

#include "OcvSsdFaceDetection.h"

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


using namespace MPF::COMPONENT;

// Temporary initializer for static member variable
log4cxx::LoggerPtr JobConfig::_log = log4cxx::Logger::getRootLogger();

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
*  Initialize SSD face detector module by setting up paths and reading configs
*  configuration variables are turned into environment variables for late
*  reference
*
* \returns   true on success
***************************************************************************** */
bool OcvSsdFaceDetection::Init() {
    string plugin_path    = GetRunDirectory() + "/OcvSsdFaceDetection";
    string config_path    = plugin_path       + "/config";

    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    _log = log4cxx::Logger::getLogger("OcvSsdFaceDetection");                  LOG4CXX_DEBUG(_log,"Initializing OcvSSDFaceDetector");
    JobConfig::_log = _log;

    // read config file and create or update any missing env variables
    string config_params_path = config_path + "/mpfOcvSsdFaceDetection.ini";
    QHash<QString,QString> params;
    if(LoadConfig(config_params_path, params)) {                               LOG4CXX_ERROR(_log, "Failed to load the OcvSsdFaceDetection config from: " << config_params_path);
        return false;
    }                                                                          LOG4CXX_TRACE(_log,"read config file:" << config_params_path);
    Properties props;
    for(auto p = params.begin(); p != params.end(); p++){
      const string key = p.key().toStdString();
      const string val = p.value().toStdString();                              LOG4CXX_TRACE(_log,"Config    Vars:" << key << "=" << val);
      const char* env_val = getenv(key.c_str());                               LOG4CXX_TRACE(_log,"Verifying ENVs:" << key << "=" << env_val);
      if(env_val == NULL){
        if(setenv(key.c_str(), val.c_str(), 0) !=0){                           LOG4CXX_ERROR(_log,"Failed to convert config to env variable: " << key << "=" << val);
          return false;
        }
      }else if(string(env_val) != val){                                        LOG4CXX_INFO(_log,"Keeping existing env variable:" << key << "=" << string(env_val));
      }
    }

    // initialize adaptive histogram equalizer
    _equalizerPtr = cv::createCLAHE(40.0,cv::Size(8,8));

    bool detectionLocationInitalizedOK = DetectionLocation::Init(_log, plugin_path);

    Properties fromEnv;
    int cudaDeviceId   = getEnv<int> (fromEnv,"CUDA_DEVICE_ID", -1);
    bool fallbackToCPU = getEnv<bool>(fromEnv,"FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", true);
    bool defaultCudaDeviceOK = DetectionLocation::trySetCudaDevice(cudaDeviceId) || fallbackToCPU;

    bool trackInitializedOK = Track::Init(_log, plugin_path);
    bool kfTrackerInitializedOK = KFTracker::Init(_log, plugin_path);

    return    detectionLocationInitalizedOK
           && defaultCudaDeviceOK
           && trackInitializedOK
           && kfTrackerInitializedOK;


}

/** ****************************************************************************
* Clean up and release any created detector objects
*
* \returns   true on success
***************************************************************************** */
bool OcvSsdFaceDetection::Close() {
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
vector<long> OcvSsdFaceDetection::_calcAssignmentVector(const TrackPtrList            &tracks,
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
  }                                                                            LOG4CXX_TRACE(_log,"cost matrix[tr=" << costs.nr() << ",det=" << costs.nc() << "]: " << dformat(costs));

  // solve cost matrix, track av[track] get assigned detection[av[track]]
  av = dlib::max_cost_assignment(costs);                                       LOG4CXX_TRACE(_log, "solved assignment vec["<< av.size() << "] = "  << av);

  // drop dummy tracks used to pad cost matrix
  av.resize(tracks.size());

  // knock out assignments that are too costly (i.e. new track needed)
  for(long t=0; t<av.size(); t++){
    if(costs(t,av[t]) == 0){
      av[t] = -1;
    }
  }                                                                            LOG4CXX_TRACE(_log, "modified assignment vec["<< av.size() << "] = "  << av);

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
void OcvSsdFaceDetection::_assignDetections2Tracks(TrackPtrList             &tracks,
                                                   DetectionLocationPtrVec  &detections,
                                                   const vector<long>       &assignmentVector){

  long t=0;
  for(auto &track:tracks){
    if(assignmentVector[t] != -1){                                              LOG4CXX_TRACE(_log,"assigning det: " << "f" << detections[assignmentVector[t]]->frameIdx << " " <<  ((MPFImageLocation)*(detections[assignmentVector[t]])) << " to track " << *track);
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
MPFImageLocationVec OcvSsdFaceDetection::GetDetections(const MPFImageJob   &job) {

  try {                                                                        LOG4CXX_DEBUG(_log, "[" << job.job_name << "Data URI = " << job.data_uri);
    JobConfig cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS){
      throw MPFDetectionException(cfg.lastError,"failed to parse image job configuration parameters");
    }
    DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfg);
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of faces detected = " << detections.size());
    MPFImageLocationVec locations;
    for(auto &det:detections){
      MPFImageLocation loc = *det;
      det.reset();                    // release frame object
      cfg.ReverseTransform(loc);
      locations.push_back(loc);
    }
    return locations;

  }catch(const runtime_error& re){
    LOG4CXX_FATAL(_log, "[" << job.job_name << "] runtime error: " << re.what());
    throw MPFDetectionException(re.what());
  }catch(const exception& ex){
    LOG4CXX_FATAL(_log, "[" << job.job_name << "] exception: " << ex.what());
    throw MPFDetectionException(ex.what());
  }catch (...){
    LOG4CXX_FATAL(_log, "[" << job.job_name << "] unknown error");
    throw MPFDetectionException(" Unknown error processing image job");
  }
                                                                               LOG4CXX_DEBUG(_log,"[" << job.job_name << "] complete.");
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
MPFVideoTrack OcvSsdFaceDetection::_convert_track(Track &track){

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
MPFVideoTrackVec OcvSsdFaceDetection::GetDetections(const MPFVideoJob &job){

  MPFVideoTrackVec  mpf_tracks;
  TrackPtrList      trackPtrs;

  try{
    JobConfig cfg(job);
    if(cfg.lastError != MPF_DETECTION_SUCCESS){
      throw MPFDetectionException(cfg.lastError,"failed to parse video job configuration parameters");
    }

    size_t detectTrigger = 0;
    while(cfg.nextFrame()) {                                                   LOG4CXX_TRACE(_log, ".");
                                                                               LOG4CXX_TRACE(_log, "processing frame " << cfg.frameIdx);
      // remove any tracks too far in the past
      trackPtrs.remove_if([&](unique_ptr<Track>& tPtr){
        if(cfg.frameIdx - tPtr->back()->frameIdx > cfg.maxFrameGap){           LOG4CXX_TRACE(_log,"dropping old track: " << *tPtr);
          mpf_tracks.push_back(_convert_track(*tPtr));
          return true;
        }
        return false;
      });

      // advance kalman predictions
      if(! cfg.kfDisabled){
        for(auto &trackPtr:trackPtrs){
          trackPtr->kalmanPredict(cfg.frameTimeInSec);
        }
      }

      if(detectTrigger == 0){                                                  LOG4CXX_TRACE(_log,"checking for new detections");
        DetectionLocationPtrVec detections = DetectionLocation::createDetections(cfg); // look for new detections

        if(detections.size() > 0){   // found some detections in current frame
          if(trackPtrs.size() >= 0 ){  // not all tracks were dropped
            vector<long> av;                                                   LOG4CXX_TRACE(_log, detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
            // intersection over union tracking and assignment
            if(! cfg.kfDisabled){
              av = _calcAssignmentVector<&DetectionLocation::kfIouDist>(trackPtrs,detections,cfg.maxIOUDist);
            }else{
              av = _calcAssignmentVector<&DetectionLocation::iouDist>(trackPtrs,detections,cfg.maxIOUDist);
            }
            _assignDetections2Tracks(trackPtrs, detections, av);               LOG4CXX_TRACE(_log,"IOU assignment complete");

            // feature-based tracking tracking and assignment
            if(detections.size() > 0){                                         LOG4CXX_TRACE(_log, detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
              av = _calcAssignmentVector<&DetectionLocation::featureDist>(trackPtrs,detections,cfg.maxFeatureDist);
              _assignDetections2Tracks(trackPtrs, detections, av);             LOG4CXX_TRACE(_log,"Feature assignment complete");
            }

            // center-to-center distance tracking and assignment
            if(detections.size() > 0){                                         LOG4CXX_TRACE(_log, detections.size() <<" detections to be matched to " << trackPtrs.size() << " tracks");
              av = _calcAssignmentVector<&DetectionLocation::center2CenterDist>(trackPtrs,detections,cfg.maxCenterDist);
              _assignDetections2Tracks(trackPtrs, detections, av);             LOG4CXX_TRACE(_log,"Center2Center assignment complete");
            }

          }
                                                                               LOG4CXX_TRACE(_log, detections.size() <<" detections left for new tracks");
          // any detection not assigned up to this point becomes a new track
          for(auto &det:detections){                                           // make any unassigned detections into new tracks
            det->getFeature();                                                 // start of tracks always get feature calculated
            trackPtrs.push_back(unique_ptr<Track>(new Track(move(det),cfg)));  LOG4CXX_TRACE(_log,"created new track " << *(trackPtrs.back()));
          }
        }
      }

      // check any tracks that didn't get a detection and use tracker to continue them if possible
      for(auto &track:trackPtrs){
        if(track->back()->frameIdx < cfg.frameIdx){  // no detections for track in current frame, try tracking
          DetectionLocationPtr detPtr = track->ocvTrackerPredict(cfg);
          if(detPtr){  // tracker returned something
            track->push_back(move(detPtr));              // add new location as tracks's tail
            track->kalmanCorrect();
          }
        }
      }

      detectTrigger++;
      detectTrigger = detectTrigger % (cfg.detFrameInterval + 1);

    }                                                                          LOG4CXX_DEBUG(_log, "[" << job.job_name << "] Number of tracks detected = " << trackPtrs.size());

    // convert any remaining active tracks to MPFVideoTracks
    for(auto &trackPtr:trackPtrs){
      mpf_tracks.push_back(_convert_track(*trackPtr));
    }

    // reverse transform all mpf tracks
    for(auto &mpf_track:mpf_tracks){
      cfg.ReverseTransform(mpf_track);
    }

    return mpf_tracks;

  }catch(const runtime_error& re){                                             LOG4CXX_FATAL(_log, "[" << job.job_name << "] runtime error: " << re.what());
    throw MPFDetectionException(re.what());
  }catch(const exception& ex){                                                 LOG4CXX_FATAL(_log, "[" << job.job_name << "] exception: " << ex.what());
    throw MPFDetectionException(ex.what());
  }catch (...) {                                                               LOG4CXX_FATAL(_log, "[" << job.job_name << "] unknown error");
    throw MPFDetectionException("Unknown error processing video job");
  }
                                                                               LOG4CXX_DEBUG(_log,"[" << job.job_name << "] complete.");
}

/** **************************************************************************
*  Perform histogram equalization
*
* \param cfg job structure containing image frame to be equalized
*
*************************************************************************** */
void OcvSsdFaceDetection::_equalizeHistogram(JobConfig &cfg){
  cv::Mat3b hsvFrame;
  cv::cvtColor(cfg.bgrFrame,hsvFrame,cv::COLOR_BGR2HSV);

  vector<cv::Mat1b> hsvComponents;
  cv::split(hsvFrame, hsvComponents);

  cv::equalizeHist(hsvComponents[2],hsvComponents[2]);
  cv::merge(hsvComponents,hsvFrame);
  cv::cvtColor(hsvFrame,cfg.bgrFrame,cv::COLOR_HSV2BGR);
}


/** **************************************************************************
*  Perform image normalization
*
* \param cfg job structure containing image frame to be normalized
*
*************************************************************************** */
void OcvSsdFaceDetection::_normalizeFrame(JobConfig &cfg){
  cv::Mat fltImg;
  cfg.bgrFrame.convertTo(fltImg,CV_32F);

  cv::Mat mu, mu_sq, sigma;
  cv::blur(fltImg, mu, cv::Size(3,3));
  cv::blur(fltImg.mul(fltImg), mu_sq, cv::Size(3,3));
  cv::sqrt(mu_sq - mu.mul(mu), sigma);

  fltImg = (fltImg - mu) / sigma;

  cv::normalize(fltImg, cfg.bgrFrame, 255,0, cv::NORM_MINMAX, CV_8U);

}

MPF_COMPONENT_CREATOR(OcvSsdFaceDetection);
MPF_COMPONENT_DELETER();
