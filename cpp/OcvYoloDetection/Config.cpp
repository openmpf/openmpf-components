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

#include <chrono>
#include <thread>

#include "types.h"
#include "util.h"
#include "Frame.h"
#include "Config.h"

using namespace MPF::COMPONENT;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr                Config::log        = log4cxx::Logger::getRootLogger();
string                            Config::pluginPath;
string                            Config::configPath;

/** **************************************************************************
*   Parse argument fromMPFJob structure to our job objects
*************************************************************************** */
void Config::_parse(const MPFJob &job){
  const Properties jpr = job.job_properties;
  confThresh        = abs(getEnv<float>(jpr,"DETECTION_CONFIDENCE_THRESHOLD",      confThresh));         LOG_TRACE( "DETECTION_CONFIDENCE_THRESHOLD: "             << confThresh);
  nmsThresh         = abs(getEnv<float>(jpr,"DETECTION_NMS_THRESHOLD",             nmsThresh));          LOG_TRACE( "DETECTION_NMS_THRESHOLD: "                    << nmsThresh);
  inputImageSize    = abs(getEnv<int>  (jpr,"DETECTION_IMAGE_SIZE",                inputImageSize));     LOG_TRACE( "DETECTION_IMAGE_SIZE: "                       << inputImageSize);
  detFrameInterval  = abs(getEnv<int>  (jpr,"DETECTION_FRAME_INTERVAL",            detFrameInterval));   LOG_TRACE( "DETECTION_FRAME_INTERVAL: "                   << detFrameInterval);
  frameBatchSize    = abs(getEnv<int>  (jpr,"DETECTION_FRAME_BATCH_SIZE",          frameBatchSize));     LOG_TRACE( "DETECTION_FRAME_BATCH_SIZE: "                 << frameBatchSize);
  numClassPerRegion = abs(getEnv<int>  (jpr,"NUMBER_OF_CLASSIFICATIONS_PER_REGION",numClassPerRegion));  LOG_TRACE( "NUMBER_OF_CLASSIFICATIONS_PER_REGION: "       << numClassPerRegion);

  maxClassDist     = abs(getEnv<float>(jpr,"TRACKING_MAX_CLASS_DIST",        maxClassDist));             LOG_TRACE( "TRACKING_MAX_CLASS_DIST: "   << maxClassDist);
  maxFeatureDist   = abs(getEnv<float>(jpr,"TRACKING_MAX_FEATURE_DIST",      maxFeatureDist));           LOG_TRACE( "TRACKING_MAX_FEATURE_DIST: " << maxFeatureDist);
  maxFrameGap      = abs(getEnv<int>  (jpr,"TRACKING_MAX_FRAME_GAP",         maxFrameGap));              LOG_TRACE( "TRACKING_MAX_FRAME_GAP: "    << maxFrameGap);
  maxCenterDist    = abs(getEnv<float>(jpr,"TRACKING_MAX_CENTER_DIST",       maxCenterDist));            LOG_TRACE( "TRACKING_MAX_CENTER_DIST: "  << maxCenterDist);
  maxIOUDist       = abs(getEnv<float>(jpr,"TRACKING_MAX_IOU_DIST",          maxIOUDist));               LOG_TRACE( "TRACKING_MAX_IOU_DIST: "     << maxIOUDist);
  maxKFResidual    = abs(getEnv<float>(jpr,"KF_MAX_ASSIGNMENT_RESIDUAL",     maxKFResidual));            LOG_TRACE( "KF_MAX_ASSIGNMENT_RESIDUAL: "<< maxKFResidual);
  edgeSnapDist     = abs(getEnv<float>(jpr,"TRACKING_EDGE_SNAP_DIST",        edgeSnapDist));             LOG_TRACE( "TRACKING_EDGE_SNAP_DIST: "   << edgeSnapDist);
  dftSize          = abs(getEnv<int>  (jpr,"TRACKING_DFT_SIZE",              dftSize));                  LOG_TRACE( "TRACKING_DFT_SIZE:"          << dftSize);
  dftHannWindowEnabled = getEnv<bool> (jpr,"TRACKING_DFT_USE_HANNING_WINDOW",dftHannWindowEnabled);      LOG_TRACE( "TRACKING_DFT_USE_HANNING_WINDOW: " << dftHannWindowEnabled);
  mosseTrackerDisabled = getEnv<bool> (jpr,"TRACKING_DISABLE_MOSSE_TRACKER" ,mosseTrackerDisabled);      LOG_TRACE( "TRACKING_DISABLE_MOSSE_TRACKER: "  << mosseTrackerDisabled);


  kfDisabled = getEnv<bool>(jpr,"KF_DISABLED", kfDisabled);                                        LOG_TRACE( "KF_DISABLED: " << kfDisabled);
  _strRN = getEnv<string>(jpr,"KF_RN",_strRN);                                                     LOG_TRACE( "KF_RN: "       << _strRN);
  _strQN = getEnv<string>(jpr,"KF_QN",_strQN);                                                     LOG_TRACE( "KF_QN: "       << _strQN);
  RN = fromString(_strRN, 4, 1, "f");
  QN = fromString(_strQN, 4, 1, "f");
  //convert stddev to variances
  RN = RN.mul(RN);
  QN = QN.mul(QN);
  fallback2CpuWhenGpuProblem = getEnv<bool>(jpr,"FALLBACK_TO_CPU_WHEN_GPU_PROBLEM",
                                            fallback2CpuWhenGpuProblem);                          LOG_TRACE( "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM: " << fallback2CpuWhenGpuProblem);
  cudaDeviceId               = getEnv<int> (jpr,"CUDA_DEVICE_ID",
                                            cudaDeviceId);                                        LOG_TRACE( "CUDA_DEVICE_ID: " << cudaDeviceId);
}

/** **************************************************************************
*   Dump config to a stream
*************************************************************************** */
ostream& operator<< (ostream& out, const Config& cfg) {
  out << "{"
      << "\"confThresh\":"                 << cfg.confThresh        << ","
      << "\"nmsThresh\":"                  << cfg.nmsThresh         << ","
      << "\"inputImageSize\":"             << cfg.inputImageSize    << ","
      << "\"detFrameInterval\":"           << cfg.detFrameInterval  << ","
      << "\"frameBatchSize\":"             << cfg.frameBatchSize    << ","
      << "\"numClassPerRegion\":"          << cfg.numClassPerRegion << ","
      << "\"maxClassDist\":"               << cfg.maxClassDist      << ","
      << "\"maxFeatureDist\":"             << cfg.maxFeatureDist    << ","
      << "\"maxFrameGap\":"                << cfg.maxFrameGap       << ","
      << "\"maxCenterDist\":"              << cfg.maxCenterDist     << ","
      << "\"maxIOUDist\":"                 << cfg.maxIOUDist        << ","
      << "\"edgeSnapDist\":"               << cfg.edgeSnapDist      << ","
      << "\"dftSize\":"                    << cfg.dftSize           << ","
      << "\"dftHannWindow\":"              <<(cfg.dftHannWindowEnabled ? "1" : "0") << ","
      << "\"maxKFResidual\":"              << cfg.maxKFResidual << ","
      << "\"kfDisabled\":"                 <<(cfg.kfDisabled ? "1":"0") << ","
      << "\"mosseTrackerDisabled\":"       <<(cfg.mosseTrackerDisabled ? "1":"0") << ","
      << "\"fallback2CpuWhenGpuProblem\":" <<(cfg.fallback2CpuWhenGpuProblem ? "1" : "0") << ","
      << "\"cudaDeviceId\":"               << cfg.cudaDeviceId << ","
      << "\"kfProcessVar\":"        << format(cfg.QN) << ","
      << "\"kfMeasurementVar\":"    << format(cfg.RN)
      << "}";
  return out;
}

/** **************************************************************************
*  Default constructor with default values
*************************************************************************** */
Config::Config():
  confThresh(0.5),
  nmsThresh(0.4),
  inputImageSize(416),
  maxFrameGap(4),
  detFrameInterval(1),
  frameBatchSize(1),
  numClassPerRegion(5),
  maxClassDist(0.25),
  maxFeatureDist(0.25),
  maxCenterDist(0.0),
  maxIOUDist(0.5),
  edgeSnapDist(0.0075),
  maxKFResidual(4.0),
  kfDisabled(false),
  mosseTrackerDisabled(false),
  cudaDeviceId(0),
  fallback2CpuWhenGpuProblem(true),
  lastError(MPF_DETECTION_SUCCESS),
  _imreaderPtr(unique_ptr<MPFImageReader>()),
  _videocapPtr(unique_ptr<MPFVideoCapture>()){

    // Kalman filter motion model noise / acceleration stddev for covariance matrix Q
    _strQN = "[100.0,100.0,100.0,100.0]";
    QN = fromString(_strQN, 4, 1, "f");

    // Kalman Bounding Box Measurement noise sdtdev for covariance Matrix R
    _strRN = "[6.0, 6.0, 6.0, 6.0]";
    RN = fromString(_strRN, 4, 1, "f");

  }

/** **************************************************************************
*  Constructor that parses parameters from MPFImageJob and load image
*************************************************************************** */
Config::Config(const MPFImageJob &job):
  Config() {
                                                                               LOG_DEBUG( "[" << job.job_name << "Data URI = " << job.data_uri);
  _parse(job);
  if(job.data_uri.empty()) {                                                   LOG_ERROR( "[" << job.job_name << "Invalid image url");
    lastError = MPF_INVALID_DATAFILE_URI;
  }else{
    _imreaderPtr = unique_ptr<MPFImageReader>(new MPFImageReader(job));
  }
}

/** **************************************************************************
*  Read image.
*
*
* \returns all frames read in a vector of pointers
*
*************************************************************************** */
Frame Config::getImageFrame() const{

  Frame frame;
  frame.bgr = _imreaderPtr->GetImage();
  if(frame.bgr.empty()){
    throw MPF_IMAGE_READ_ERROR;
  }                                                                            LOG_DEBUG( "image = " << frame.bgr.size());
  return frame;
}

/** **************************************************************************
*  Asyncronous thread to keep frameQ filled with frames
*************************************************************************** */
void Config::frameReaderThread(){
  while(_capturing){
    if(_frameQ.size() < 2 * frameBatchSize){
      Frame frame(_videocapPtr->GetCurrentFramePosition(),
                  _videocapPtr->GetCurrentTimeInMillis() * 0.001,
                  1.0   / _videocapPtr->GetFrameRate(),
                  cv::Mat());
      try{
        if(_videocapPtr->Read(frame.bgr)){
          lock_guard<mutex> guard(_captureMtx);
          _frameQ.push(frame);
        }else{
          _capturing = false;
        }
      }catch(...){
        lock_guard<mutex> guard(_captureMtx);
        _capturing = false;
        lastError = MPF_COULD_NOT_READ_DATAFILE;
      }
    }
  }
}

/** **************************************************************************
*  Constructor that parses parameters from MPFVideoJob and initializes
*  video capture / reader
*************************************************************************** */
Config::Config(const MPFVideoJob &job):
  Config() {
  _parse(job);

  if(job.data_uri.empty()) {                                                   LOG_ERROR( "[" << job.job_name << "Invalid video url");
    lastError = MPF_INVALID_DATAFILE_URI;
  }else{
    _videocapPtr = unique_ptr<MPFVideoCapture>(new MPFVideoCapture(job, true, true));
    if(!_videocapPtr->IsOpened()){                                             LOG_ERROR( "[" << job.job_name << "] Could not initialize capturing");
      lastError = MPF_COULD_NOT_OPEN_DATAFILE;
    }else{
      _capturing = true;
      _captureThreadPtr = unique_ptr<thread>(new thread([this](){frameReaderThread();}));
    }
  }
}

/** **************************************************************************
*  Read next frames of video.
*
* \param numFrames the number of frames to attempt to read.
*
* \returns all frames read in a vector of pointers
*
*************************************************************************** */
FrameVec Config::getVideoFrames(int numFrames = 1) {

  FrameVec frames;
  while(frames.size() < numFrames){
    if(!_frameQ.empty()){
      lock_guard<mutex> guard(_captureMtx);
      frames.push_back(move(_frameQ.front()));
      _frameQ.pop();
    }else if(!_capturing){
      break;
    }
  }

  lock_guard<mutex> guard(_captureMtx);
  if(lastError != MPF_DETECTION_SUCCESS) throw lastError;

  return frames;
}

/** **************************************************************************
* Destructor to release image / video readers
*************************************************************************** */
Config::~Config(){
  if(_imreaderPtr){
    _imreaderPtr.reset();
  }
  if(_videocapPtr){
    _videocapPtr->Release();
    _videocapPtr.reset();
  }
  if(_captureThreadPtr){
    _capturing = false;
    _captureThreadPtr->join();
  }
}