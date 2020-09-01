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

#include "types.h"
#include "Frame.h"
#include "Config.h"

using namespace MPF::COMPONENT;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr                Config::log        = log4cxx::Logger::getRootLogger();
string                            Config::pluginPath;
string                            Config::configPath;

/** **************************************************************************
*   Parse a string into a opencv matrix
*   e.g. [1,2,3,4, 5,6,7,8]
*************************************************************************** */
cv::Mat Config::_fromString(const string data,const int rows,const int cols,const string dt="f"){
  stringstream ss;
  ss << "{\"mat\":{\"type_id\":\"opencv-matrix\""
     << ",\"rows\":" << rows
     << ",\"cols\":" << cols
     << ",\"dt\":\"" << dt << '"'
     << ",\"data\":" << data << "}}";
  cv::FileStorage fs(ss.str(), cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
  cv::Mat mat;
  fs["mat"] >> mat;
  return mat;
}

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
  RN = _fromString(_strRN, 4, 1, "f");
  QN = _fromString(_strQN, 4, 1, "f");
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
    QN = _fromString(_strQN, 4, 1, "f");

    // Kalman Bounding Box Measurement noise sdtdev for covariance Matrix R
    _strRN = "[6.0, 6.0, 6.0, 6.0]";
    RN = _fromString(_strRN, 4, 1, "f");

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
* \param numImages the number of frames to attempt to read.
*
* \returns all frames read in a vector of pointers
*
*************************************************************************** */
FramePtrVec Config::getImageFrames(int numImages = 1) const{
  assert(numImages == 1);  // only one image at a time is implemented

  FramePtrVec frames;
  Frame nextFrame;
  auto framePtr = make_shared<Frame>();
  framePtr->bgr = _imreaderPtr->GetImage();
  if(framePtr->bgr.empty()){
    throw MPF_IMAGE_READ_ERROR;
  }
  frames.push_back(framePtr);                                                  LOG_DEBUG( "image = " << frames.back()->bgr.size());
  return frames;
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
FramePtrVec Config::getVideoFrames(int numFrames = 1) const {
  FramePtrVec frames;
  Frame nextFrame;
  for(int i=0; i<numFrames;i++){
    auto framePtr = make_shared<Frame>();
    framePtr->idx      =         _videocapPtr->GetCurrentFramePosition();
    framePtr->time     = 0.001 * _videocapPtr->GetCurrentTimeInMillis() ;
    framePtr->timeStep = 1.0   / _videocapPtr->GetFrameRate();
    if(_videocapPtr->Read(framePtr->bgr)){
      frames.push_back(framePtr);
    }else{
      break;
    }
  }
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
}