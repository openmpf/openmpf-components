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
#include "JobConfig.h"

using namespace MPF::COMPONENT;

/** **************************************************************************
*   Parse a string into a opencv matrix
*   e.g. [1,2,3,4, 5,6,7,8]
*************************************************************************** */
cv::Mat JobConfig::_fromString(const string data,const int rows,const int cols,const string dt="f"){
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
void JobConfig::_parse(const MPFJob &job){
  const Properties jpr = job.job_properties;
  minDetectionSize     = abs(getEnv<int>  (jpr,"MIN_DETECTION_SIZE",             minDetectionSize));      LOG4CXX_TRACE(_log, "MIN_DETECTION_SIZE: "             << minDetectionSize);
  confThresh           = abs(getEnv<float>(jpr,"DETECTION_CONFIDENCE_THRESHOLD", confThresh));            LOG4CXX_TRACE(_log, "DETECTION_CONFIDENCE_THRESHOLD: " << confThresh);

  frameInterval        = abs(getEnv<int>  (jpr,"FRAME_INTERVAL",                 1));                     LOG4CXX_TRACE(_log, "FRAME_INTERVAL: "                 << frameInterval);
  detFrameInterval     = abs(getEnv<int>  (jpr,"DETECTION_FRAME_INTERVAL",       detFrameInterval));      LOG4CXX_TRACE(_log, "DETECTION_FRAME_INTERVAL: "       << detFrameInterval);

  frameInterval = (frameInterval < 1) ? 1 : frameInterval;
  detFrameInterval = (detFrameInterval < 1) ? 1 : detFrameInterval;

  adjustedFrameInterval = 1; // perform detection on every frame we read from MPFVideoCapture
  if (frameInterval > detFrameInterval) {
      LOG4CXX_WARN(_log, "FRAME_INTERVAL of " << frameInterval << " is greater than DETECTION_FRAME_INTERVAL of "
                                              << detFrameInterval << ". Using " << frameInterval << " for DETECTION_FRAME_INTERVAL.");
  }
  else if (frameInterval < detFrameInterval) {
      adjustedFrameInterval = detFrameInterval/frameInterval; // perform detection on every adjustedFrameInterval-1 frames from MPFVideoCapture
      LOG4CXX_WARN(_log, "FRAME_INTERVAL of " << frameInterval << " is less than DETECTION_FRAME_INTERVAL of "
                                              << detFrameInterval << ". Using " << (adjustedFrameInterval*frameInterval) << " for DETECTION_FRAME_INTERVAL.");
  }

  maxFeatureDist       = abs(getEnv<float>(jpr,"TRACKING_MAX_FEATURE_DIST",      maxFeatureDist));        LOG4CXX_TRACE(_log, "TRACKING_MAX_FEATURE_DIST: " << maxFeatureDist);
  maxFrameGap          = abs(getEnv<int>  (jpr,"TRACKING_MAX_FRAME_GAP",         maxFrameGap));           LOG4CXX_TRACE(_log, "TRACKING_MAX_FRAME_GAP: "    << maxFrameGap);
  maxCenterDist        = abs(getEnv<float>(jpr,"TRACKING_MAX_CENTER_DIST",       maxCenterDist));         LOG4CXX_TRACE(_log, "TRACKING_MAX_CENTER_DIST: "  << maxCenterDist);
  maxIOUDist           = abs(getEnv<float>(jpr,"TRACKING_MAX_IOU_DIST",          maxIOUDist));            LOG4CXX_TRACE(_log, "TRACKING_MAX_IOU_DIST: "     << maxIOUDist);

  kfDisabled = getEnv<bool>(jpr,"KF_DISABLED", kfDisabled);                            LOG4CXX_TRACE(_log, "KF_DISABLED: " << kfDisabled);

  _strRN = getEnv<string>(jpr,"KF_RN",_strRN);                                         LOG4CXX_TRACE(_log, "KF_RN: "       << _strRN);
  _strQN = getEnv<string>(jpr,"KF_QN",_strQN);                                         LOG4CXX_TRACE(_log, "KF_QN: "       << _strQN);
  RN = _fromString(_strRN, 4, 1, "f");
  QN = _fromString(_strQN, 4, 1, "f");
  //convert stddev to variances
  RN = RN.mul(RN);
  QN = QN.mul(QN);
  fallback2CpuWhenGpuProblem = getEnv<bool>(jpr,"FALLBACK_TO_CPU_WHEN_GPU_PROBLEM",
                                            fallback2CpuWhenGpuProblem);               LOG4CXX_TRACE(_log, "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM: " << fallback2CpuWhenGpuProblem);
  cudaDeviceId               = getEnv<int> (jpr,"CUDA_DEVICE_ID",
                                            cudaDeviceId);                             LOG4CXX_TRACE(_log, "CUDA_DEVICE_ID: " << cudaDeviceId);
}

/** **************************************************************************
*   Dump config to a stream
*************************************************************************** */
ostream& operator<< (ostream& out, const JobConfig& cfg) {
  out << "{"
      <<  "\"minDetectionSize\": "     << cfg.minDetectionSize << ","
      <<  "\"confThresh\":"            << cfg.confThresh       << ","
      <<  "\"frameInterval\":"         << cfg.frameInterval    << ","
      <<  "\"detFrameInterval\":"      << cfg.detFrameInterval << ","
      <<  "\"adjustedFrameInterval\":" << cfg.adjustedFrameInterval << ","
      <<  "\"maxFeatureDist\":"        << cfg.maxFeatureDist   << ","
      <<  "\"maxFrameGap\":"           << cfg.maxFrameGap      << ","
      <<  "\"maxCenterDist\":"         << cfg.maxCenterDist    << ","
      <<  "\"maxIOUDist\":"            << cfg.maxIOUDist       << ","
      <<  "\"kfDisabled\":"            << (cfg.kfDisabled ? "1":"0") << ","
      <<  "\"kfProcessVar\":"          << format(cfg.QN) << ","
      <<  "\"kfMeasurementVar\":"      << format(cfg.RN) << ","
      <<  "\"fallback2CpuWhenGpuProblem\":" << (cfg.fallback2CpuWhenGpuProblem ? "1" : "0") << ","
      <<  "\"cudaDeviceId\":"          << cfg.cudaDeviceId
      << "}";
  return out;
}

/** **************************************************************************
*  Default constructor with default values
*************************************************************************** */
JobConfig::JobConfig():
  minDetectionSize(46),
  confThresh(0.5),
  maxFrameGap(4),
  detFrameInterval(1),
  maxFeatureDist(0.25),
  maxCenterDist(0.0),
  maxIOUDist(0.5),
  kfDisabled(false),
  cudaDeviceId(0),
  fallback2CpuWhenGpuProblem(true),
  frameIdx(-1),
  frameTimeInSec(-1.0),
  frameTimeStep(-1.0),
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
JobConfig::JobConfig(const MPFImageJob &job):
  JobConfig() {
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "Data URI = " << job.data_uri);
  _parse(job);

  if(job.data_uri.empty()) {                                                   LOG4CXX_ERROR(_log, "[" << job.job_name << "Invalid image url");
    lastError = MPF_INVALID_DATAFILE_URI;
  }else{
    _imreaderPtr = unique_ptr<MPFImageReader>(new MPFImageReader(job));
    bgrFrame = _imreaderPtr->GetImage();
    if(bgrFrame.empty()){                                                      LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not read image file: " << job.data_uri);
      lastError = MPF_IMAGE_READ_ERROR;
    }                                                                          LOG4CXX_DEBUG(_log, "[" << job.job_name << "] image.width  = " << bgrFrame.cols);
                                                                               LOG4CXX_DEBUG(_log, "[" << job.job_name << "] image.height = " << bgrFrame.rows);
  }
}

/** **************************************************************************
*  Constructor that parses parameters from MPFVideoJob and initializes
*  video capture / reader
*************************************************************************** */
JobConfig::JobConfig(const MPFVideoJob &job):
  JobConfig() {

  _parse(job);

  if(job.data_uri.empty()) {                                                   LOG4CXX_ERROR(_log, "[" << job.job_name << "Invalid video url");
    lastError = MPF_INVALID_DATAFILE_URI;
  }else{
    _videocapPtr = unique_ptr<MPFVideoCapture>(new MPFVideoCapture(job, true, true));
    if(!_videocapPtr->IsOpened()){                                             LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not initialize capturing");
      lastError = MPF_COULD_NOT_OPEN_DATAFILE;
    }
    // pre-compute diagonal normalization factor for distance normalizations
    cv::Size fs = _videocapPtr->GetFrameSize();
    float diag  = sqrt(fs.width*fs.width + fs.height*fs.height);
    widthOdiag  = fs.width  / diag;
    heightOdiag = fs.height / diag;

    frameTimeStep = 1.0 / _videocapPtr->GetFrameRate();
  }
}

/** **************************************************************************
*  Read next frame of video into current bgrFrame member variable and advance
*  frame index counter.
*************************************************************************** */
bool JobConfig::nextFrame(){
  frameIdx++;
  frameTimeInSec = _videocapPtr->GetCurrentTimeInMillis() * 0.001;
  return _videocapPtr->Read(bgrFrame);
}

/** **************************************************************************
* Destructor to release image / video readers
*************************************************************************** */
JobConfig::~JobConfig(){
  if(_imreaderPtr){
    _imreaderPtr.reset();
  }
  if(_videocapPtr){
    _videocapPtr->Release();
    _videocapPtr.reset();
  }
}