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

void JobConfig::_parse(const MPFJob &job){
  const Properties jpr = job.job_properties;
  minDetectionSize = abs(getEnv<int>  (jpr,"MIN_DETECTION_SIZE",             minDetectionSize));  LOG4CXX_TRACE(_log, "MIN_DETECTION_SIZE: "             << minDetectionSize);
  confThresh       = abs(getEnv<float>(jpr,"DETECTION_CONFIDENCE_THRESHOLD", confThresh));        LOG4CXX_TRACE(_log, "DETECTION_CONFIDENCE_THRESHOLD: " << confThresh);
  detFrameInterval = abs(getEnv<int>  (jpr,"DETECTION_FRAME_INTERVAL",       detFrameInterval));  LOG4CXX_TRACE(_log, "DETECTION_FRAME_INTERVAL: "       << detFrameInterval);

  maxFeatureDist   = abs(getEnv<float>(jpr,"TRACKING_MAX_FEATURE_DIST",      maxFeatureDist));    LOG4CXX_TRACE(_log, "TRACKING_MAX_FEATURE_DIST: " << maxFeatureDist);
  maxFrameGap      = abs(getEnv<int>  (jpr,"TRACKING_MAX_FRAME_GAP",         maxFrameGap));       LOG4CXX_TRACE(_log, "TRACKING_MAX_FRAME_GAP: "    << maxFrameGap);
  maxCenterDist    = abs(getEnv<float>(jpr,"TRACKING_MAX_CENTER_DIST",       maxCenterDist));     LOG4CXX_TRACE(_log, "TRACKING_MAX_CENTER_DIST: "  << maxCenterDist);
  maxIOUDist       = abs(getEnv<float>(jpr,"TRACKING_MAX_IOU_DIST",          maxIOUDist));        LOG4CXX_TRACE(_log, "TRACKING_MAX_IOU_DIST: "     << maxIOUDist);

  kfDisabled = getEnv<bool>(jpr,"KF_DISABLED", kfDisabled);                    LOG4CXX_TRACE(_log, "KF_DISABLED: " << kfDisabled);

  R.at<float>(0,0) = abs(getEnv<float>(jpr,"KF_MEASUREMENT_NOISE_X_VAR",      R.at<float>(0,0))); LOG4CXX_TRACE(_log, "KF_MEASUREMENT_NOISE_X_VAR: "      << R.at<float>(0,0));
  R.at<float>(1,1) = abs(getEnv<float>(jpr,"KF_MEASUREMENT_NOISE_Y_VAR",      R.at<float>(1,1))); LOG4CXX_TRACE(_log, "KF_MEASUREMENT_NOISE_Y_VAR: "      << R.at<float>(1,1));
  R.at<float>(2,2) = abs(getEnv<float>(jpr,"KF_MEASUREMENT_NOISE_WIDTH_VAR",  R.at<float>(2,2))); LOG4CXX_TRACE(_log, "KF_MEASUREMENT_NOISE_WIDTH_VAR: "  << R.at<float>(2,2));
  R.at<float>(3,3) = abs(getEnv<float>(jpr,"KF_MEASUREMENT_NOISE_HEIGHT_VAR", R.at<float>(3,3))); LOG4CXX_TRACE(_log, "KF_MEASUREMENT_NOISE_HEIGHT_VAR: " << R.at<float>(3,3));

  Q.at<float>(0,0) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_X_VAR",              Q.at<float>(0,0)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_X_VAR: "               << Q.at<float>(0,0));
  Q.at<float>(1,1) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_Y_VAR",              Q.at<float>(1,1)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_Y_VAR: "               << Q.at<float>(1,1));
  Q.at<float>(2,2) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_X_VELOCITY_VAR",     Q.at<float>(2,2)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_X_VELOCITY_VAR: "      << Q.at<float>(2,2));
  Q.at<float>(3,3) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_Y_VELOCITY_VAR",     Q.at<float>(3,3)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_Y_VELOCITY_VAR: "      << Q.at<float>(3,3));
  Q.at<float>(4,4) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_WIDTH_VAR",          Q.at<float>(4,4)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_WIDTH_VAR: "           << Q.at<float>(4,4));
  Q.at<float>(5,5) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_HEIGHT_VAR",         Q.at<float>(5,5)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_HEIGHT_VAR: "          << Q.at<float>(5,5));
  Q.at<float>(6,6) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_WIDTH_VELOCITY_VAR", Q.at<float>(6,6)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_WIDTH_VELOCITY_VAR: "  << Q.at<float>(6,6));
  Q.at<float>(7,7) = abs(getEnv<float>(jpr,"KF_PROCESS_NOISE_HEIGHT_VELOCITY_VAR",Q.at<float>(7,7)));  LOG4CXX_TRACE(_log, "KF_PROCESS_NOISE_HEIGHT_VELOCITY_VAR: " << Q.at<float>(7,7));

}

/* **************************************************************************
*   Dump config to a stream
*************************************************************************** */
ostream& operator<< (ostream& out, const JobConfig& cfg) {
  out << "{"
      << "\"minDetectionSize\": " << cfg.minDetectionSize << ","
      << "\"confThresh\":"        << cfg.confThresh       << ","
      << "\"detFrameInterval\":"  << cfg.detFrameInterval << ","
      <<  "\"maxFeatureDist\":"   << cfg.maxFeatureDist   << ","
      <<  "\"maxFrameGap\":"      << cfg.maxFrameGap      << ","
      <<  "\"maxCenterDist\":"    << cfg.maxCenterDist    << ","
      <<  "\"maxIOUDist\":"       << cfg.maxIOUDist       << ","
      <<  "\"kfDisabled\":"       << (cfg.kfDisabled ? "1":"0") << ","
      <<  "\"kfProcessCov\":"     << format(cfg.Q) << ","
      <<  "\"kfProcessCov\":"     << format(cfg.R) << ","
      << "}";
  return out;
}

/* **************************************************************************
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
  frameIdx(-1),
  lastError(MPF_DETECTION_SUCCESS),
  _frameTimeInMillis(-1.0),
  _imreaderPtr(unique_ptr<MPFImageReader>()),
  _videocapPtr(unique_ptr<MPFVideoCapture>()){

    // Kalman Measurement Matrix H
    //    | 1  0  0  0  0  0  0  0 |       | x |
    //    | 0  1  0  0  0  0  0  0 |       | y |
    //    | 0  0  0  0  1  0  0  0 | * X = | w |
    //    | 0  0  0  0  0  1  0  0 |       | h |
    H = cv::Mat_<float>::zeros(KF_MEAS_DIM,KF_STATE_DIM);
    H.at<float>(0,0) = 1.0f;
    H.at<float>(1,1) = 1.0f;
    H.at<float>(2,4) = 1.0f;
    H.at<float>(3,5) = 1.0f;

    // Kalman Motion Model Process noise covariance Matrix Q
    //    |qx  0  0  0  0  0  0  0 |
    //    | 0 qy  0  0  0  0  0  0 |
    //    | 0  0 qvx 0  0  0  0  0 |
    //    | 0  0  0 qvy 0  0  0  0 |
    //    | 0  0  0  0 qw  0  0  0 |
    //    | 0  0  0  0  0 qh  0  0 |
    //    | 0  0  0  0  0  0 qvw 0 |
    //    | 0  0  0  0  0  0  0 qvh|
    Q = cv::Mat_<float>::zeros(KF_STATE_DIM,KF_STATE_DIM);
    Q.at<float>(0,0) = 1.0f;  // model x position noise variance guess
    Q.at<float>(1,1) = 1.0f;  // model y position noise variance guess
    Q.at<float>(2,2) = 5.0f;  // model x velocity noise variance guess
    Q.at<float>(3,3) = 5.0f;  // model y velocity noise variance guess
    Q.at<float>(4,4) = 1.0f;  // model width noise variance guess
    Q.at<float>(5,5) = 1.0f;  // model height noise variance guess
    Q.at<float>(6,6) = 5.0f;  // model width velocity noise variance guess
    Q.at<float>(7,7) = 5.0f;  // model height velocity noise variance guess

    // Kalman Bounding Box Measurement noise covariance Matrix R
    //    |rx  0  0  0 |
    //    | 0 ry  0  0 |
    //    | 0  0 rw  0 |
    //    | 0  0  0 rh |
    R = cv::Mat_<float>::zeros(KF_MEAS_DIM,KF_MEAS_DIM);
    R.at<float>(0,0) = 1.0f;  // measurement x position noise variance guess
    R.at<float>(1,1) = 1.0f;  // measurement y position noise variance guess
    R.at<float>(2,2) = 10.0f; // measurement width noise variance guess
    R.at<float>(3,3) = 10.0f; // measurement height noise variance guess

  }

/* **************************************************************************
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

/* **************************************************************************
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
  }
}

/* **************************************************************************
*  Get current frame timestamp
*************************************************************************** */
const double JobConfig::GetCurrentTimeInMillis() const {
  if(_frameTimeInMillis < 0.0){
    _frameTimeInMillis = _videocapPtr->GetCurrentTimeInMillis();
    return _frameTimeInMillis;
  }else{
    return _frameTimeInMillis;
  }
}

/* **************************************************************************
*  Read next frame of video into current bgrFrame member variable and advance
*  frame index counter.
*************************************************************************** */
bool JobConfig::nextFrame(){
  frameIdx++;
  return _videocapPtr->Read(bgrFrame);
}

/* **************************************************************************
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