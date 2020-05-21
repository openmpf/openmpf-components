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

#include "JobConfig.h"

using namespace MPF::COMPONENT;

void JobConfig::_parse(const MPFJob &job){
  const Properties jpr = job.job_properties;
  minDetectionSize = abs(getEnv<int>  (jpr,"MIN_DETECTION_SIZE",             minDetectionSize));  LOG4CXX_TRACE(_log, "MIN_DETECTION_SIZE: " << minDetectionSize);
  confThresh       = abs(getEnv<float>(jpr,"DETECTION_CONFIDENCE_THRESHOLD", confThresh));        LOG4CXX_TRACE(_log, "DETECTION_CONFIDENCE_THRESHOLD: " << confThresh);
  detFrameInterval = abs(getEnv<int>  (jpr,"DETECTION_FRAME_INTERVAL",       detFrameInterval));  LOG4CXX_TRACE(_log, "DETECTION_FRAME_INTERVAL: " << detFrameInterval);

  maxFeatureDist   = abs(getEnv<float>(jpr,"TRACKING_MAX_FEATURE_DIST",      maxFeatureDist));    LOG4CXX_TRACE(_log, "TRACKING_MAX_FEATURE_DIST: " << maxFeatureDist);
  maxFrameGap      = abs(getEnv<int>  (jpr,"TRACKING_MAX_FRAME_GAP",         maxFrameGap));       LOG4CXX_TRACE(_log, "TRACKING_MAX_FRAME_GAP: " << maxFrameGap);
  maxCenterDist    = abs(getEnv<float>(jpr,"TRACKING_MAX_CENTER_DIST",       maxCenterDist));     LOG4CXX_TRACE(_log, "TRACKING_MAX_CENTER_DIST: " << maxCenterDist);
  maxIOUDist       = abs(getEnv<float>(jpr,"TRACKING_MAX_IOU_DIST",          maxIOUDist));        LOG4CXX_TRACE(_log, "TRACKING_MAX_IOU_DIST: " << maxIOUDist);

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
      <<  "\"maxIOUDist\":"       << cfg.maxIOUDist
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
  frameIdx(-1),
  lastError(MPF_DETECTION_SUCCESS),
  _imreaderPtr(unique_ptr<MPFImageReader>()),
  _videocapPtr(unique_ptr<MPFVideoCapture>()){}

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