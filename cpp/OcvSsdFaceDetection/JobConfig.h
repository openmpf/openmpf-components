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

#ifndef OCVSSDFACEDETECTION_JOBCONFIG_H
#define OCVSSDFACEDETECTION_JOBCONFIG_H

#include <log4cxx/logger.h>
#include <opencv2/opencv.hpp>

#include "detectionComponentUtils.h"
#include "adapters/MPFImageAndVideoDetectionComponentAdapter.h"
#include "MPFImageReader.h"
#include "MPFVideoCapture.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;


  /** ****************************************************************************
  *   get MPF properties of various types
  ***************************************************************************** */
  template<typename T>
  T get(const Properties &p, const string &k, const T def){
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  *   get configuration from environment variables if not
  *   provided by job configuration
  ***************************************************************************** */
  template<typename T>
  T getEnv(const Properties &p, const string &k, const T def){
    auto iter = p.find(k);
    if (iter == p.end()){
      const char* env_p = getenv(k.c_str());
      if(env_p != NULL){
        map<string,string> envp;
        envp.insert(pair<string,string>(k,string(env_p)));
        return DetectionComponentUtils::GetProperty<T>(envp,k,def);
      }else{
        return def;
      }
    }
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  *   exception macro so we can see where in the code it happened
  ****************************************************************************** */
  #define THROW_EXCEPTION(MSG){                                  \
  string path(__FILE__);                                         \
  string f(path.substr(path.find_last_of("/\\") + 1));           \
  throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
  }

  // Kalman Filter Constants
  const int KF_STATE_DIM = 8; ///< dimensionality of kalman state vector       [x, y, v_x, v_y, w, h, v_w, v_h]
  const int KF_MEAS_DIM  = 4; ///< dimensionality of kalman measurement vector [x, y, w, h]
  const int KF_CTRL_DIM  = 0; ///< dimensionality of kalman control input      []

  /* **************************************************************************
  *   Configuration parameters populated with appropriate values & defaults
  *************************************************************************** */
  class JobConfig{
    public:
      static log4cxx::LoggerPtr _log;  ///< shared log object
      size_t minDetectionSize;         ///< minimum bounding box dimension
      float  confThresh;               ///< detection confidence threshold
      long   detFrameInterval;         ///< number of frames between looking for new detection (tracking only)

      float  maxFeatureDist;           ///< maximum feature distance to maintain track continuity
      float  maxCenterDist;            ///< maximum spatial distance normalized by diagonal to maintain track continuity
      long   maxFrameGap;              ///< maximum temporal distance (frames) to maintain track continuity
      float  maxIOUDist;               ///< maximum for (1 - Intersection/Union) to maintain track continuity

      float   widthOdiag;              ///< image (width/diagonal)
      float   heightOdiag;             ///< image (height/diagonal)
      size_t  frameIdx;                ///< index of current frame
      cv::Mat bgrFrame;                ///< current BGR image frame


      bool  kfDisabled;                ///< if true kalman filtering is disabled


      cv::Mat_<float> H;          ///< kalman filter measurement matrix
      cv::Mat_<float> R;          ///< kalman filter measurement noise matrix
      cv::Mat_<float> Q;          ///< kalman filter process noise matrix

      MPFDetectionError   lastError;   ///< last MPF error that should be returned

      JobConfig();
      JobConfig(const MPFImageJob &job);
      JobConfig(const MPFVideoJob &job);
      ~JobConfig();

      void ReverseTransform(MPFImageLocation loc){_imreaderPtr->ReverseTransform(loc);}
      void ReverseTransform(MPFVideoTrack  track){_videocapPtr->ReverseTransform(track);}
      const double GetCurrentTimeInMillis() const;
      bool nextFrame();
      inline double GetCurrentTimeInMillis() const {  return _videocapPtr->GetCurrentTimeInMillis();}

    private:
      mutable double              _frameTimeInMillis;  ///< time of current frame in milli sec
      unique_ptr<MPFImageReader>  _imreaderPtr;
      unique_ptr<MPFVideoCapture> _videocapPtr;

      void _parse(const MPFJob &job);


  };

  ostream& operator<< (ostream& out, const JobConfig& cfg);  ///< Dump JobConfig to a stream

 }
}

#endif