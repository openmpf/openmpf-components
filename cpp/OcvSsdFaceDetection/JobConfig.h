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
  *
  * \param T   data type to cast the result to
  * \param p   properties map to get property from
  * \param k   string key to use to retrieve property
  * \param def default value to return if key is not found
  *
  * \return  type converted value of property retrived with key or the default
  *
  ***************************************************************************** */
  template<typename T>
  T get(const Properties &p, const string &k, const T def){
    return DetectionComponentUtils::GetProperty<T>(p,k,def);
  }

  /** ****************************************************************************
  *   get configuration from environment variables if not
  *   provided by job configuration
  *
  * \param T   data type to cast the result to
  * \param p   properties map to get property from
  * \param k   string key to use to retrieve property
  * \param def default value to return if key is not found
  *
  * \return  type converted value of property retrived with key or the default
  *
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



  /* **************************************************************************
  *   Configuration parameters populated with appropriate values & defaults
  *************************************************************************** */
  class JobConfig{
    public:
      static    log4cxx::LoggerPtr _log;  ///< shared log object
      size_t    minDetectionSize;         ///< minimum bounding box dimension
      float     confThresh;               ///< detection confidence threshold
      float     nmsThresh;                ///< non-maximum suppression threshold for removing redundant overlapping bounding boxes
      float     bboxScaleFactor;          ///< scale factor for  width and height of the detector bounding box
      bool      rotateDetect;             ///< perform multiple passes at different image rotations to increase detections found
      OrientVec inferenceOrientations;    ///< ccw rotations of frame to inference (only multiples of 90 are accepted)
      int       inferenceSize;            ///< max image dimension to use for inferencing e.g. 300 (-1 will use original but run slower)
      long      detFrameInterval;         ///< number of frames between looking for new detection (tracking only)

      float     maxFeatureDist;           ///< maximum feature distance to maintain track continuity
      float     maxCenterDist;            ///< maximum spatial distance normalized by diagonal to maintain track continuity
      long      maxFrameGap;              ///< maximum temporal distance (frames) to maintain track continuity
      float     maxIOUDist;               ///< maximum for (1 - Intersection/Union) to maintain track continuity

      float     widthOdiag;              ///< image (width/diagonal)
      float     heightOdiag;             ///< image (height/diagonal)
      size_t    frameIdx;                ///< index of current frame
      double    frameTimeInSec;          ///< time of current frame in sec
      double    frameTimeStep;           ///< time interval between frames in sec

      cv::Mat   bgrFrame;                ///< current BGR image frame

      bool  kfDisabled;                ///< if true kalman filtering is disabled
      cv::Mat_<float> RN;              ///< kalman filter measurement noise matrix
      cv::Mat_<float> QN;              ///< kalman filter process noise variances (i.e. unknown accelerations)

      bool fallback2CpuWhenGpuProblem; ///< fallback to cpu if there is a gpu problem
      int  cudaDeviceId;               ///< gpu device id to use for cuda

      mutable MPFDetectionError   lastError;   ///< last MPF error that should be returned

      JobConfig();
      JobConfig(const MPFImageJob &job);
      JobConfig(const MPFVideoJob &job);
      ~JobConfig();

      void ReverseTransform(MPFImageLocation loc){_imreaderPtr->ReverseTransform(loc);}
      void ReverseTransform(MPFVideoTrack  track){_videocapPtr->ReverseTransform(track);}
      bool nextFrame();

    private:
      unique_ptr<MPFImageReader>  _imreaderPtr;
      unique_ptr<MPFVideoCapture> _videocapPtr;
      string                      _strRN;           ///< kalman filter measurement noise matrix serialized to string
      string                      _strQN;           ///< kalman filter process noise matrix serialized to string
      string                      _strOrientations; ///< ccw rotations of frame to inference (only multiples of 90 are accepted)

      void    _parse(const MPFJob &job);

  };

  ostream& operator<< (ostream& out, const JobConfig& cfg);  ///< Dump JobConfig to a stream

 }
}

#endif