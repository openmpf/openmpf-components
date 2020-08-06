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

#ifndef OCVYOLODETECTION_Config_H
#define OCVYOLODETECTION_Config_H

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

  /** ****************************************************************************
  * logging shorthand macros
  ****************************************************************************** */
  #define LOG_TRACE(MSG){ LOG4CXX_TRACE(Config::log, MSG) }
  #define LOG_DEBUG(MSG){ LOG4CXX_DEBUG(Config::log, MSG) }
  #define LOG_INFO( MSG){ LOG4CXX_INFO (Config::log, MSG) }
  #define LOG_WARN( MSG){ LOG4CXX_WARN (Config::log, MSG) }
  #define LOG_ERROR(MSG){ LOG4CXX_ERROR(Config::log, MSG) }
  #define LOG_FATAL(MSG){ LOG4CXX_FATAL(Config::log, MSG) }

  /* **************************************************************************
  *   Configuration parameters populated with appropriate values & defaults
  *************************************************************************** */
  class Config{
    public:
      float   confThresh;               ///< detection confidence threshold
      float   nmsThresh;                ///< non-maximum supresssion threshold to remove redundant bboxes
      int     inputImageSize;           ///< network imge input size (320, 416, 608)
      int     numClassPerRegion;        ///< number of class labels and confidence scores to return for a bbox
      long    detFrameInterval;         ///< number of frames between looking for new detection (tracking only)

      float   maxFeatureDist;           ///< maximum feature distance to maintain track continuity
      float   maxCenterDist;            ///< maximum spatial distance normalized by diagonal to maintain track continuity
      long    maxFrameGap;              ///< maximum temporal distance (frames) to maintain track continuity
      float   maxIOUDist;               ///< maximum for (1 - Intersection/Union) to maintain track continuity

      int     dftSize;                  ///< size of dft used for bbox alignment
      bool    dftHannWindowEnabled;     ///< use hanning windowing with dft

      float   widthOdiag;               ///< image (width/diagonal)
      float   heightOdiag;              ///< image (height/diagonal)
      float   aspectRatio;              ///< image aspect ratio (width/height)

      size_t  frameIdx;                 ///< index of current frame
      double  frameTimeInSec;           ///< time of current frame in sec
      double  frameTimeStep;            ///< time interval between frames in sec
      cv::Mat bgrFrame;                 ///< current BGR image frame

      bool    kfDisabled;               ///< if true kalman filtering is disabled
      cv::Mat1f RN;               ///< kalman filter measurement noise matrix
      cv::Mat1f QN;               ///< kalman filter process noise variances (i.e. unknown accelerations)

      bool fallback2CpuWhenGpuProblem; ///< fallback to cpu if there is a gpu problem
      int  cudaDeviceId;               ///< gpu device id to use for cuda

      MPFDetectionError   lastError;   ///< last MPF error that should be returned

      // configuration values shared by all jobs
      static string             pluginPath;        ///< path to the plugin
      static string             configPath;        ///< path to configuration file(s)
      static log4cxx::LoggerPtr log;               ///< shared log object

      Config();
      Config(const MPFImageJob &job);
      Config(const MPFVideoJob &job);
      ~Config();

      void ReverseTransform(MPFImageLocation loc) const {_imreaderPtr->ReverseTransform(loc);}
      void ReverseTransform(MPFVideoTrack  track) const {_videocapPtr->ReverseTransform(track);}
      bool nextFrame();

    private:
      unique_ptr<MPFImageReader>  _imreaderPtr;
      unique_ptr<MPFVideoCapture> _videocapPtr;

      static cv::Mat1f      _defaultHanningWindow;  ///< default hannning window matching _dftSize

      string                      _strRN;                 ///< kalman filter measurement noise matrix serialized to string
      string                      _strQN;                 ///< kalman filter process noise matrix serialized to string

      void    _parse(const MPFJob &job);
      cv::Mat _fromString(const string data,
                          const int    rows,
                          const int    cols,
                          const string dt);

  };

  ostream& operator<< (ostream& out, const Config& cfg);  ///< Dump Config to a stream

 }
}

#endif