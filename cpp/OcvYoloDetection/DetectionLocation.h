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

#ifndef OCVYOLODETECTION_DETECTIONLOCATION_H
#define OCVYOLODETECTION_DETECTIONLOCATION_H

#include <log4cxx/logger.h>

#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

//#include <dlib/image_processing.h>
//#include <dlib/opencv.h>

#include "types.h"
#include "Config.h"

#define CALL_MEMBER_FUNC(object,ptrToMember)  ((object).*(ptrToMember))

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  typedef float (DetectionLocation::*DetectionLocationCostFunc)(const Track &tr) const; ///< cost member-function pointer type

  class DetectionLocation: public MPFImageLocation{ // extend MPFImageLocation

    public:
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors for MPFImageLocation

      cv::Point2f       center;            ///< bounding box center normalized to image dimensions
      size_t            frameIdx;          ///< frame index frame where detection is located (for videos)
      double            frameTimeInSec;    ///< frame time in sec where detection is located (for videos)

      static bool Init();                                                      ///< setup class shared members
      static DetectionLocationPtrVec createDetections(const ConfigPtr cfgPtr); ///< created detection objects from image frame

      const cv::Mat&       getFeature()  const;  ///< get dft for phase correlation
      const cv::Mat&       getBGRFrame() const;  ///< get cropped image data associated with detection

      const cv::Rect2i  getRect() const;                ///< get location   as an opencv rectange
      void              setRect(const cv::Rect2i& rec); ///< set location from an opencv rectangle

      void copyFeature(const DetectionLocation& d) const; ///< copy DNN feature from another detection

      float           iouDist(const Track &tr) const;   ///< 1 - compute intersection over union
      float         kfIouDist(const Track &tr) const;   ///< 1 - compute intersection over union using kalman predicted location
      float         frameDist(const Track &tr) const;   ///< compute temporal frame gap
      float center2CenterDist(const Track &tr) const;   ///< compute normalized center to center distance
      float       featureDist(const Track &tr) const;   ///< compute deep feature similarity distance

      void releaseBGRFrame();                                ///< release reference to image frame
      static bool trySetCudaDevice(const int cudaDeviceId);  ///< try set CUDA to use specified GPU device

      DetectionLocation(const ConfigPtr   cfgPtr,
                        const cv::Rect2d  bbox,
                        const float       conf,
                        const cv::Point2f center);     ///< constructor for createDetections()

    private:

      const ConfigPtr     _cfgPtr;               ///< job configuration and shared config state
      mutable cv::Mat     _feature;              ///< dft for matching-up detections via phase correlation
      cv::Mat             _bgrFrame;             ///< frame associated with detection (openCV memory managed :( )

      static cv::dnn::Net       _net;                  ///< DNN detector network
      static stringVec          _netClasses;           ///< list of classes for DNN
      static stringVec          _netOutputNames;       ///< list of DNN output names

      float              _iouDist(const cv::Rect2i &rect) const;    ///< compute intersectino over union
      cv::Point2d _phaseCorrelate(const Track      &tr)   const;    ///< get bbox alignment via phase correlation

      static void _setCudaBackend(const bool enabled);    ///< turn on or off cuda backend for inferencing
  };

  /** **************************************************************************
  *   Dump MPFLocation to a stream
  *************************************************************************** */
  inline
  ostream& operator<< (ostream& out, const DetectionLocation& d) {
    out  << "[" << (MPFImageLocation)d
         << "]";
    return out;
  }

 }
}

#endif