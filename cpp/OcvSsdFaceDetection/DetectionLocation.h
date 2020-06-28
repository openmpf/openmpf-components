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

#ifndef OCVSSDFACEDETECTION_DETECTIONLOCATION_H
#define OCVSSDFACEDETECTION_DETECTIONLOCATION_H

#include <log4cxx/logger.h>

#include <opencv2/dnn.hpp>
#include <opencv2/face.hpp>

#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include "types.h"
#include "JobConfig.h"

#define CALL_MEMBER_FUNC(object,ptrToMember)  ((object).*(ptrToMember))

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  typedef float (DetectionLocation::*DetectionLocationCostFunc)(const Track &tr) const; ///< cost member-function pointer type

  class DetectionLocation: public MPFImageLocation{ // extend MPFImageLocation

    public:
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors for MPFImageLocation

      const cv::Point2f       center;            ///< bounding box center normalized to image dimensions
      const size_t            frameIdx;          ///< frame index frame where detection is located (for videos)
      const double            frameTimeInSec;    ///< frame time in sec where detection is located (for videos)

      static bool Init(log4cxx::LoggerPtr log, const string plugin_path);    ///< setup class shared members
      static DetectionLocationPtrVec createDetections(const JobConfig &cfg); ///< created detection objects from image frame

      const cvPoint2fVec&  getLandmarks() const;                             ///< get landmark points for detection
      const cv::Mat&       getThumbnail() const;                             ///< get thumbnail image for detection
      const cv::Mat&       getFeature()   const;                             ///< get DNN features for detection
      const cv::Mat&       getBGRFrame()  const;                             ///< get image data associated with detection

      const cv::Rect2i  getRect() const;                                     ///< get location as an opencv rectange
      void              setRect(const cv::Rect2i& rec);                      ///< set location from an opencv rectangle

      void copyFeature(const DetectionLocation& d);                          ///< copy DNN feature from another detection

      float           iouDist(const Track &tr) const;             ///< 1 - compute intersection over union
      float         frameDist(const Track &tr) const;             ///< compute temporal frame gap
      float center2CenterDist(const Track &tr) const;             ///< compute normalized center to center distance
      float       featureDist(const Track &tr) const;             ///< compute deep feature similarity distance

      void drawLandmarks(cv::Mat &img, const cv::Scalar drawColor) const;    ///< draw landmark point on image
      void releaseBGRFrame();                                                ///< release reference to image frame
      static bool trySetCudaDevice(const int cudaDeviceId);                  ///< try set CUDA to use specified GPU device

      DetectionLocation(int x,int y,int width,int height,float conf,
                        cv::Point2f center,
                        size_t frameIdx, double frameTimeInMillis,
                        cv::Mat bgrFrame);                                   ///< private constructor for createDetections()
    private:

      static log4cxx::LoggerPtr                _log;                ///< shared log object
      static cv::dnn::Net                      _ssdNet;             ///< single shot DNN face detector network
      static cv::dnn::Net                      _openFaceNet   ;     ///< feature generator
      static unique_ptr<dlib::shape_predictor> _shapePredFuncPtr;   ///< landmark detector

      float _iouDist(const cv::Rect2i &rect) const;                 ///< compute intersectino over union

      mutable cvPoint2fVec    _landmarks;                 ///< vector of landmarks (e.g. eyes, nose, etc.)
      mutable cv::Mat         _thumbnail;                 ///< 96x96 image comprising an aligned thumbnail
      mutable cv::Mat         _feature;                   ///< DNN feature for matching-up detections
      cv::Mat                 _bgrFrame;                  ///< frame associated with detection (openCV memory managed :( )

      static void _setCudaBackend(const bool enabled);    ///< turn on or off cuda backend for inferencing
  };

  /** **************************************************************************
  *   Dump MPFLocation to a stream
  *************************************************************************** */
  inline
  ostream& operator<< (ostream& out, const DetectionLocation& d) {
    out  << "[" << (MPFImageLocation)d
                << " F[" << d.getFeature().size() << "] T["
                << d.getThumbnail().rows << "," << d.getThumbnail().cols << "]";
    return out;
  }

 }
}

#endif