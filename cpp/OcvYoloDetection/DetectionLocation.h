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
#include "Frame.h"

#define CALL_MEMBER_FUNC(object,ptrToMember)  ((object).*(ptrToMember))

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  typedef float (DetectionLocation::*DetectionLocationCostFunc)(const Track &tr) const; ///< cost member-function pointer type

  class DetectionLocation: public MPFImageLocation{ // extend MPFImageLocation

    public:
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors for MPFImageLocation

      cv::Point2f       center;            ///< bounding box center normalized to image dimensions
      const FramePtr    framePtr;          ///< frame associated with detection (openCV memory managed :( )

      const cv::Mat&       getClassFeature() const;        ///< get unit vector of scores
      const cv::Mat&       getDFTFeature()   const;        ///< get dft for phase correlation
      const cv::Rect2i     getRect()         const;        ///< get location   as an opencv rectange
      void                 setRect(const cv::Rect2i& rec); ///< set location from an opencv rectangle

      float           iouDist(const Track &tr) const;   ///< 1 - compute intersection over union
      float         kfIouDist(const Track &tr) const;   ///< 1 - compute intersection over union using kalman predicted location
      float         frameDist(const Track &tr) const;   ///< compute temporal frame gap
      float center2CenterDist(const Track &tr) const;   ///< compute normalized center to center distance
      float       featureDist(const Track &tr) const;   ///< compute deep feature similarity distance
      float         classDist(const Track &tr) const;   ///< compute class similarity distance
      float    kfResidualDist(const Track &tr) const;   ///< comput kalman filter residual distance

      static bool                       Init();                                                                ///< setup class shared members
      static DetectionLocationPtrVecVec createDetections(const ConfigPtr cfgPtr,const FramePtrVec &framePtrs); ///< created detection objects from image frame
      static bool                       loadNetToCudaDevice(const int cudaDeviceId);                           ///< load network to active CUDA device
      static cv::Rect2i snapToEdges(const cv::Rect2i& rt, const cv::Rect2i& rm,
                                    const cv::Size2i& frameSize, const float edgeSnapDist);                    ///< snap a rectangle to frame edges if close

      DetectionLocation(const ConfigPtr   cfgPtr,
                        const FramePtr    frmPtr,
                        const cv::Rect2d  bbox,
                        const float       conf,
                        const cv::Point2f ctr,
                        const cv::Mat     classFeature);

      DetectionLocation(const ConfigPtr   cfgPtr,
                        const FramePtr    frmPtr,
                        const cv::Rect2d  bbox,
                        const float       conf,
                        const cv::Point2f ctr,
                        const cv::Mat     classFeature,
                        const cv::Mat     dftFeature);


    private:

      const ConfigPtr     _cfgPtr;               ///< job configuration and shared config state
      const cv::Mat       _classFeature;         ///< unit vector of with elements proportional to scores for each classes
      mutable cv::Mat     _dftFeature;           ///< dft for matching-up detections via phase correlation

      static cv::dnn::Net _net;                  ///< DNN detector network
      static stringVec    _netClasses;           ///< list of classes for DNN
      static stringVec    _netOutputNames;       ///< list of DNN output names
      static cv::Mat1f    _netConfusion;         ///< classifier confusion matrix
      static intVec       _classGroupIdx;        ///< class index to class group mapping from confusion matrix

      float              _iouDist(const cv::Rect2i &rect) const;    ///< compute intersection over union

      cv::Point2d       _phaseCorrelate(  const Track      &tr)    const;  ///< get bbox alignment via phase correlation
      const cv::Mat1f   _getHanningWindow(const cv::Size   &size)  const;  ///< get hanning window of specified size

      static void _loadNet(const bool useCUDA);    ///< turn on or off cuda backend for inferencing
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