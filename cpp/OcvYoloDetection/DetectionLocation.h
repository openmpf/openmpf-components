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

#include "types.h"
#include "util.h"
#include "Config.h"
#include "Frame.h"

namespace MPF{
 namespace COMPONENT{

  using namespace std;

  using DetectionLocationCostFunc = float (DetectionLocation::*)(const Track &tr) const; ///< assignment cost member-function type

  class DetectionLocation: public MPFImageLocation{ // extend MPFImageLocation

    public:
      using MPFImageLocation::MPFImageLocation;  // C++11 inherit all constructors for MPFImageLocation

      const Frame frame;                                   ///< frame associated with detection

      const cv::Mat&       getClassFeature() const;        ///< get unit vector of scores
      const cv::Mat&       getDFTFeature()   const;        ///< get dft for phase correlation
      const cv::Rect2i     getRect()         const;        ///< get location   as an opencv rectange
      void                 setRect(const cv::Rect2i& rec); ///< set location from an opencv rectangle

      float           iouDist(const Track &tr) const;   ///< 1 - compute intersection over union
      float center2CenterDist(const Track &tr) const;   ///< compute normalized center to center distance
      float       featureDist(const Track &tr) const;   ///< compute deep feature similarity distance
      float    kfResidualDist(const Track &tr) const;   ///< compute kalman filter residual distance

      static bool                       Init();                                                                ///< setup class shared members
      static DetectionLocationListVec   createDetections(const Config &cfg,const FrameVec &frames);            ///< created detection objects from image frame
      static bool                       loadNetToCudaDevice(const int cudaDeviceId);                           ///< load network to active CUDA device

      DetectionLocation(const Config     &cfg,
                        const Frame       frm,
                        const cv::Rect2d  bbox,
                        const float       conf,
                        const cv::Mat     classFeature,
                        const cv::Mat     dftFeature);

      DetectionLocation(DetectionLocation &&d):MPFImageLocation(move(d)),
                                               frame(move(d.frame)),
                                               _cfg(d._cfg),
                                               _classFeature(move(d._classFeature)),
                                               _dftFeature(move(d._dftFeature))
                                               {}

    private:
      const Config       &_cfg;                  ///< job configuration and shared config state
      const cv::Mat     _classFeature;           ///< unit vector of with elements proportional to scores for each classes
      const cv::Mat     _dftFeature;             ///< dft for matching-up detections via phase correlation

      static cv::dnn::Net _net;                  ///< DNN detector network
      static stringVec    _netClasses;           ///< list of classes for DNN
      static stringVec    _netOutputNames;       ///< list of DNN output names
      static cv::Mat1f    _netConfusion;         ///< classifier confusion matrix
      static intVec       _classGroupIdx;        ///< class index to class group mapping from confusion matrix

      float _iouDist(const cv::Rect2i &rect) const;    ///< compute intersection over union

      cv::Point2d     _phaseCorrelate(  const Track      &tr)    const;  ///< get bbox alignment via phase correlation
      const cv::Mat1f _getHanningWindow(const cv::Size   &size)  const;  ///< get hanning window of specified size

      static void _loadNet(const bool useCUDA);    ///< turn on or off cuda backend for inferencing
  };

  /** **************************************************************************
  *   Dump DetectionLocation to a stream
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