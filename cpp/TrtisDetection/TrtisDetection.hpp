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


#ifndef OPENMPF_COMPONENTS_TRTISDETECTION_H
#define OPENMPF_COMPONENTS_TRTISDETECTION_H

#include <log4cxx/logger.h>
#include <adapters/MPFImageAndVideoDetectionComponentAdapter.h>

// Nvidia TensorRT Inference Server (trtis) client lib includes
// (see https://github.com/NVIDIA/tensorrt-inference-server)

#include "request_grpc.h"
#include "request_http.h"
#include "model_config.pb.h"

namespace MPF{
 namespace COMPONENT{

  namespace       ni   = nvidia::inferenceserver;             ///< namespace alias for inference server
  namespace       nic  = nvidia::inferenceserver::client;

  using namespace std;

  typedef vector<uint8_t>              BytVec;                ///< vector of bytes
  typedef vector<int>                  IntVec;                ///< vector of integers
  typedef vector<int64_t>              LngVec;                ///< vector of 64bit integers
  typedef vector<float>                FltVec;                ///< vector of floats

  typedef vector<MPFVideoTrack>        MPFVideoTrackVec;
  typedef vector<MPFImageLocation>     MPFImageLocationVec;

  typedef nic::InferContext            InferCtx;              ///< trtis inferencing message context
  typedef nic::InferContext::Input     InferCtxInp;           ///< inferencing message input context
  typedef nic::InferContext::Result    InferCtxRes;           ///< inferencing message output context
  typedef nic::InferContext::Options   InferCtxOpt;           ///< inferencing message context options
  typedef nic::InferContext::Request   InferCtxReq;           ///< inferencing context request
  typedef unique_ptr<InferCtx>         uPtrInferCtx;          ///< inferencing message context  pointer
  typedef unique_ptr<InferCtxOpt>      uPtrInferCtxOpt;       ///< inference options pointer
  typedef unique_ptr<InferCtxRes>      uPtrInferCtxRes;       ///< inference results pointer
  typedef shared_ptr<InferCtxInp>      sPtrInferCtxInp;       ///< inference input context pointer
  typedef shared_ptr<InferCtxReq>      sPtrInferCtxReq;       ///< inference request context pointer
  typedef map<string,uPtrInferCtxRes>  StrUPtrInferCtxResMap; ///< map of inference outputs keyed by output name

  class TrtisJobConfig{
    public:
      string trtis_server;                 ///< url with port for trtis server e.g. localhost:8001
      string model_name;                   ///< name of model as served by trtis
      int model_version;                   ///< version of model (e.g. -1 for latest)
      size_t maxInferConcurrency;          ///< maximum number of concurrent video frame inferencing request
      size_t contextWaitTimeoutSec;        ///< max time to wait for an inference context from pool in sec

      TrtisJobConfig(const MPFJob &job);
  };

  class TrtisIpIrv2CocoJobConfig : public TrtisJobConfig{
    public:

      bool   clientScaleEnabled;          ///< perform image scaling client side
      bool   frameFeatEnabled;            ///< process frame average feature
      bool   classFeatEnabled;            ///< process recognized coco objects
      bool   extraFeatEnabled;            ///< process extra unclassified object
      bool   userFeatEnabled;             ///< process user feature per BBox
      size_t image_width;                 ///< width of image or frame
      size_t image_height;                ///< height of image or frame
      size_t image_x_max;                 ///< maximum x pixel coordinate (width - 1)
      size_t image_y_max;                 ///< maximum y pixel coordinate (height - 1)
      size_t userBBox_x;                  ///< user bounding box upper left x1
      size_t userBBox_y;                  ///< user bounding box upper left y1
      size_t userBBox_width;              ///< user bounding box width
      size_t userBBox_height;             ///< user bounding box height
      LngVec userBBox;                    ///< user bounding box as [y1,x1,x2,x2]
      FltVec userBBoxNorm;                ///< user bounding box normalized with image dimensions
      bool   recognitionEnroll;           ///< enroll features in recognition framework

      float  confThreshold;               ///< object detection confidence threshold
      float  extraConfThreshold;          ///< extra detections confidence threshold
      float  maxFeatureGap;               ///< max distance of object track members in feature space
      size_t maxFrameGap;                 ///< max distance of object track members in frame space
      float  maxSpaceGap;                 ///< max center to center spacial distance of object track members normalized with image diagonal
      float  maxSpaceGapPxSq;             ///< squared center to center distance in pixels

      TrtisIpIrv2CocoJobConfig(const MPFJob &job,
                               const size_t image_width,
                               const size_t image_height);
  };

  class TrtisDetection : public MPFImageAndVideoDetectionComponentAdapter {
    public:
      bool Init() override;
      bool Close() override;
      MPFDetectionError GetDetections(const MPFVideoJob &job,
                                      MPFVideoTrackVec  &tracks) override;
      MPFDetectionError GetDetections(const MPFImageJob   &job,
                                      MPFImageLocationVec &locations) override;
      string GetDetectionType() override;

    private:

      log4cxx::LoggerPtr                 _log;            ///< log object
      map<string,vector<string>>         _class_labels;   ///< possible class labels keyed by model names
      map<string, vector<uPtrInferCtx*>> _infCtxs;        ///< pool of inference contexts for models

      void _readClassNames(string model,
                           string class_label_file,
                           int    class_label_count);                           ///< read in class labels for a model from a file

      vector<uPtrInferCtx*> _niGetInferContexts(const TrtisJobConfig cfg);      ///< get cached inference contexts

      static string  _niType2Str(ni::DataType dt);                              ///< nvidia data type to string
      cv::Mat        _niResult2CVMat(const size_t batch_idx,
                                     const string name,
                                     StrUPtrInferCtxResMap& results);           ///< make an openCV mat header for nvidia tensor

      cv::Mat _cvResize(const cv::Mat &img,
                        double        &scaleFactor,
                        const size_t target_width,
                        const size_t target_height);                            ///< aspect preserving resize image to ~[target_width, target_height]

      BytVec  _cvRGBBytes(const cv::Mat &img,
                          LngVec        &shape);                                ///< convert image to 8-bit RGB

      void _ip_irv2_coco_prepImageData(const TrtisIpIrv2CocoJobConfig &cfg,
                                       const cv::Mat                  &img,
                                       const uPtrInferCtx*             ctx,
                                       LngVec                        &shape,
                                       BytVec                        &imgDat);  ///< prep image for inferencing

      void _ip_irv2_coco_getDetections(const TrtisIpIrv2CocoJobConfig &cfg,
                                       StrUPtrInferCtxResMap          &res,
                                       MPFImageLocationVec       &locations);   ///< parse inference results and get detections

      void _ip_irv2_coco_tracker(const TrtisIpIrv2CocoJobConfig &cfg,
                                 MPFImageLocation               &loc,
                                 const int                       frameIdx,
                                 MPFVideoTrackVec               &tracks);       ///< tracking using time, space and feature proximity

      void _base64DecodeStopFeatures(MPFVideoTrackVec &tracks);                 ///< base64 decode "FEATURE" property of stop location
      void _base64EncodeStopFeatures(MPFVideoTrackVec &tracks);                 ///< base64 encode "FEATURE" property of stop location

      void _addToTrack(MPFImageLocation &location,
                       int               frame_index,
                       MPFVideoTrack     &track);                               ///< add location to a track

  };
 }
}
#endif //OPENMPF_COMPONENTS_TRTISDETECTION_H
