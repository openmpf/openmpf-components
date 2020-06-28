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

#include <cassert>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

#include "DetectionLocation.h"
#include "Track.h"

using namespace MPF::COMPONENT;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr                DetectionLocation::_log;
cv::dnn::Net                      DetectionLocation::_ssdNet;                     ///< single shot DNN face detector network
cv::dnn::Net                      DetectionLocation::_openFaceNet;                ///< feature generator
unique_ptr<dlib::shape_predictor> DetectionLocation::_shapePredFuncPtr  = NULL;   ///< landmark detector function pointer

/** ****************************************************************************
*  Draw polylines to visualize landmark features
*
* \param im        image to draw landmarks on
* \param landmarks all landmark point only some of which will be drawn
* \param start     start landmark point index for polyline
* \param end       end landmark point index for polyline
* \param isClosed  if true polyline draw a closed shape (end joined to start)
* \param drawColor color to use for drawing
*
***************************************************************************** */
void drawPolyline(cv::Mat &im, const cvPoint2fVec &landmarks,
                  const int start, const int end, bool isClosed = false,
                  const cv::Scalar drawColor = cv::Scalar(255, 200,0)){
    cvPointVec points;
    for (int i = start; i <= end; i++){
        points.push_back(cv::Point(landmarks[i].x, landmarks[i].y));
    }
    cv::polylines(im, points, isClosed, drawColor, 2, 16);
}

/** ****************************************************************************
* Visualize landmark point on image by drawing them.  If 68 landmarks are
* available, they are drawn as polygons, otherwise just as points
*
* \param [in,out] im        image to draw landmarks on
* \param          drawColor color to use for drawing
*
***************************************************************************** */
void DetectionLocation::drawLandmarks(cv::Mat &img,
                   const cv::Scalar drawColor = cv::Scalar(255, 200,0)) const {
  cvPoint2fVec landmarks = getLandmarks();
  if(landmarks.size() == 68){
    drawPolyline(img, landmarks,  0, 16, false, drawColor);    // Jaw line
    drawPolyline(img, landmarks, 17, 21, false, drawColor);    // Left eyebrow
    drawPolyline(img, landmarks, 22, 26, false, drawColor);    // Right eyebrow
    drawPolyline(img, landmarks, 27, 30, false, drawColor);    // Nose bridge
    drawPolyline(img, landmarks, 30, 35, true,  drawColor);    // Lower nose
    drawPolyline(img, landmarks, 36, 41, true,  drawColor);    // Left eye
    drawPolyline(img, landmarks, 42, 47, true,  drawColor);    // Right Eye
    drawPolyline(img, landmarks, 48, 59, true,  drawColor);    // Outer lip
    drawPolyline(img, landmarks, 60, 67, true,  drawColor);    // Inner lip
  }else {
    for(size_t i = 0; i < landmarks.size(); i++){
      cv::circle(img,landmarks[i],3, drawColor, cv::FILLED);
    }
  }
}

/** **************************************************************************
* Compute (1 - Intersection Over Union) metric between a rectangel and detection
* comprised of 1 - the ratio of the area of the intersection of the detection
* rectangles divided by the area of the union of the detection rectangles.
*
* \param   rect rectangle to compute iou with
* \returns 1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::_iouDist(const cv::Rect2i &rect) const {
  int ulx = max(x_left_upper         , rect.x           );
  int uly = max(y_left_upper         , rect.y           );
  int lrx = min(x_left_upper + width , rect.x + rect.width );
  int lry = min(y_left_upper + height, rect.y + rect.height);

  float inter_area = max(0, lrx - ulx) * max(0, lry - uly);
  float union_area = width * height + rect.width * rect.height - inter_area;
  float dist = 1.0f - inter_area / union_area;                                 LOG4CXX_TRACE(_log,"iou dist = " << dist);
  return dist;
}

/** **************************************************************************
* Compute 1 - Intersection Over Union metric between track tail and detection
*
* \param   tr track
* \returns 1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::iouDist(const Track &tr) const {
  return _iouDist(cv::Rect2i(tr.back()->x_left_upper,
                             tr.back()->y_left_upper,
                             tr.back()->width,
                             tr.back()->height));
}

/** **************************************************************************
* Compute the temporal distance (frame count) between track tail and detection
*
* \param   tr track
* \returns   absolute difference in frame indices
*
*************************************************************************** */
float DetectionLocation::frameDist(const Track &tr) const {
  if(frameIdx > tr.back()->frameIdx){
    return frameIdx - tr.back()->frameIdx;
  }else{
    return tr.back()->frameIdx - frameIdx;
  }
}

/** **************************************************************************
* Compute euclidean center to distance center distance from normalized centers
*
* \param   tr track
* \returns   normalized center to center distance [0 ... Sqrt(2)]
*
*************************************************************************** */
float  DetectionLocation::center2CenterDist(const Track &tr) const {
  float dx = center.x - tr.back()->center.x;
  float dy = center.y - tr.back()->center.y;
  float dist = sqrt( dx*dx + dy*dy );                                          LOG4CXX_TRACE(_log,"center-2-center dist = " << dist);
  return dist;
}


/** **************************************************************************
* Compute feature distance (similarity) to track's tail detection's
* feature vectors
*
* \param   tr track
* \returns cos distance [0 ... 1.0]
*
* \note Feature vectors are expected to be of unit magnitude
*
*************************************************************************** */
float DetectionLocation::featureDist(const Track &tr) const {
  float dist = 1.0f - max(0.0f,static_cast<float>(getFeature().dot(tr.back()->getFeature())));  LOG4CXX_TRACE(_log,"feature dist = " << dist);
  return dist;
}

/** **************************************************************************
* Lazy accessor method to get/compute landmark points
* 5-Landmark detector returns outside and inside eye corners and bottom of nose
* 68-Landmark detector returns "standard" facial landmarks (see data/landmarks.png)
*
* \returns landmarks
*
*************************************************************************** */
const cvPoint2fVec& DetectionLocation::getLandmarks() const {
  if(_landmarks.empty()){
    try{
      dlib::cv_image<dlib::bgr_pixel> cimg(_bgrFrame);
      dlib::full_object_detection
        shape = (*_shapePredFuncPtr)(cimg, dlib::rectangle(x_left_upper,               // left
                                                            y_left_upper,              // top
                                                            x_left_upper + width-1,    // right
                                                            y_left_upper + height-1)); // bottom

      for(size_t i=0; i<shape.num_parts(); i++){
        dlib::point pt = shape.part(i);
        _landmarks.push_back(cv::Point2f(pt.x(),pt.y()));
      }
    }catch(...){
      exception_ptr eptr = current_exception();
      LOG4CXX_FATAL(_log, "failed to determine landmarks");
      rethrow_exception(eptr);
    }
  }
  return _landmarks;
}

/** **************************************************************************
* Lazy accessor method to get/copy-create 92x92 thumbnail image for
* feature generation
*
* \returns thumbnail image of detection
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getThumbnail() const {
  if(_thumbnail.empty()){
    const int  THUMBNAIL_SIZE_X = 96;
    const int  THUMBNAIL_SIZE_Y = 96;
    const cv::Size THUMB_SIZE(THUMBNAIL_SIZE_X,THUMBNAIL_SIZE_Y);

    // Landmark indices for OpenFace nn4.v2, nn4.small1.v1, nn4.small2.v1
    const size_t lmIdx[] = {2,0,4};       // if using 5 pt landmarks
    const cv::Mat dst =  (cv::Mat_<float>(3,2)
                          << 0.1941570 * THUMBNAIL_SIZE_X, 0.16926692 * THUMBNAIL_SIZE_Y,
                              0.7888591 * THUMBNAIL_SIZE_X, 0.15817115 * THUMBNAIL_SIZE_Y,
                              0.4949509 * THUMBNAIL_SIZE_X, 0.51444140 * THUMBNAIL_SIZE_Y);

    cv::Mat src = cv::Mat_<float>(3,2);
    for(size_t r=0; r<3; r++){
      float* rowPtr = src.ptr<float>(r);
      rowPtr[0] = getLandmarks()[lmIdx[r]].x;
      rowPtr[1] = getLandmarks()[lmIdx[r]].y;
    }
    cv::Mat xfrm = cv::getAffineTransform(src,dst);
    try{
      _thumbnail = cv::Mat(THUMB_SIZE,_bgrFrame.type());
      cv::warpAffine(_bgrFrame,_thumbnail,cv::getAffineTransform(src,dst),
                    THUMB_SIZE,cv::INTER_CUBIC,cv::BORDER_REPLICATE);
    }catch (...) {
      exception_ptr eptr = current_exception();
      LOG4CXX_FATAL(_log, "failed to generate thumbnail for f" << frameIdx << *this);
      rethrow_exception(eptr);
    }

  }
  return _thumbnail;
}


/** **************************************************************************
* accessor method to get image associate with detection
*
* \returns image associated with detection
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getBGRFrame() const{
  #ifndef NDEBUG  // debug build
    if(! _bgrFrame.empty()){
      return _bgrFrame;
    }else{
      LOG4CXX_TRACE(_log,"BRG frame missing for detection: f" << this->frameIdx << *this);
      THROW_EXCEPTION("BRG frame is not allocated");
    }
  #else           // release build
    return _bgrFrame;
  #endif

}

/** **************************************************************************
* release reference to image frame
*************************************************************************** */
void DetectionLocation::releaseBGRFrame() {                                    LOG4CXX_TRACE(_log,"releasing bgrFrame for  f" << this->frameIdx << *this);
  _bgrFrame.release();
}

/** **************************************************************************
* get the location as an opencv rectange
*************************************************************************** */
const cv::Rect2i DetectionLocation::getRect() const {
  return cv::Rect2i(x_left_upper,y_left_upper,width,height);
}

/** **************************************************************************
* set the location from an opencv rectange
*************************************************************************** */
void DetectionLocation::setRect(const cv::Rect2i& rec){
  x_left_upper = rec.x;
  y_left_upper = rec.y;
        width  = rec.width;
        height = rec.height;
}

/** **************************************************************************
* copy feature vector from another detection
*
* \param   d detection from which to copy feature vector
*
*************************************************************************** */
void DetectionLocation::copyFeature(const DetectionLocation& d){
  _feature = d.getFeature();
}

/** **************************************************************************
* Lazy accessor method to get/compute feature vector based on thumbnail
*
* \returns unit magnitude feature vector
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getFeature() const {
  if(_feature.empty()){
    float aspect_ratio = width / height;
    if(   (   x_left_upper > 0
           && y_left_upper > 0
           && x_left_upper + width  < _bgrFrame.cols - 1
           && y_left_upper + height < _bgrFrame.rows - 1)
       || (   0.8 < aspect_ratio && aspect_ratio < 1.2)){
      const double inScaleFactor = 1.0 / 255.0;
      const cv::Size blobSize(96, 96);
      const cv::Scalar meanVal(0.0, 0.0, 0.0);  // BGR mean pixel color
      cv::Mat inputBlob = cv::dnn::blobFromImage(getThumbnail(), // BGR image
                                                inScaleFactor,   // no pixel value scaling (e.g. 1.0/255.0)
                                                blobSize,        // expected network input size: 90x90
                                                meanVal,         // mean BGR pixel value
                                                true,            // swap RB channels
                                                false);          // center crop
      _openFaceNet.setInput(inputBlob);
      _feature = _openFaceNet.forward().clone();             // need to clone as mem gets reused

    }else{                                                   // can't trust feature of detections on the edge
                                                             LOG4CXX_TRACE(_log,"'Zero-feature' for detection at frame edge with poor aspect ratio = " << aspect_ratio);
      _feature = cv::Mat::zeros(1, 128, CV_32F);             // zero feature will wipe out any dot products...
    }
  }
  return _feature;
}

/** ****************************************************************************
* Detect objects using SSD DNN opencv face detector network
*
* \param  cfg job configuration setting including image frame
*
* \returns found face detections that meet size requirements.
*
* \note each detection hang on to a reference to the bgrFrame which
*       should be released once no longer needed (i.e. features care computed)
*
*************************************************************************** */
DetectionLocationPtrVec DetectionLocation::createDetections(const JobConfig &cfg){
  DetectionLocationPtrVec detections;

  const double inScaleFactor = 1.0;
  const cv::Size blobSize(300, 300);
  const cv::Scalar meanVal(104.0, 117.0, 124.0);  // BGR mean pixel color

  cv::Mat inputBlob = cv::dnn::blobFromImage(cfg.bgrFrame, // BGR image
                                            inScaleFactor, // no pixel value scaling (e.g. 1.0/255.0)
                                            blobSize,      // expected network input size: 300x300
                                            meanVal,       // mean BGR pixel value
                                            true,          // swap RB channels
                                            false          // center crop
                                            );
  _ssdNet.setInput(inputBlob,"data");
  cv::Mat detection = _ssdNet.forward("detection_out");
  cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

  for(int i = 0; i < detectionMat.rows; i++){
    float conf = detectionMat.at<float>(i, 2);
    if(conf > cfg.confThresh){
      cv::Point2f ul(detectionMat.at<float>(i, 3),detectionMat.at<float>(i, 4));
      cv::Point2f lr(detectionMat.at<float>(i, 5),detectionMat.at<float>(i, 6));
      cv::Point2f wh = lr - ul;
      int x1     = static_cast<int>(ul.x * cfg.bgrFrame.cols);
      int y1     = static_cast<int>(ul.y * cfg.bgrFrame.rows);
      int width  = static_cast<int>(wh.x * cfg.bgrFrame.cols);
      int height = static_cast<int>(wh.y * cfg.bgrFrame.rows);

      if(    width  >= cfg.minDetectionSize
          && height >= cfg.minDetectionSize){
        detections.push_back(DetectionLocationPtr(
          new DetectionLocation(x1, y1, width, height, conf, (ul + lr) / 2.0,
                                cfg.frameIdx, cfg.frameTimeInSec,
                                cfg.bgrFrame)));                               LOG4CXX_TRACE(_log,"detection:" << *detections.back());
      }
    }
  }
  return detections;
}

/** **************************************************************************
*************************************************************************** */
void DetectionLocation::_setCudaBackend(const bool enabled){
  if(enabled){
    _ssdNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    _ssdNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    _openFaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    _openFaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);                LOG4CXX_TRACE(_log,"Enabled CUDA acceleration on device " << cv::cuda::getDevice());
  }else{
    _ssdNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    _ssdNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    _openFaceNet.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    _openFaceNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);                 LOG4CXX_INFO(_log,"Disabled CUDA acceleration");
  }
}

/** **************************************************************************
* try set CUDA to use specified GPU device
*
* \param cudaDeviceId device to use for hardware acceleration (-1 to disable)
*
* \returns true if successful false otherwise
*************************************************************************** */
bool DetectionLocation::trySetCudaDevice(const int cudaDeviceId){
  static int lastCudaDeviceId = -1;
  const string err_msg = "Failed to configure CUDA for deviceID=";
  try{
    if(lastCudaDeviceId != cudaDeviceId){
      if(lastCudaDeviceId >=0) cv::cuda::resetDevice();  // if we were using a cuda device prior clean up old contex / cuda resources
      if(cudaDeviceId >=0){
        cv::cuda::setDevice(cudaDeviceId);
        _setCudaBackend(true);
        lastCudaDeviceId = cudaDeviceId;
      }else{
        _setCudaBackend(false);
        lastCudaDeviceId = -1;
      }
    }
    return true;
  }catch(const runtime_error& re){                                           LOG4CXX_FATAL(_log, err_msg << cudaDeviceId << " Runtime error: " << re.what());
  }catch(const exception& ex){                                               LOG4CXX_FATAL(_log, err_msg << cudaDeviceId << " Exception: " << ex.what());
  }catch(...){                                                               LOG4CXX_FATAL(_log, err_msg << cudaDeviceId << " Unknown failure occurred. Possible memory corruption");
  }
  _setCudaBackend(false);
  lastCudaDeviceId = -1;
  return false;
}

/** **************************************************************************
* Setup class shared static configurations and initialize / load shared
* detectors and feature generator objects.  Set default CUDA acceleration
*
* \param log         logger object for logging
* \param plugin_path plugin directory so configs and models can be loaded
*
* \return true if everything was properly initialized, false otherwise
*************************************************************************** */
bool DetectionLocation::Init(log4cxx::LoggerPtr log, string plugin_path){

  _log = log;

  // Load SSD Tensor Flow Network
  string  tf_model_path = plugin_path + "/data/opencv_face_detector_uint8.pb";
  string tf_config_path = plugin_path + "/data/opencv_face_detector.pbtxt";

  string  sp_model_path = plugin_path + "/data/shape_predictor_5_face_landmarks.dat";
  string  tr_model_path = plugin_path + "/data/nn4.small2.v1.t7";

  const string err_msg = "Failed to load models: " + tf_config_path + ", " + tf_model_path;
  try{
      // load detector net
      _ssdNet = cv::dnn::readNetFromTensorflow(tf_model_path, tf_config_path);

      _shapePredFuncPtr = unique_ptr<dlib::shape_predictor>(new dlib::shape_predictor());
      dlib::deserialize(sp_model_path) >> *_shapePredFuncPtr;

      // load feature generator
      _openFaceNet = cv::dnn::readNetFromTorch(tr_model_path);


  }catch(const runtime_error& re){                                           LOG4CXX_FATAL(_log, err_msg << " Runtime error: " << re.what());
    return false;
  }catch(const exception& ex){                                               LOG4CXX_FATAL(_log, err_msg << " Exception: " << ex.what());
    return false;
  }catch(...){                                                               LOG4CXX_FATAL(_log, err_msg << " Unknown failure occurred. Possible memory corruption");
    return false;
  }

  return true;
}




/** **************************************************************************
* constructor
**************************************************************************** */
DetectionLocation::DetectionLocation(int x,int y,int width,int height,float conf,
                                     cv::Point2f center,
                                     size_t frameIdx, double frameTimeInSec,
                                     cv::Mat bgrFrame):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx),
        frameTimeInSec(frameTimeInSec){
        _bgrFrame = bgrFrame.clone();
}