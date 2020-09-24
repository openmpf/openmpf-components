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
#include <opencv2/cvconfig.h>

#include "util.h"
#include "DetectionLocation.h"
#include "Track.h"

using namespace MPF::COMPONENT;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr                DetectionLocation::_log;
cv::dnn::Net                      DetectionLocation::_ssdNet;                     ///< single shot DNN face detector network
cv::dnn::Net                      DetectionLocation::_openFaceNet;                ///< feature generator
unique_ptr<dlib::shape_predictor> DetectionLocation::_shapePredFuncPtr  = NULL;   ///< landmark detector function pointer
string                            DetectionLocation::_modelsPath;                 ///< where to find model files

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
* Compute 1 - Intersection Over Union metric between Kalman filter predicted
* location of track tail and a detection
*
* \param   tr track
* \returns 1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::kfIouDist(const Track &tr) const {
  return _iouDist(tr.kalmanPredictedBox());
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
*
*
* \returns landmarks
*
*************************************************************************** */
const cvPoint2fVec& DetectionLocation::getLandmarks() const {
  if(_landmarks.empty()){

    try{
      dlib::cv_image<dlib::bgr_pixel> cimg(_bgrFrameRot);
      cv::Rect2i bbox = rotate(getRect(),
                              _detectionOrientation,
                              cv::Size2i(_bgrFrameRot.cols-1,_bgrFrameRot.rows-1));
      dlib::full_object_detection
        shape = (*_shapePredFuncPtr)(cimg, dlib::rectangle( bbox.x,              // left
                                                            bbox.y,              // top
                                                            bbox.x + bbox.width-1,    // right
                                                            bbox.y + bbox.height-1)); // bottom

      cv::Size2f canvas(_bgrFrame.cols-1,_bgrFrame.rows-1);
      for(size_t i=0; i<shape.num_parts(); i++){
        dlib::point pt = shape.part(i);
        cv::Point2f lm = rotate(cv::Point2f(pt.x(),pt.y()), inv(_detectionOrientation), canvas);
        _landmarks.push_back(lm);
      }

      //calculate rotation angle
      cv::Point2f vec =  0.25 * (_landmarks[0] + _landmarks[1] + _landmarks[2] + _landmarks[3]) - _landmarks[4];
      int angle = static_cast<int>(degCCWfromVertical(vec) + 0.5);
      if(abs(angleDiff(angle, _angle)) < 45){ // check if it makes sense
        _angle = angle;
      }else{                                  // guess landmarks based on orientation
        cv::Point2f orig(bbox.x,bbox.y);
        _landmarks[2] = rotate(orig + cv::Point2f(bbox.width*0.1941570,bbox.height*0.16926692),inv(_detectionOrientation),canvas);
        _landmarks[0] = rotate(orig + cv::Point2f(bbox.width*0.7888591,bbox.height*0.15817115),inv(_detectionOrientation),canvas);
        _landmarks[4] = rotate(orig + cv::Point2f(bbox.width*0.4949509,bbox.height*0.51444140),inv(_detectionOrientation),canvas);
        _landmarks[1] = _landmarks[3] = (_landmarks[0] + _landmarks[2]) / 2.0;
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
    const cv::Mat dst =  (cv::Mat1f(3,2)
                          << 0.1941570 * THUMBNAIL_SIZE_X, 0.16926692 * THUMBNAIL_SIZE_Y,
                             0.7888591 * THUMBNAIL_SIZE_X, 0.15817115 * THUMBNAIL_SIZE_Y,
                             0.4949509 * THUMBNAIL_SIZE_X, 0.51444140 * THUMBNAIL_SIZE_Y);

    cv::Mat src = cv::Mat1f(3,2);
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
* accessor method to get face rotation angle
*
* \returns angle of face
*
*************************************************************************** */
const int DetectionLocation::getAngle() const{
  return _angle;
}

/** **************************************************************************
* release reference to image frame
*************************************************************************** */
void DetectionLocation::releaseBGRFrame() {                                    LOG4CXX_TRACE(_log,"releasing bgrFrames for  f" << this->frameIdx << *this);
  _bgrFrameRot.release();
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
* copy feature vector and orientations from another detection
*
* \param   d detection from which to copy feature vector
*
*************************************************************************** */
void DetectionLocation::copyFeature(const DetectionLocation& d){
  _feature = d.getFeature();
  _angle   = d._angle;
  _detectionOrientation = d._detectionOrientation;
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
       || (   0.5 < aspect_ratio && aspect_ratio < 2.0)){
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

  DetectionLocationPtrVec detections;                      // detection vector to be returned

  // some vars for NMS
  cvRectVec               bboxes;                          // bounding boxes found for all orientations
  cvPoint2fVec            centers;                         // centers of found bounding boxes for all orientations
  vector<float>           scores;                          // confidence scores of found detections for all orientations
  vector<OrientationType> orientations;                    // orientation for each found bounding box

  // setup related image variables
  map<OrientationType,cv::Mat> rotImages;                      // rotated image cache for making detections objects
  int maxSide = max(cfg.bgrFrame.cols,cfg.bgrFrame.rows);      // maximum image side length
  cv::Rect2i canvas(0,0,cfg.bgrFrame.cols,cfg.bgrFrame.rows);  // canvas size for rotations

  // create zero padded square version of bgrFrame
  cv::Mat square;
  cv::copyMakeBorder(cfg.bgrFrame, square,
                     0, maxSide - cfg.bgrFrame.rows,
                     0, maxSide - cfg.bgrFrame.cols,
                     cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
  int infSize = (cfg.inferenceSize > 0 ) ? cfg.inferenceSize : maxSide;  // size to use for inferencing

  for(auto& orientation:cfg.inferenceOrientations){

    cv::Mat image        = rotate(square, orientation);
    cv::Rect2i canvasRot = rotate(canvas, orientation,cv::Size2i(maxSide,maxSide));
    rotImages.insert(pair<OrientationType,cv::Mat>(orientation,image(canvasRot)));

    cv::Mat inputBlob = cv::dnn::blobFromImage(image,                           // BGR image
                                               1.0,                             // no pixel value scaling (e.g. 1.0/255.0)
                                               cv::Size(infSize,infSize),       // expected network input size: e.g.300x300
                                               cv::Scalar(104.0, 117.0, 124.0), // mean BGR pixel value
                                               true,                            // swap RB channels
                                               false);                          // center crop

    _ssdNet.setInput(inputBlob,"data");
    cv::Mat netOut = _ssdNet.forward("detection_out");
    cv::Mat_<float> res = cv::Mat(netOut.size[2], netOut.size[3], CV_32F, netOut.ptr<float>(0));
    for(int i = 0; i < res.rows; i++){
      float conf = res(i,2);
      if(conf > cfg.confThresh){
        cv::Rect2f rbox(res(i,3), res(i,4), res(i,5) - res(i,3),res(i,6) - res(i,4));   // normalized coords bounding box on rotated image
        cv::Rect2f    bbox = rotate(rbox, inv(orientation),cv::Size2f(1.0, 1.0));       // normalized coords  cords bounding box on brgFrame
        cv::Point2f center = 0.5 * (bbox.tl() + bbox.br());                             // normalized coords bounding box center
        cv::Point2f     wh = cv::Point2f(bbox.width,bbox.height) * cfg.bboxScaleFactor; // normalized scaled width and height

        int width  = static_cast<int>(wh.x * maxSide);                                  // width  in image pixels
        int height = static_cast<int>(wh.y * maxSide);                                  // height in image pixels

        if(width >= cfg.minDetectionSize && height >= cfg.minDetectionSize){
          cv::Point2f ul = center - 0.5 * wh;                             // normalized coords upper left coordinate
          int x1     = static_cast<int>(ul.x * maxSide);                  // upper left x in image pixels
          int y1     = static_cast<int>(ul.y * maxSide);                  // upper left y in image pixels

          cv::Rect2i iBbox = cv::Rect2i(x1,y1,width,height) & canvas;     // bounding box in pixels on image
          if(iBbox.width  > 0 && iBbox.height > 0){
            bboxes.push_back(iBbox);
            scores.push_back(conf);
            centers.push_back(center);
            orientations.push_back(orientation);
          }
        }
      }
    }
  }
  //#define DIAGNOSTIC_IMAGES
  #ifdef DIAGNOSTIC_IMAGES
  cv::Mat lm = cfg.bgrFrame.clone();
  #endif
  // Perform non maximum supression (NMS)
  vector<int> keepIdxs;
  cv::dnn::NMSBoxes(bboxes, scores, cfg.confThresh, cfg.nmsThresh, keepIdxs);
  for(int i:keepIdxs){
    detections.push_back(DetectionLocationPtr(
      new DetectionLocation(bboxes[i].x, bboxes[i].y, bboxes[i].width, bboxes[i].height,
                            scores[i], centers[i],
                            cfg.frameIdx, cfg.frameTimeInSec,
                            cfg.bgrFrame,
                            rotImages[orientations[i]],
                            orientations[i])));                               LOG4CXX_TRACE(_log,"detection:" << *detections.back());
    #ifdef DIAGNOSTIC_IMAGES
    stringstream fn; fn << "thumb_"<< cfg.frameIdx <<"_" << i << ".png";
    cv::imwrite(fn.str(),detections.back()->getThumbnail());
    detections.back()->drawLandmarks(lm,cv::Scalar(0,0,0));
    #endif
  }
  #ifdef DIAGNOSTIC_IMAGES
  stringstream fn; fn << "landmarks_"<< cfg.frameIdx << ".png";
  cv::imwrite(fn.str(),lm);
  #endif

  return detections;
}

/** **************************************************************************
*************************************************************************** */
void DetectionLocation::_loadNets(const bool enabled){

  // Load SSD Tensor Flow Network
  string  tf_model_path = _modelsPath + "opencv_face_detector_uint8.pb";
  string tf_config_path = _modelsPath + "opencv_face_detector.pbtxt";

  string  tr_model_path = _modelsPath + "nn4.small2.v1.t7";

  // load detector net
  _ssdNet = cv::dnn::readNetFromTensorflow(tf_model_path, tf_config_path);
  // load feature generator
  _openFaceNet = cv::dnn::readNetFromTorch(tr_model_path);

  #ifdef HAVE_CUDA
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
  #endif
}

/** **************************************************************************
* try set CUDA to use specified GPU device
*
* \param cudaDeviceId device to use for hardware acceleration (-1 to disable)
*
* \returns true if successful false otherwise
*************************************************************************** */
bool DetectionLocation::loadNetToCudaDevice(const int cudaDeviceId){
  static int lastCudaDeviceId = -1;
  const string err_msg = "Failed to configure CUDA for deviceID=";
  try{
    #ifdef HAVE_CUDA
    if(lastCudaDeviceId != cudaDeviceId){
      if(lastCudaDeviceId >=0){
        if(!_ssdNet.empty())           _ssdNet.~Net();         // need to release network this prior to device reset
        if(!_openFaceNet.empty()) _openFaceNet.~Net();
        cv::cuda::resetDevice();
      }
      if(cudaDeviceId >=0){
        cv::cuda::setDevice(cudaDeviceId);
        _loadNets(true);
        lastCudaDeviceId = cudaDeviceId;
      }else{
        _loadNets(false);
        lastCudaDeviceId = -1;
      }
    }
    return true;
    #endif
  }catch(const runtime_error& re){                                           LOG4CXX_FATAL(_log, err_msg << cudaDeviceId << " Runtime error: " << re.what());
  }catch(const exception& ex){                                               LOG4CXX_FATAL(_log, err_msg << cudaDeviceId << " Exception: " << ex.what());
  }catch(...){                                                               LOG4CXX_FATAL(_log, err_msg << cudaDeviceId << " Unknown failure occurred. Possible memory corruption");
  }
  _loadNets(false);
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

  int     cudaDeviceId     = getEnv<int> ({},"CUDA_DEVICE_ID",0);

  _log = log;
  _modelsPath = plugin_path + "/data/";

  const string err_msg = "Failed to load models from  " + _modelsPath;
  try{
      // load detector net
      loadNetToCudaDevice(cudaDeviceId);

      _shapePredFuncPtr = unique_ptr<dlib::shape_predictor>(new dlib::shape_predictor());
      dlib::deserialize(_modelsPath + "shape_predictor_5_face_landmarks.dat") >> *_shapePredFuncPtr;

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
*  Private constructor for a detection object
*
* \param x                     top left x coordinate of bounding box
* \param y                     top left y coordinate of bounding box
* \param width                 bounding box width
* \param height                bounding box height
* \param conf                  confidence score of detection
* \param center                normalized center coord of detection
* \param frameIdx              video frame sequence number
* \param frameTimeInSec        time stamp of video frame in sec
* \param bgrFrame              frame in which detection was found
* \param bgrFrameRot           bgrFrame rotated to detectionOrientation
* \param detectionOrientation  orientation of the frame when detected
*
**************************************************************************** */
DetectionLocation::DetectionLocation(int x,int y,int width,int height,float conf,
                                     cv::Point2f center,
                                     size_t frameIdx, double frameTimeInSec,
                                     cv::Mat bgrFrame,
                                     cv::Mat bgrFrameRot,
                                     OrientationType detectionOrientation):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx),
        frameTimeInSec(frameTimeInSec),
        _detectionOrientation(detectionOrientation),
        _angle(degCCWfromVertical(detectionOrientation)){
        _bgrFrame = bgrFrame; //.clone();
        _bgrFrameRot = bgrFrameRot;
}

/** **************************************************************************
*   Dump MPFLocation to a stream
*************************************************************************** */
ostream& MPF::COMPONENT::operator<< (ostream& out, const DetectionLocation& d) {
  out  << "[" << (MPFImageLocation)d
              << " F[" << d.getFeature().size() << "] T["
              << d.getThumbnail().rows << "," << d.getThumbnail().cols << "]";
  return out;
}