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

#include <opencv2/imgcodecs.hpp>

#include "DetectionLocation.h"

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
* Compute 1 - Intersection Over Union metric for two detections comprised of
* 1 - the ratio of the area of the intersection of the detection rectangles
* divided by the area of the union of the detection rectangles.
*
* \param   d second detection
* \returns 1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::iouDist(const DetectionLocation &d) const {
  int ulx = max(x_left_upper         , d.x_left_upper           );
  int uly = max(y_left_upper         , d.y_left_upper           );
  int lrx = min(x_left_upper + width , d.x_left_upper + d.width );
  int lry = min(y_left_upper + height, d.y_left_upper + d.height);

  float inter_area = max(0, lrx - ulx) * max(0, lry - uly);
  float union_area = width * height + d.width * d.height - inter_area;
  float dist = 1.0f - inter_area / union_area;                                 LOG4CXX_TRACE(_log,"iou dist = " << dist);
  return dist;
}

/** **************************************************************************
* Compute the temporal distance in frame counts between two detections
*
* \param   d second detection
* \returns   absolute difference in frame indices
*
*************************************************************************** */
float DetectionLocation::frameDist(const DetectionLocation &d) const {
  if(frameIdx > d.frameIdx){
    return frameIdx - d.frameIdx;
  }else{
    return d.frameIdx - frameIdx;
  }
}

/** **************************************************************************
* Compute euclidean center to distance center distance from normalized centers
*
* \param   d second detection
* \returns   normalized center to center distance [0 ... Sqrt(2)]
*
*************************************************************************** */
float  DetectionLocation::center2CenterDist(const DetectionLocation &d) const {
  float dx = center.x - d.center.x;
  float dy = center.y - d.center.y;
  float dist = sqrt( dx*dx + dy*dy );                                          LOG4CXX_TRACE(_log,"center-2-center dist = " << dist);
  return dist;
}


/** **************************************************************************
* Compute feature distance (similarity) between two detection feature vectors
*
* \param   d second detection
* \returns cos distance [0 ... 1.0]
*
* \note Feature vectors are expected to be of unit magnitude
*
*************************************************************************** */
float DetectionLocation::featureDist(const DetectionLocation &d) const {
  float dist = 1.0f - max(0.0f,static_cast<float>(getFeature().dot(d.getFeature())));  LOG4CXX_TRACE(_log,"feature dist = " << dist);
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
      LOG4CXX_FATAL(_log, "failed to generate thumbnail");
      rethrow_exception(eptr);
    }

  }
  return _thumbnail;
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
        detections.push_back(unique_ptr<DetectionLocation>(
          new DetectionLocation(x1, y1, width, height, conf, (ul + lr) / 2.0,
                                cfg.frameIdx, cfg.bgrFrame)));                 LOG4CXX_TRACE(_log,"detection:" << *detections.back());
      }
    }
  }
  return detections;
}

/** **************************************************************************
* Setup class shared static configurations and initialize / load shared
* detectors and feature generator objects.
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

  string err_msg = "Failed to load models: " + tf_config_path + ", " + tf_model_path;
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
*  Get a new DetectionLocation from an existing one based on a frame
*
* \param
* \returns ptr to new location based on tracker's estimation
*
* \note    tracker is passed on to the new location on success
*
**************************************************************************** */
unique_ptr<DetectionLocation> DetectionLocation::ocvTrackerPredict(const JobConfig &cfg){

  if(_trackerPtr.empty()){   // initialize a new tracker if we don't have one already
    _trackerPtr = cv::TrackerMOSSE::create();  // could try different trackers here. e.g. cv::TrackerKCF::create();
    _trackerPtr->init(_bgrFrame,cv::Rect2d(x_left_upper,y_left_upper,width,height));  LOG4CXX_TRACE(_log,"tracker created for " << (MPFImageLocation)*this);
    _trackerStartFrameIdx = cfg.frameIdx;
  }

  cv::Rect2d p;
  unique_ptr<DetectionLocation> detPtr;
  if(cfg.frameIdx - _trackerStartFrameIdx <= cfg.maxFrameGap){
    if(_trackerPtr->update(cfg.bgrFrame,p)){
      detPtr = unique_ptr<DetectionLocation>(new DetectionLocation(
        p.x,p.y,p.width,p.height,0.0,
        cv::Point2f((p.x + p.width/2.0f )/static_cast<float>(cfg.bgrFrame.cols),
                    (p.y + p.height/2.0f)/static_cast<float>(cfg.bgrFrame.rows)),
        cfg.frameIdx,
        cfg.bgrFrame));                                                                  LOG4CXX_TRACE(_log,"tracking " << (MPFImageLocation)*this << "to  " << (MPFImageLocation)*detPtr);

      detPtr->_trackerPtr = _trackerPtr;
      detPtr->_trackerStartFrameIdx = _trackerStartFrameIdx;
      _trackerPtr.release();
      detPtr->_feature = getFeature();  // clone feature of prior detection
    }else{
                                                                                        LOG4CXX_TRACE(_log,"could not track " << (MPFImageLocation)*this << " to new location");
    }
  }else{
                                                                                        LOG4CXX_TRACE(_log,"extrapolation tracking stopped" << (MPFImageLocation)*this << " frame gap = " << cfg.frameIdx - _trackerStartFrameIdx << " > " <<  cfg.maxFrameGap);
  }
  return detPtr;
}

/** **************************************************************************
*   Private constructor
**************************************************************************** */
DetectionLocation::DetectionLocation(int x,int y,int width,int height,float conf,
                                     cv::Point2f center, size_t frameIdx,
                                     cv::Mat bgrFrame):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx){
        _bgrFrame = bgrFrame.clone();
}