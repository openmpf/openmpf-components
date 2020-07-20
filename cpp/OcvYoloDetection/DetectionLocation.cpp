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
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>

#include "DetectionLocation.h"
#include "Track.h"
#include "ocv_phasecorr.h"

using namespace MPF::COMPONENT;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr                DetectionLocation::_log;
cv::dnn::Net                      DetectionLocation::_yolo;
stringVec                         DetectionLocation::_classes;
stringVec                         DetectionLocation::_yoloOutputNames;

/** ****************************************************************************
*  Draw polylines to visualize landmark scoresVectors
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
* Compute offset required for pixel alignment between track's tail detection
* and location
*
* \param   tr track
* \returns point giving the offset for pixel alignment
*
*************************************************************************** */
cv::Point2d  DetectionLocation::_phaseCorrelate(const Track &tr) const {
  cv::Mat_<float> P, Pm, C;
  cv::mulSpectrums(getFeature(), tr.back()->getFeature(), P, 0, true);
  cv::magSpectrums(P, Pm);
  cv::divSpectrums(P, Pm, C, 0, false);
  cv::idft(C,C);
  cv::fftShift(C);
  cv::Point peakLoc;
  cv::minMaxLoc(C, NULL, NULL, NULL, &peakLoc);
  double response;
  return cv::Point2d(_dftSize/2.0, _dftSize/2.0) -
         cv::weightedCentroid(C, peakLoc, cv::Size(5, 5), &response);
}


/** **************************************************************************
* Compute feature distance (similarity) to track's tail detection's
* feature vectors
*
* \param   tr track
* \returns distance [0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::featureDist(const Track &tr) const {
  //float dist = 1.0f - max(0.0f,static_cast<float>(getFeature().dot(tr.back()->getFeature())));  LOG4CXX_TRACE(_log,"feature dist = " << dist);

  // get overlap roi
  cv::Rect2d olRoi =  cv::Rect2d(-_phaseCorrelate(tr), tr.back()->_bgrFrame.size())
                    & cv::Rect2d(    cv::Point2d(0,0),            _bgrFrame.size());

  cv::Point2d ctr = olRoi.tl() + 0.5 * cv::Point2d(olRoi.size());

  cv::Mat comp;
  cv::getRectSubPix(_bgrFrame,olRoi.size(), ctr, comp);
cv::imwrite("comp.png",comp);
  cv::absdiff(tr.back()->_bgrFrame(cv::Rect2i(cv::Point2i(0,0),comp.size())),
              comp,comp);
cv::imwrite("diff.png",comp);
  cv::Scalar mean = cv::mean(comp);
  float dist = mean.ddot(mean);
is 128 top lef ther best place to grab features?
  return dist;
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
* Lazy accessor method to get feature vector
*
* \returns unit magnitude feature vector
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getFeature() const {
  if(_feature.empty()){

    // make normalized grayscale float image
    cv::Mat gray;
    cv::cvtColor(_bgrFrame, gray, CV_BGR2GRAY);                                LOG4CXX_TRACE(_log,"Converted to gray scale " << cv::typeToString(gray.type()) << " (" << gray.size << ")");
    cv::Scalar mean,std;
    cv::meanStdDev(gray, mean, std);
    std[0] = max(std[0],1/255.0);
    gray.convertTo(gray,CV_32FC1,1.0/std[0],-mean[0]/std[0]);                  LOG4CXX_TRACE(_log,"Converted to zero mean unit std float " << cv::typeToString(gray.type()) <<"(" << gray.size << ")");

    // zero padd (or clip) grayscale image into dft buffer
    cv::Rect roi(0,0,min(_dftSize,_bgrFrame.cols),min(_dftSize,_bgrFrame.rows));
    cv::Mat_<float> feature=cv::Mat_<float>::zeros(_dftSize,_dftSize);
    gray(roi).copyTo(feature(roi));                                            LOG4CXX_TRACE(_log,"Zero padded/clipped to " << cv::typeToString(feature.type()) << "(" << feature.size << ")");

    // take dft (CCS packed)
    cv::dft(feature,feature,cv::DFT_REAL_OUTPUT);                              LOG4CXX_TRACE(_log,"Created CCS packed dft " << cv::typeToString(feature.type()) << "(" << feature.size << ")");
    _feature = feature.clone();
  }                                                                            LOG4CXX_TRACE(_log,"returning feature " << cv::typeToString(_feature.type()) << "(" << _feature.size << ")");
  assert(_feature.type() == CV_32FC1);
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
*       should be released once no longer needed (i.e. scoresVectors care computed)
*
*************************************************************************** */
DetectionLocationPtrVec DetectionLocation::createDetections(const JobConfig &cfg){
  DetectionLocationPtrVec detections;

  float szScaleFactor = static_cast<float>(cfg.inputImageSize) / static_cast<float>(max(cfg.bgrFrame.cols,cfg.bgrFrame.rows));
  const cv::Size blobSize(static_cast<int>(cfg.bgrFrame.cols * szScaleFactor + 0.5),
                          static_cast<int>(cfg.bgrFrame.rows * szScaleFactor + 0.5));

  // determine padding to make image square of size cfg.inputImageSize
  const int padWidth  = cfg.inputImageSize - blobSize.width;                   // horizonatal padding needed
  const int padHeight = cfg.inputImageSize - blobSize.height;                  // vertical padding needed
  cv::Point2f pad(padWidth  / 2.0, padHeight / 2.0);                           // padding x-y coordinate offsets
  const int padLeft   = static_cast<int>(pad.x + 0.5f);                        // left padding (1/2 of horizontal padding)
  const int padTop    = static_cast<int>(pad.y + 0.5f);                        // top  padding (1/2 of vertical padding)

  // scale and padd image for inferencing
  cv::Mat img,blobImg;
  cv::resize(cfg.bgrFrame,img,blobSize);                                       // resize so largest dim matches cfg.inputImageSize
  cv::copyMakeBorder(img, blobImg,
                     padTop , padHeight - padTop,
                     padLeft, padWidth  - padLeft,
                     cv::BORDER_CONSTANT,cv::Scalar(0,0,0));                   // add black bars to keep image square (yolo3 should do this but ocv fails)

  cv::Mat inputBlob = cv::dnn::blobFromImage(blobImg,                          // square BGR image
                                            1/255.0,                           // no pixel value scaling (e.g. 1.0/255.0)
                                            cv::Size(cfg.inputImageSize,cfg.inputImageSize),
                                            cv::Scalar(0,0,0),                 // mean BGR pixel value
                                            true,                              // swap RB channels
                                            false);                            // center crop
  _yolo.setInput(inputBlob,"data");
  cvMatVec outs;
  _yolo.forward(outs, _yoloOutputNames);                                       // inference

  pad = pad / static_cast<float>(cfg.inputImageSize);                          // convert x-y offset to fractional
  float revScaleFactor = cfg.inputImageSize / szScaleFactor;                   // calc reverse scaling factor

  // grab outputs for NMS
  cvRectVec  bboxes;
  cvPointVec centers;
  floatVec   topConfidences;
  cvMatVec   scoresVectors;
  for(auto &out : outs){
    float* data = (float*)out.data;
    for(int j = 0; j < out.rows; ++j, data += out.cols){
      cv::Mat scores = out.row(j).colRange(5,out.cols);
      cv::Point idx;
      double conf;
      cv::minMaxLoc(scores,nullptr,&conf,nullptr,&idx);
      if(conf >= cfg.confThresh){
        cv::Point2f center(data[0],data[1]);
        center = center - pad;                                                 // yolo zeropads top,bottom,left,right to get square image
        centers.push_back(center);
        bboxes.push_back(
          cv::Rect2d((center.x - data[2] / 2.0) * revScaleFactor,
                     (center.y - data[3] / 2.0) * revScaleFactor,
                                 data[2]        * revScaleFactor,
                                 data[3]        * revScaleFactor));
        topConfidences.push_back(conf);
        scoresVectors.push_back(scores);
      }
    }
  }

  // Perform non maximum supression (NMS)
  intVec keepIdxs;
  cv::dnn::NMSBoxes(bboxes,topConfidences,cfg.confThresh,cfg.nmsThresh,keepIdxs);

  // Create detection objects
  for(int& keepIdx : keepIdxs){
    cv::Rect2d bbox = bboxes[keepIdx];
    detections.push_back(DetectionLocationPtr(
      new DetectionLocation(static_cast<int>(bboxes[keepIdx].x + 0.5),
                            static_cast<int>(bboxes[keepIdx].y + 0.5),
                            static_cast<int>(bboxes[keepIdx].width + 0.5),
                            static_cast<int>(bboxes[keepIdx].height + 0.5),
                            topConfidences[keepIdx],
                            centers[keepIdx],
                            cfg.frameIdx, cfg.frameTimeInSec,
                            cfg.bgrFrame,
                            cfg.dftSize)));

    // add top N classifications
    vector<int> sort_idx(scoresVectors[keepIdx].cols);
    iota(sort_idx.begin(), sort_idx.end(), 0);  //initialize sort_index
    stable_sort(sort_idx.begin(), sort_idx.end(),
      [&scoresVectors,&keepIdx](int i1, int i2) {
        return scoresVectors[keepIdx].at<float>(0,i1) > scoresVectors[keepIdx].at<float>(0,i2); });
    stringstream class_list;
    stringstream score_list;
    class_list << _classes.at(sort_idx[0]);
    score_list << scoresVectors[keepIdx].at<float>(0,sort_idx[0]);
    for(int i=1; i < sort_idx.size(); i++){
      if(scoresVectors[keepIdx].at<float>(0,sort_idx[i]) < numeric_limits<float>::epsilon()) break;
      class_list << "; " << _classes.at(sort_idx[i]);
      score_list << "; " << scoresVectors[keepIdx].at<float>(0,sort_idx[i]);
      if(i >= cfg.numClassPerRegion) break;
    }
    detections.back()->detection_properties.insert({
      {"CLASSIFICATION", _classes[sort_idx[0]]},
      {"CLASSIFICATION LIST", class_list.str()},
      {"CLASSIFICATION CONFIDENCE LIST", score_list.str()}});         LOG4CXX_TRACE(_log,"Detection " << (MPFImageLocation)(*detections.back()));
  }

  return detections;
}

/** **************************************************************************
*************************************************************************** */
void DetectionLocation::_setCudaBackend(const bool enabled){
  #ifdef HAVE_CUDA
  if(enabled){
    _yolo.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    _yolo.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
  }else{
    _yolo.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    _yolo.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
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
bool DetectionLocation::trySetCudaDevice(const int cudaDeviceId){
  static int lastCudaDeviceId = -1;
  const string err_msg = "Failed to configure CUDA for deviceID=";
  try{
    #ifdef HAVE_CUDA
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
    #endif
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

  // Load Yolo Network
  string  model_path   = plugin_path + "/data/" + getEnv<string>({},"MODEL_WEIGHTS_FILE", "yolov3.weights");
  string  config_path  = plugin_path + "/data/" + getEnv<string>({},"MODEL_CONFIG_FILE" , "yolov3.cfg");
  string  classes_path = plugin_path + "/data/" + getEnv<string>({},"MODEL_CLASS_FILE"  , "coco.names");

  const string err_msg = "Failed to load model: " + config_path + ", " + model_path;
  try{
      // load output class names
      ifstream classfile(classes_path);
      string str;
      _classes.clear();
      while(getline(classfile, str)){
        _classes.push_back(str);
      }

      // load detector net
      _yolo = cv::dnn::readNetFromDarknet(config_path, model_path);

      // get names of output layers
      _yoloOutputNames = _yolo.getUnconnectedOutLayersNames();


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
                                     cv::Mat bgrFrame, int dftSize):
        MPFImageLocation(x,y,width,height,conf),
        center(center),
        frameIdx(frameIdx),
        frameTimeInSec(frameTimeInSec),
        _dftSize(dftSize)
        {
        _bgrFrame = bgrFrame(cv::Rect(x,y,width,height)).clone();
}