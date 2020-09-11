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
#include <opencv2/cvconfig.h>

#include "util.h"
#include "DetectionLocation.h"
#include "Track.h"
#include "ocv_phasecorr.h"

using namespace MPF::COMPONENT;

cv::dnn::Net DetectionLocation::_net;                  ///< DNN detector network
stringVec    DetectionLocation::_netClasses;           ///< list of classes for DNN
stringVec    DetectionLocation::_netOutputNames;       ///< list of DNN output names
cv::Mat1f    DetectionLocation::_netConfusion;         ///< DNN detector classification confusion matrix
intVec       DetectionLocation::_classGroupIdx;        ///< class index to class group mapping from confusion matrix

/** **************************************************************************
* Compute (1 - Intersection Over Union) metric between a rectangel and detection
* comprised of 1 - the ratio of the area of the intersection of the detection
* rectangles divided by the area of the union of the detection rectangles.
*
* \param   tr track to compute iou with
* \returns 1- intersection over union [0.0 ... 1.0]
*
*************************************************************************** */
float DetectionLocation::iouDist(const Track &tr) const {

  cv::Rect2i rect = tr.predictedBox();

  int ulx = max(x_left_upper         , rect.x           );
  int uly = max(y_left_upper         , rect.y           );
  int lrx = min(x_left_upper + width , rect.x + rect.width );
  int lry = min(y_left_upper + height, rect.y + rect.height);

  float inter_area = max(0, lrx - ulx) * max(0, lry - uly);
  float union_area = width * height + rect.width * rect.height - inter_area;
  float dist = 1.0f - inter_area / union_area;                                 LOG_TRACE("iou dist = " << dist << " between " << *this << " and " << tr);
  return dist;
}

/** **************************************************************************
* Compute the temporal distance (frame count) between track tail and detection
*
* \param   tr track
* \returns   absolute difference in frame indices
*
*************************************************************************** */
float DetectionLocation::frameDist(const Track &tr) const {
  int trIdx = tr.locations.back().frame.idx;
  return (frame.idx > trIdx) ? frame.idx - trIdx : trIdx - frame.idx;  // unsigned long diff...
}

/** **************************************************************************
* Compute euclidean center to distance center distance from normalized centers
*
* \param   tr track
* \returns   normalized center to center distance [0 ... Sqrt(2)]
*
*************************************************************************** */
float  DetectionLocation::center2CenterDist(const Track &tr) const {
  cv::Rect2i r1 = getRect();
  cv::Rect2i r2 = tr.locations.back().getRect();
  cv::Point2f d((r1.tl() - r2.tl() + r1.br() - r2.br())/2.0f);
  d.x /= static_cast<float>(frame.bgr.cols);
  d.y /= static_cast<float>(frame.bgr.rows);
  float dist = sqrt( d.x*d.x + d.y*d.y );                                          LOG_TRACE("center-2-center dist: " << dist << " between " << *this << " and " << tr);
  return dist;
}

/** **************************************************************************
* Class distance
*
*************************************************************************** */
float DetectionLocation::classDist(const Track& tr) const {
  float dist = 1.0f - max( 0.0f, min( 1.0f, static_cast<float>(_classFeature.dot(tr.locations.back().getClassFeature()))));  LOG_TRACE("class dist = " << dist << " between " << *this << " and " << tr);
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
  cv::Mat1f P, Pm, C;
  cv::mulSpectrums(getDFTFeature(), tr.locations.back().getDFTFeature(), P, 0, true);
  cv::magSpectrums(P, Pm);
  cv::divSpectrums(P, Pm, C, 0, false);
  cv::idft(C,C);
  cv::fftShift(C);
  cv::Point peakLoc;
  cv::minMaxLoc(C, NULL, NULL, NULL, &peakLoc);
  double response;
  return cv::Point2d(_cfg.dftSize/2.0, _cfg.dftSize/2.0) -
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
  float dist;
  cv::Rect2d  trRoi(tr.locations.back().getRect());                                   // roi of track object in tr's frame
  if(trRoi.area() > 0.0){

    cv::Point2d ctr = cv::Point2d(x_left_upper + 0.5 * width,
                                  y_left_upper + 0.5 * height)
                      - _phaseCorrelate(tr);                                   // center of shifted track roi

    if(cv::Rect2d(cv::Point2d(0,0),frame.bgr.size()).contains(ctr)){        // center ok
      cv::Mat comp;
      cv::getRectSubPix(frame.bgr,trRoi.size(), ctr, comp);                // grab corresponding region from bgrFrame with border replication
      cv::absdiff(tr.locations.back().frame.bgr(trRoi), comp, comp);                 // compute pixel wise absolute diff (could do HSV transform 1st?!)
      assert(frame.bgr.depth() == CV_8U);
      cv::Scalar mean = cv::mean(comp) / 255.0;                                // get mean normalized BGR pixel difference
      dist = mean.ddot(mean) / frame.bgr.channels();                       // combine BGR: d = BGR.BGR
    }else{                                                                     LOG_TRACE("ctr " << ctr << " outside bgrFrame");
      dist = 1.0;
    }
  }else{                                                                       LOG_TRACE("tr tail " << trRoi << " has zero area");
    dist = 1.0;
  }                                                                            LOG_TRACE("feature dist: " << dist << " between " << *this << " and " << tr);
  return dist;
}


/** **************************************************************************
* Compute kalman distance (error from predicted) to track's tail detection's
*
* \param   tr track
* \returns distance
*
*************************************************************************** */
float DetectionLocation::kfResidualDist(const Track &tr) const {
  float dist = tr.testResidual(getRect(), _cfg.edgeSnapDist);                      LOG_TRACE("residual dist: " << dist << " between " << *this << " and " << tr);
  return dist;
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
  cv::Rect2i snaped = snapToEdges(rec, rec, frame.bgr.size(), _cfg.edgeSnapDist);
  x_left_upper = snaped.x;
  y_left_upper = snaped.y;
        width  = snaped.width;
        height = snaped.height;
}

/** **************************************************************************
* copy feature vector from another detection
*
* \param   d detection from which to copy feature vector
*
*************************************************************************** */
//void DetectionLocation::copyFeature(const DetectionLocation& d) {
//  _dftFeature   = d.getDFTFeature();
//}

/** **************************************************************************
* get hanning window of specified size
*
* \param size  size of hanning windowing to return
*
*************************************************************************** */
const cv::Mat1f   DetectionLocation::_getHanningWindow(const cv::Size &size) const {
  if(   size.width  == _cfg.dftSize
     && size.height == _cfg.dftSize){

    static cv::Mat1f defWin;          // most common size so cache it.
    if(   defWin.empty()
       || defWin.rows != _cfg.dftSize){
      cv::createHanningWindow(defWin,cv::Size(_cfg.dftSize,_cfg.dftSize),CV_32F);  //LOG_TRACE("Created default hanning window of size " << defWin.size());
    }
    return defWin;

  }else{
    cv::Mat1f customWin;
    cv::createHanningWindow(customWin,size,CV_32F);                                        //LOG_TRACE("Created custom hanning window of size " << customWin.size());
    return customWin;
  }
}
/** **************************************************************************
* get class feature vector consisting of a unit vector of scores
*
* \returns unit magnitude class score feature vector
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getClassFeature() const {
  return _classFeature;
}

/** **************************************************************************
* Lazy accessor method to get feature vector
*
* \returns unit magnitude feature vector
*
*************************************************************************** */
const cv::Mat& DetectionLocation::getDFTFeature() const {
  if(_dftFeature.empty()){

    // make normalized grayscale float image
    cv::Mat gray;
    cv::cvtColor(frame.bgr(getRect()), gray, CV_BGR2GRAY);                     // Convert to gray scale
    cv::Scalar mean,std;
    cv::meanStdDev(gray, mean, std);
    std[0] = max(std[0],1/255.0);
    gray.convertTo(gray,CV_32FC1,1.0/std[0],-mean[0]/std[0]);                  // Convert to zero mean unit std

    // zero pad (or clip) center of grayscale image into center of dft buffer
    cv::Mat1f feature=cv::Mat1f::zeros(_cfg.dftSize,_cfg.dftSize);
    cv::Rect2i grRoi((gray.size() - feature.size())/2, feature.size());
    grRoi = grRoi & cv::Rect2i(cv::Point2i(0,0),gray.size());                  // Image  source region
    cv::Rect2i ftRoi((feature.size() - grRoi.size())/2,grRoi.size());          // Buffer target region
    if(_cfg.dftHannWindowEnabled){                                             // Apply hanning window
      cv::Mat1f hann = _getHanningWindow(grRoi.size());
      cv::multiply(gray(grRoi),hann,gray(grRoi));
    }
    gray(grRoi).copyTo(feature(ftRoi));                                        // add region from image to buffer

    // take dft of buffer (CCS packed)
    cv::dft(feature,feature,cv::DFT_REAL_OUTPUT);                              // created CCS packed dft
    *(const_cast<cv::Mat*>(&_dftFeature)) = feature.clone();                   // lazy initialize const member
  }
  assert(_dftFeature.type() == CV_32FC1);
  return _dftFeature;
}

/** ****************************************************************************
* Detect objects using SSD DNN opencv face detector network
*
* \param  cfg       job configuration setting
* \param  framePtrs frames to process in one inference call
*
* \returns found face detections that meet size requirements.
*
* \note each detection hang on to a reference to the bgrFrame which
*       should be released once no longer needed (i.e. scoresVectors care computed)
*
*************************************************************************** */
DetectionLocationListVec DetectionLocation::createDetections(const Config& cfg, const FrameVec &frames){
                                                                               LOG_TRACE("Creating detections in frame batch of size " << frames.size());
  int inputImageSize = cfg.inputImageSize;
  if(inputImageSize <= 0){                                                     //LOG_TRACE("Determining max image dimension in batch");
    for(auto& f:frames){
      int maxImageDim = max(f.bgr.cols,f.bgr.rows);
      if(maxImageDim > inputImageSize){
        inputImageSize = maxImageDim;
      }
    }                                                                          //LOG_TRACE("Using max batch dimension = " <<  inputImageSize <<" as input size");
  }

  floatVec         sizeScaleFactors;                                           // scale factor for each image
  cvPoint2fVec     padOffsets;                                                 // coordinate offsets due to padding to make image square
  cvMatVec         blobImgs;                                                   // padded, square image of inputSize
  for(auto& f : frames){
    float maxImageDim     = fmax(f.bgr.cols,f.bgr.rows);
    float sizeScaleFactor = inputImageSize / maxImageDim;
    sizeScaleFactors.push_back(sizeScaleFactor);                               LOG_TRACE("Required scale factor " << sizeScaleFactor);

    cv::Mat scaledImg;
    cv::resize(f.bgr, scaledImg,
               cv::Size2i(f.bgr.cols * sizeScaleFactor,
                          f.bgr.rows * sizeScaleFactor));                      LOG_TRACE("Resized image to " << scaledImg.size());

    const int padLeft = (inputImageSize - scaledImg.cols)  / 2;
    const int padTop  = (inputImageSize - scaledImg.rows)  / 2;
    padOffsets.push_back(cv::Point2f(padLeft, padTop)/inputImageSize);         LOG_TRACE("Fractional padding coordinate offset " << padOffsets.back());

    cv::Mat paddedScaledImg;
    cv::copyMakeBorder(scaledImg, paddedScaledImg,                             // add black bars to keep image square (yolo3 should do this but ocv fails)
                       padTop , inputImageSize - padTop  - scaledImg.rows,
                       padLeft, inputImageSize - padLeft - scaledImg.cols,
                       cv::BORDER_REPLICATE,cv::Scalar(0,0,0));                // replicated border seems to not create as many large false positives
    blobImgs.push_back(paddedScaledImg);                                       LOG_TRACE("Inference image size " << paddedScaledImg.size());
  }

  cv::Mat inputBlob = cv::dnn::blobFromImages(blobImgs,                        // square BGR image
                                              1/255.0,                         // no pixel value scaling (e.g. 1.0/255.0)
                                              cv::Size(inputImageSize,
                                                       inputImageSize),
                                              cv::Scalar(0,0,0),               // mean BGR pixel value
                                              true,                            // swap RB channels
                                              false);                          // center crop

  _net.setInput(inputBlob,"data");                                             // different output layers for different scales, e.g. yolo_82,yolo_94,yolo_106 for yolo_v3
  cvMatVec outs;                                                               // outs[output-layer][frame][detection][feature] =  0:cent_x, 1:cent_y, 2:width, 3:height, 4:objectness, 5..84:class-scores
  _net.forward(outs, _netOutputNames);                                         // https://answers.opencv.org/question/212588/how-to-process-output-of-detection-network-when-batch-of-images-is-used-as-network-input/
                                                                               // https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b

  DetectionLocationListVec detections;
  detections.reserve(frames.size());
  for(int fr=0; fr<frames.size(); fr++){                                       LOG_TRACE("Processing inference results for batch frame: " << fr);
    float revSizeScaleFactor = inputImageSize / sizeScaleFactors[fr];
    cv::Point2f padOffset = padOffsets[fr];

    // grab outputs for NMS
    cvRect2dVec  bboxes;
    floatVec   topConfidences;
    cvMatVec   scoresVectors;
    for(int  lr=0; lr<outs.size(); lr++){
      cv::Mat1f data = cv::Mat(outs[lr].size[outs[lr].dims-2],
                               outs[lr].size[outs[lr].dims-1],
                               CV_32F,outs[lr].ptr<float>(fr));
      assert(data.cols == 85);                                                 // yolo output feature are 85 floats
      for(int rw=0; rw<data.rows; rw++){
        cv::Mat scores = data.row(rw).colRange(5,data.cols);
        cv::Point idx;
        double conf;
        cv::minMaxLoc(scores,nullptr,&conf,nullptr,&idx);
        if(conf >= cfg.confThresh){
          float* rowPtr = data.ptr<float>(rw);
          cv::Point2f center(rowPtr[0],rowPtr[1]);
          center = center - padOffset;                                         // yolo zeropads top,bottom,left,right to get square image
          bboxes.push_back(
            cv::Rect2d((center.x - rowPtr[2] / 2.0) * revSizeScaleFactor,
                       (center.y - rowPtr[3] / 2.0) * revSizeScaleFactor,
                                   rowPtr[2]        * revSizeScaleFactor,
                                   rowPtr[3]        * revSizeScaleFactor));
          topConfidences.push_back(conf);
          scoresVectors.push_back(scores);
        }
      } // feature
    }  // layer

    // Perform non maximum supression (NMS)
    intVec keepIdxs;                                                           LOG_TRACE("Performing non max supression of " <<bboxes.size() << " detections");
    cv::dnn::NMSBoxes(bboxes,topConfidences,
                      cfg.confThresh,
                      cfg.nmsThresh,
                      keepIdxs);                                               LOG_TRACE("Kept " << keepIdxs.size() << " detections");

    // Create detection objects for frame
    detections.push_back(DetectionLocationList());
    for(int& keepIdx : keepIdxs){
      cv::Mat classFeature;
      cv::normalize(scoresVectors[keepIdx] * _netConfusion, classFeature,1.0,0.0,cv::NORM_L2);
      detections.back().push_back(
        DetectionLocation(cfg,
                          frames[fr],
                          bboxes[keepIdx],
                          topConfidences[keepIdx],
                          classFeature,cv::Mat()));
      // add top N classifications
      vector<int> sort_idx(scoresVectors[keepIdx].cols);
      iota(sort_idx.begin(), sort_idx.end(), 0);  //initialize sort_index
      stable_sort(sort_idx.begin(), sort_idx.end(),
        [&scoresVectors,&keepIdx](int i1, int i2) {
          return scoresVectors[keepIdx].at<float>(0,i1) > scoresVectors[keepIdx].at<float>(0,i2);
        });
      stringstream classList;
      stringstream scoreList;
      classList << _netClasses.at(sort_idx[0]);
      scoreList << scoresVectors[keepIdx].at<float>(0,sort_idx[0]);
      for(int i=1; i < sort_idx.size(); i++){
        if(scoresVectors[keepIdx].at<float>(0,sort_idx[i]) < numeric_limits<float>::epsilon()) break;
        classList << "; " << _netClasses.at(sort_idx[i]);
        scoreList << "; " << scoresVectors[keepIdx].at<float>(0,sort_idx[i]);
        if(i >= cfg.numClassPerRegion) break;
      }
      detections.back().back().detection_properties.insert({
        {"CLASSIFICATION", _netClasses[sort_idx[0]]},
        {"CLASSIFICATION LIST", classList.str()},
        {"CLASSIFICATION CONFIDENCE LIST", scoreList.str()}});         LOG_TRACE("Detection " << (MPFImageLocation)detections.back().back());


    }  // keep indicies
  } // frames

  return detections;
}

/** **************************************************************************
*************************************************************************** */
void DetectionLocation::_loadNet(const bool useCUDA){

  string  modelPath          = Config::pluginPath + "/data/" + getEnv<string>({},"MODEL_WEIGHTS_FILE",   "yolov3.weights");
  string  modelConfigPath    = Config::pluginPath + "/data/" + getEnv<string>({},"MODEL_CONFIG_FILE" ,   "yolov3.cfg");

  //_netPtr = NetPtr(new (cv::dnn::Net(move(cv::dnn::readNetFromDarknet(modelConfigPath, modelPath))));
  _net = cv::dnn::readNetFromDarknet(modelConfigPath, modelPath);

  #ifdef HAVE_CUDA
  if(useCUDA){
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);                      LOG_TRACE("Set backend DNN_BACKEND_CUDA");
    _net.setPreferableTarget( cv::dnn::DNN_TARGET_CUDA);                       LOG_TRACE("Set target  DNN_TARGET_CUDA");
  }else{
    _net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);                   LOG_TRACE("Set backend DNN_BACKEND_DEFAULT");
    _net.setPreferableTarget( cv::dnn::DNN_TARGET_CPU);                        LOG_TRACE("Set target  DNN_TARGET_CPU");
  }
  #endif
}

/** **************************************************************************
* try set CUDA to use specified GPU device and load the network
*
* \param cudaDeviceId device to use for hardware acceleration (-1 to disable)
*
* \returns true if successful false otherwise
*************************************************************************** */
bool DetectionLocation::loadNetToCudaDevice(const int cudaDeviceId){
  static int lastCudaDeviceId = -2;
  string errMsg = "Failed to configure CUDA for deviceID=";
  try{
    #ifdef HAVE_CUDA
    if(lastCudaDeviceId != cudaDeviceId){
      if(lastCudaDeviceId >=0){
        if(! _net.empty()) _net.~Net();                                        // need to release network this prior to device reset
        cv::cuda::resetDevice();                                               // if we were using a cuda device prior clean up old contex / cuda resources
      }
      if(cudaDeviceId >=0){
        cv::cuda::setDevice(cudaDeviceId);
        _loadNet(true);
        #ifndef NDEBUG
        cv::cuda::DeviceInfo di;                                               LOG_DEBUG("CUDA Device:" << di.name());
        #endif
        lastCudaDeviceId = cudaDeviceId;
      }else{
        _loadNet(false);
        lastCudaDeviceId = -1;
      }
    }
    return true;
    #endif
  }catch(const runtime_error& re){                                             LOG_FATAL( errMsg << cudaDeviceId << " Runtime error: " << re.what());
  }catch(const exception& ex){                                                 LOG_FATAL( errMsg << cudaDeviceId << " Exception: " << ex.what());
  }catch(...){                                                                 LOG_FATAL( errMsg << cudaDeviceId << " Unknown failure occurred. Possible memory corruption");
  }
  _loadNet(false);
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
bool DetectionLocation::Init(){

  // Load DNN Network
  string  modelClassesPath = Config::pluginPath + "/data/" + getEnv<string>({},"MODEL_CLASS_FILE"  , "coco.names");
  string  modelConfusionPath = Config::pluginPath + "/data/" + getEnv<string>({},"MODEL_CONFUSION_FILE", "yolov3-confusion-matrix.json");

  int     cudaDeviceId     = getEnv<int> ({},"CUDA_DEVICE_ID",0);

  const string errMsg = "Failed to load model: " + modelClassesPath;
  try{
      // load output class names
      ifstream classfile(modelClassesPath);
      string str;
      _netClasses.clear();
      while(getline(classfile, str)){
        _netClasses.push_back(str);
      }

      // load classification confusion matrix for yolov3-tiny
      cv::FileStorage fs(modelConfusionPath,cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
      fs["confusion"] >> _netConfusion;
      fs.release();
      cv::Mat1f cf = _netConfusion.clone();
      cv::transpose(_netConfusion,_netConfusion);  // transpose for use on score row vectors

      // calculate class groups
      cf = cf + _netConfusion;
      _classGroupIdx = intVec(cf.rows,-1);
      int g = 0;
      for(int r=0; r< cf.rows; r++){
        if(_classGroupIdx[r] < 0){
          for(int c=0; c<cf.cols; c++){
            if(cf(r,c) > 0.1){
              _classGroupIdx[c] = g;
            }
          }
          g++;
        }
      }

      // load detector net
      loadNetToCudaDevice(cudaDeviceId);

      // get names of output layers
      _netOutputNames = _net.getUnconnectedOutLayersNames();

  }catch(const runtime_error& re){                                           LOG_FATAL( errMsg << " Runtime error: " << re.what());
    return false;
  }catch(const exception& ex){                                               LOG_FATAL( errMsg << " Exception: " << ex.what());
    return false;
  }catch(...){                                                               LOG_FATAL( errMsg << " Unknown failure occurred. Possible memory corruption");
    return false;
  }

  return true;
}




/** **************************************************************************
* constructor
**************************************************************************** */
DetectionLocation::DetectionLocation(const Config     &cfg,
                                     const Frame       frame,
                                     const cv::Rect2d  bbox,
                                     const float       conf,
                                     const cv::Mat     classFeature,
                                     const cv::Mat     dftFeature = cv::Mat()):
   frame(frame),
  _cfg(cfg),
  _classFeature(classFeature),
  _dftFeature(dftFeature)
{
  confidence = conf;
  setRect(bbox & cv::Rect2d(0,0,frame.bgr.cols,frame.bgr.rows));
}