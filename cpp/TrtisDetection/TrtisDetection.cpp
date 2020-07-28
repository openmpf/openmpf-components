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

#include <fstream>
#include <stdexcept>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <chrono>

#ifdef OUTPUT_IMAGES
  #include <opencv2/opencv.hpp>
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <log4cxx/logmanager.h>
#include <log4cxx/xml/domconfigurator.h>

#include <Utils.h>
#include <detectionComponentUtils.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>

#include "TrtisDetection.hpp"
#include "base64.h"

using namespace MPF::COMPONENT;


/** ****************************************************************************
// Macro for throwing exception so we can see where in the code it happened
***************************************************************************** */
#define THROW_TRTISEXCEPTION(MSG){                               \
  string path(__FILE__);                                         \
  string f(path.substr(path.find_last_of("/\\") + 1));           \
  throw runtime_error(f + "[" + to_string(__LINE__)+"] " + MSG); \
}

/** ****************************************************************************
* Macro for error checking / logging of inference server client lib
***************************************************************************** */
#define NI_CHECK_OK(X, MSG) {                                                 \
    nic::Error err = (X);                                                     \
    if (!err.IsOk()) {                                                        \
      stringstream ss("TrtisDetection.cpp["+to_string(__LINE__)+"] ");        \
      ss << MSG << ":" << err;                                                \
      THROW_TRTISEXCEPTION(ss.str());                                         \
    }                                                                         \
}

/** ****************************************************************************
* Convenience << operator template for dumping vectors
***************************************************************************** */
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  out << "{";
  size_t last = v.size() - 1;
  for(size_t i = 0; i < v.size(); ++i){
    out << v[i];
    if(i != last) out << ", ";
  }
  out << "}";
  return out;
}

/** ****************************************************************************
* shorthands for getting MPF properties of various types
***************************************************************************** */
template<typename T>
T get(const Properties &p, const string &k, const T def){
  return DetectionComponentUtils::GetProperty<T>(p,k,def);
}

/** ****************************************************************************
* Parse TRTIS setting out of MPFJob
***************************************************************************** */
TrtisJobConfig::TrtisJobConfig(const MPFJob &job){
  const Properties jpr = job.job_properties;

  string trtis_server_default = "localhost:8001";
  trtis_server = get<string>(jpr,"TRTIS_SERVER" , trtis_server_default);

  if (trtis_server.empty()) {
      char const* tmp_env = std::getenv("TRTIS_SERVER");
      if (tmp_env != NULL) {
          trtis_server = std::string(tmp_env);
      } else {
          trtis_server = trtis_server_default;
      }
  }

  std::cout << "trtis_server: " << trtis_server << std::endl; // DEBUG

  model_name    = get<string>(jpr,"MODEL_NAME"   , "ip_irv2_coco");
  model_version = get<int>   (jpr,"MODEL_VERSION", -1);
  maxInferConcurrency = get<size_t>(jpr,"MAX_INFER_CONCURRENCY", 5);
  contextWaitTimeoutSec = get<size_t>(jpr,"CONTEXT_WAIT_TIMEOUT_SEC", 30);
}

/** ****************************************************************************
* Parse ip_irv2_coco model setting out of MPFJob
***************************************************************************** */
TrtisIpIrv2CocoJobConfig::TrtisIpIrv2CocoJobConfig(const MPFJob &job,
                                                   const size_t  image_width,
                                                   const size_t  image_height)
  :TrtisJobConfig(job),
  image_width(image_width),
  image_height(image_height),
  image_x_max(image_width-1),
  image_y_max(image_height-1),
  userBBoxNorm({0.0f,0.0f,1.0f,1.0f})
{
  const Properties jpr = job.job_properties;
  userFeatEnabled    = get<bool>(jpr, "USER_FEATURE_ENABLE",false);
  frameFeatEnabled   = get<bool>(jpr,"FRAME_FEATURE_ENABLE",true);
  classFeatEnabled   = get<bool>(jpr,"CLASS_FEATURE_ENABLE",true);
  extraFeatEnabled   = get<bool>(jpr,"EXTRA_FEATURE_ENABLE",true);
  clientScaleEnabled = get<bool>(jpr,"CLIENT_PRESCALING_ENABLE",true);
  recognitionEnroll  = get<bool>(jpr,"RECOGNITION_ENROLL_ENABLE",false);
  if(userFeatEnabled) {
    userBBox_x      = get<int64_t>(jpr, "USER_FEATURE_X_LEFT_UPPER",0);
    userBBox_y      = get<int64_t>(jpr, "USER_FEATURE_Y_LEFT_UPPER",0);
    userBBox_width  = get<int64_t>(jpr, "USER_FEATURE_WIDTH" ,image_width);
    if (userBBox_width <= 0) {
        userBBox_width = image_width;
    }
    userBBox_height = get<int64_t>(jpr, "USER_FEATURE_HEIGHT",image_height);
    if (userBBox_height <= 0) {
        userBBox_height = image_height;
    }
    userBBox.push_back(userBBox_y);
    userBBox.push_back(userBBox_x);
    userBBox.push_back(userBBox_y + userBBox_height - 1);
    userBBox.push_back(userBBox_x + userBBox_width  - 1);
    if(   userBBox[0] < 0 || userBBox[0] > image_y_max
       || userBBox[2] < 0 || userBBox[2] > image_y_max
       || userBBox[1] < 0 || userBBox[1] > image_x_max
       || userBBox[3] < 0 || userBBox[3] > image_x_max){
      THROW_TRTISEXCEPTION("Bad USER_FEATURE BBOX specification");
    }
    userBBoxNorm[0] = ((float)userBBox[0]) / image_y_max;
    userBBoxNorm[1] = ((float)userBBox[1]) / image_x_max;
    userBBoxNorm[2] = ((float)userBBox[2]) / image_y_max;
    userBBoxNorm[3] = ((float)userBBox[3]) / image_x_max;
  }

  confThreshold      = get<float>(jpr, "CONFIDENCE_THRESHOLD", 0.0);
  extraConfThreshold = get<float>(jpr, "EXTRA_CONFIDENCE_THRESHOLD", 0.0);
  maxFeatureGap      = get<float> (jpr,"TRACK_MAX_FEATURE_GAP", 0.25);
  maxFrameGap        = get<size_t>(jpr,"TRACK_MAX_FRAME_GAP", 30);
  maxSpaceGap        = get<float> (jpr,"TRACK_MAX_SPACE_GAP", 0.3);

  size_t frameDiagSq = image_width*image_width + image_height*image_height;
  maxSpaceGapPxSq    = maxSpaceGap * maxSpaceGap * frameDiagSq;

}

/** ****************************************************************************
* While more of a FEATURE detector type, 'CLASS' seems to fit
***************************************************************************** */
string TrtisDetection::GetDetectionType(){
 return "CLASS";
}

/******************************************************************************/
bool TrtisDetection::Close(){
  _infCtxs.clear();  // release inference context pool
   return true;
}

/** ****************************************************************************
* Read in classes for a model from a file, class_id is provided by the line
* number of the class label.
*
* \param model              name of inference server model (as it appear in url)
* \param class_label_file   filename from which to read class labels
* \param class_label_count  highest class_id/line number (as an integrity check)
***************************************************************************** */
void TrtisDetection::_readClassNames(string model,
                                     string class_label_file,
                                     int    class_label_count) {
  ifstream fp(class_label_file);
  if(!fp.is_open()){
    THROW_TRTISEXCEPTION("Could not read class label file: " + class_label_file);
  }

  vector<string> class_labels;
  class_labels.reserve(class_label_count);
  string line;
  while (getline(fp, line)){
    if(line.length()){ class_labels.push_back(line); }
  }
  fp.close();

  if(class_labels.size() != class_label_count){
    THROW_TRTISEXCEPTION("Read class label count of "
                        + to_string(class_labels.size()) + " in file '"
                        + class_label_file
                        + "' does not match expected count of "
                        + to_string(class_label_count));
  }else{
    _class_labels[model] = class_labels;
  }
}


/** ****************************************************************************
* Convert image colorspace to RGB and turn to bytes blob in prep for
* inference server
*
* \param      img OpenCV  image to prep for inferencing
* \param[out] shape       Shape of the scaled output data
*
* \returns  A continuous RGB data vector ready for inference server
***************************************************************************** */
BytVec TrtisDetection::_cvRGBBytes(const cv::Mat &img, LngVec& shape){
  cv::Mat rgbImg;
  if(      img.channels() == 3){
     cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);                              LOG4CXX_TRACE(_log, "Converted 3 channel BGR image to RGB");
  }else if(img.channels() == 4){
     cv::cvtColor(img, rgbImg, cv::COLOR_BGRA2RGB,3);                           LOG4CXX_TRACE(_log, "Converted 4 channel BGRA image to RGB");
  }else if(img.channels() == 1){
     cv::cvtColor(img, rgbImg, cv::COLOR_GRAY2RGB);                             LOG4CXX_TRACE(_log, "Converted 1 channel GRAY image to RGB");
  }else{
    THROW_TRTISEXCEPTION("Image could not be converted to RGB.");
  }

  if( rgbImg.type() != CV_8UC3){
    rgbImg.convertTo(rgbImg, CV_8UC3);                                          LOG4CXX_TRACE(_log, "Converted Image to CV_8U precision");
  }

  // return continuous chunk of image data in a byte vector
  BytVec data;
  size_t img_byte_size = rgbImg.total() * rgbImg.elemSize();
  data.resize(img_byte_size);

  if(rgbImg.isContinuous()){
    memcpy(&(data[0]), rgbImg.datastart, img_byte_size);
  }else{                                                                        LOG4CXX_TRACE(_log, "Converting image to have continuous data allocation");
    size_t pos = 0;
    size_t row_byte_size = rgbImg.cols * rgbImg.elemSize();
    for(int r = 0; r < rgbImg.rows; ++r){
      memcpy(&(data[pos]), rgbImg.ptr<uint8_t>(r), row_byte_size);
      pos += row_byte_size;
    }
  }
  shape = {rgbImg.rows, rgbImg.cols ,3};
  return data;
}


/** ****************************************************************************
* Scale image dimensions to ~ [target_width,target_height]
* shorter side equal target_height pixel but keep longer side below
* target_width pixels and preserve aspect ratio.
*
* \param      img          OpenCV image to be resized to fit model
* \param[out] scaleFactor  scaling factor that was used to get to target dims
*
* \returns  A scaled OpenCV image
***************************************************************************** */
cv::Mat TrtisDetection::_cvResize(const cv::Mat &img,
                                  double &scaleFactor,
                                  const size_t target_width,
                                  const size_t target_height){

  if(img.cols > img.rows){                          // landscape image
    scaleFactor = (float)target_height / img.rows;
    if(scaleFactor * img.cols > target_width){
      scaleFactor = (float)target_width / img.cols;
    }
  }else{                                             // portrait image
    scaleFactor = (float)target_height / img.cols;
    if(scaleFactor * img.rows > target_width){
      scaleFactor = (float)target_width / img.rows;
    }
  }
  cv::Mat scaledImg;
  cv::resize(img,scaledImg,cv::Size(),scaleFactor,scaleFactor,cv::INTER_CUBIC);
                                                                                LOG4CXX_TRACE(_log,"Scaled image by factor " << scaleFactor << " from ["<< img.cols << "," << img.rows << "] to [" << scaledImg.cols << "," << scaledImg.rows << "]");
  return scaledImg;
}

/** ****************************************************************************
* Scale image colorspace and dimensions and prep for inference server
*
* \param      cfg     configuration settings
* \param      img     OpenCV image to prep for inferencing
* \param      ctx     pointer to a unique inference context pointer
* \param[out] shape   shape of the imgDat tensor
* \param[out] imgDat  image tensor data
*
* \note  shape and imgDat need to persist till inference call
***************************************************************************** */
void TrtisDetection::_ip_irv2_coco_prepImageData(
                                        const TrtisIpIrv2CocoJobConfig &cfg,
                                        const cv::Mat                  &img,
                                        const uPtrInferCtx*             ctx,
                                        LngVec                         &shape,
                                        BytVec                         &imgDat){
  double scaleFactor=1.0;                                                       LOG4CXX_TRACE(_log, "Preparing image data for inferencing");
  if(cfg.clientScaleEnabled){
    imgDat = _cvRGBBytes(_cvResize(img,scaleFactor,1024,600), shape);           LOG4CXX_TRACE(_log, "using client side image scaling");
  }else{
    imgDat = _cvRGBBytes(img, shape);                                           LOG4CXX_TRACE(_log, "using TRTIS model's image scaling");
  }

  // Initialize the inputs with the data.
  sPtrInferCtxInp inImgDat, inBBox;
  NI_CHECK_OK((*ctx)->GetInput("image_input", &inImgDat),
               "unable to get image_input");
  NI_CHECK_OK((*ctx)->GetInput("bbox_input",  &inBBox),
              "unable to get bbox_input");
  NI_CHECK_OK(inImgDat->Reset(),
              "unable to reset image_input");
  NI_CHECK_OK(  inBBox->Reset(),
              "unable to reset bbox_input");
  NI_CHECK_OK(inImgDat->SetShape(shape),
              "failed setting image_input shape");
  NI_CHECK_OK(inImgDat->SetRaw(imgDat),
              "failed setting image_input");
  NI_CHECK_OK(inBBox->SetRaw((uint8_t*)(&(cfg.userBBoxNorm[0])),
                                          cfg.userBBoxNorm.size()*sizeof(float)),
              "failed setting bbox_input");                                     LOG4CXX_TRACE(_log,"Prepped data for inferencing");
}

/** ****************************************************************************
* Get or create inference context for a model.  The context(s) are cached in
* a local static container so they don't need recreated on every use.
*
* \param cfg  job configuration settings containing TRTIS server info
*
* \returns  vector of pointers to unique inferencing context pointer
*           to use for inferencing requests
***************************************************************************** */
vector<uPtrInferCtx*> TrtisDetection::_niGetInferContexts(
                                                     const TrtisJobConfig cfg)
{
  stringstream ss;
  ss << std::this_thread::get_id() << ":" << cfg.model_name << ":" << cfg.model_version;
  string key = ss.str();

  nic::Error err;
  size_t numNewCtx = cfg.maxInferConcurrency - _infCtxs[key].size();
  for(int i=0; i < numNewCtx; i++){
    uPtrInferCtx* ctx = new uPtrInferCtx;

    std::cout << "cfg.trtis_server: " << cfg.trtis_server << std::endl; // DEBUG

    NI_CHECK_OK(
      nic::InferGrpcContext::Create(ctx, cfg.trtis_server,
                                         cfg.model_name,
                                         cfg.model_version),
      "unable to create TRTIS inference context");

    // Configure context for 'batch_size'=1  and return all outputs
    uPtrInferCtxOpt options;
    NI_CHECK_OK(nic::InferContext::Options::Create(&options),
                "failed initializing TRTIS inference options");
    options->SetBatchSize(1);
    for (const auto& output : (*ctx)->Outputs()) {
      options->AddRawResult(output);
    }
    NI_CHECK_OK((*ctx)->SetRunOptions(*options),
                "failed initializing TRTIS batch size and outputs");
    _infCtxs[key].push_back(ctx);
  }
  return _infCtxs[key];

}

/** ****************************************************************************
* convert ni::DataType enum to string for logging
*
* \param dt  NVIDIA DataType enum
*
* \returns  string descriptor of enum value
***************************************************************************** */
string TrtisDetection::_niType2Str(ni::DataType dt){
  switch(dt){
    case ni::TYPE_INVALID: return "INVALID";
    case ni::TYPE_BOOL:    return "BOOL";
    case ni::TYPE_UINT8:   return "UINT8";
    case ni::TYPE_UINT16:  return "UINT16";
    case ni::TYPE_UINT32:  return "UINT32";
    case ni::TYPE_UINT64:  return "UINT64";
    case ni::TYPE_INT8:    return "INT8";
    case ni::TYPE_INT16:   return "INT16";
    case ni::TYPE_INT32:   return "INT32";
    case ni::TYPE_INT64:   return "INT64";
    case ni::TYPE_FP16:    return "FP16";
    case ni::TYPE_FP32:    return "FP32";
    case ni::TYPE_FP64:    return "FP64";
    case ni::TYPE_STRING:  return "STRING";
    default:               return "UNKNOWN";
  }
}

/** ****************************************************************************
* Create OpenCV matrix header to more easily access InferenceServer results
* Note: No actual data is moved or copied
*
* \param batch_idx  index for batches of images (set to 0 for batch size of 1)
* \param name       name of the inference server result tensor to convert
* \param results    collection of tensors keyed by string names
*
* \returns  an openCV matrix corresponding to the tensor
***************************************************************************** */
cv::Mat  TrtisDetection::_niResult2CVMat(const size_t           batch_idx,
                                         const string           name,
                                         StrUPtrInferCtxResMap &results){

  // get raw data pointer and size
  const uint8_t* ptrRaw;
  size_t cntRaw;
  uPtrInferCtxRes *res = &results.at(name);
  NI_CHECK_OK((*res)->GetRaw(batch_idx,&ptrRaw,&cntRaw),
              "Failed to get inference server result raw data");

  // get raw data shape
  LngVec shape;
  NI_CHECK_OK((*res)->GetRawShape(&shape),
    "Failed to get inference server result shape");
  int ndim = shape.size();
  if(ndim < 2){ // force matrix for vector with single col?!
    ndim = 2;
    shape.push_back(1);
  }

  // calculate num elements from shape
  IntVec iShape;
  size_t numElementsFromShape = 1;
  for (const auto& d: shape){
    numElementsFromShape *= d;
    iShape.push_back((int)d);
  }

  // determine opencv type and calculate num elements from raw size and data type
  size_t cvType;
  size_t sizeofEl;
  ni::DataType niType = (*res)->GetOutput()->DType();
  switch(niType){
    case ni::TYPE_UINT8:  cvType = CV_8UC(ndim-1);  sizeofEl=sizeof(uint8_t);   break;
    case ni::TYPE_UINT16: cvType = CV_16UC(ndim-1); sizeofEl=sizeof(uint16_t);  break;
    case ni::TYPE_INT8:   cvType = CV_8SC(ndim-1);  sizeofEl=sizeof(int8_t);    break;
    case ni::TYPE_INT16:  cvType = CV_16SC(ndim-1); sizeofEl=sizeof(int16_t);   break;
    case ni::TYPE_INT32:  cvType = CV_32SC(ndim-1); sizeofEl=sizeof(int32_t);   break;
    case ni::TYPE_FP32:   cvType = CV_32FC(ndim-1); sizeofEl=sizeof(float);     break;
    case ni::TYPE_FP64:   cvType = CV_64FC(ndim-1); sizeofEl=sizeof(double);    break;
    // OpenCV does not support these types ?!
    case ni::TYPE_UINT32: //cvType = CV_32UC(ndim-1); sizeofEl=sizeof(uint32_t);  break;
    case ni::TYPE_UINT64: //cvType = CV_64UC(ndim-1); sizeofEl=sizeof(uint64_t);  break;
    case ni::TYPE_INT64:  //cvType = CV_64SC(ndim-1); sizeofEl=sizeof(int64_t);   break;
    case ni::TYPE_FP16:   //cvType = CV_16FC(ndim-1); sizeofEl=sizeof(float16_t); break;
    case ni::TYPE_BOOL:
    case ni::TYPE_STRING:
    case ni::TYPE_INVALID:
      THROW_TRTISEXCEPTION("Unsupported data_type " + _niType2Str(niType) + " in cv:Mat conversion");
  }

  if( cntRaw/sizeofEl == numElementsFromShape){
    return cv::Mat(ndim, iShape.data(), cvType, (void*)ptrRaw);
  }else{
    stringstream ss("Shape ");
    ss << shape << " and data-type " << _niType2Str(niType) << "are inconsistent with buffer size " << cntRaw;
    THROW_TRTISEXCEPTION(ss.str());
  }
}

/** ****************************************************************************
* Initialize the Trtis detector module by setting up paths and reading configs
*
* \returns  true on success
***************************************************************************** */
bool TrtisDetection::Init() {

    // Determine where the executable is running
    string run_dir = GetRunDirectory();
    if(run_dir.empty()){ run_dir = "."; }

    string plugin_path = run_dir + "/TrtisDetection";
    string config_path = plugin_path + "/config";
    string models_path = plugin_path + "/models";

    // Configure logger
    log4cxx::xml::DOMConfigurator::configure(config_path + "/Log4cxxConfig.xml");
    _log = log4cxx::Logger::getLogger("TrtisDetection");

    try {
      // Read class labels for model(s)
      _readClassNames("ip_irv2_coco", models_path + "/ip_irv2_coco/ip_irv2_coco.labels", 90);
    }catch(const exception &ex){
        LOG4CXX_ERROR(_log, "Init failed: " << ex.what())
        return false;
    }                                                                           LOG4CXX_DEBUG(_log, "Plugin path: " << plugin_path);

    return true;
}



/** ****************************************************************************
* Get detected object from an image/video frame and add it to locations vector
* with appropriate properties for the model inferenced.
*
* \param         cfg        configuration for MPFImage of VideoJob job
* \param         res        inference results from TRTIS
* \param[in,out] locations  vector of locations for found detections
*
* \note ip_irv2_coco model: (only one implemented so far)
*       This model supports COCO classifications and provides custom features
*       suitable for similarity searching using cos/inner-product distance.
*       The user can dictate a bounding box to specify an custom object location
*       for which a feature vector will be returned.  Additionally 'extra'
*       detection bounding boxes that did not readily classify to a COCO class
*       can also be returned. Finally a global image/frame feature is provided
*       that consists of a size weighted average of the detection features.
*       This can be useful in similarity searches or possibly scene segmentation.
*
***************************************************************************** */
void TrtisDetection::_ip_irv2_coco_getDetections(
                                 const TrtisIpIrv2CocoJobConfig &cfg,
                                 StrUPtrInferCtxResMap          &res,
                                 MPFImageLocationVec            &locations) {

  if(cfg.frameFeatEnabled){                                                     LOG4CXX_TRACE(_log,"processing global feature");
    cv::Mat g_feat    = _niResult2CVMat(0,"G_Feat"   ,res);
    cv::normalize(g_feat,g_feat,1,0,cv::NORM_L2);                               LOG4CXX_TRACE(_log,"normalized global feature");
    locations.push_back(MPFImageLocation(
      0 ,0 ,cfg.image_x_max ,cfg.image_y_max, -1.0,
      {{"FEATURE-TYPE","FRAME"},
       {"FEATURE",string((const char*)g_feat.data,
                         g_feat.total() * g_feat.elemSize())}}));               LOG4CXX_TRACE(_log,"added global feature to locations");
  }

  if(cfg.userFeatEnabled){                                                      LOG4CXX_TRACE(_log,"processing user bbox specified feature");
    cv::Mat m_feat    = _niResult2CVMat(0,"M_Feat"   ,res);
    cv::normalize(m_feat,m_feat,1,0,cv::NORM_L2);                               LOG4CXX_TRACE(_log,"normalized user bbox specified feature");
    locations.push_back(MPFImageLocation(
      cfg.userBBox_x,cfg.userBBox_y,cfg.userBBox_width ,cfg.userBBox_height,
     -1.0,
      {{"FEATURE-TYPE","USER"},
       {"FEATURE",string((const char*)m_feat.data,
                         m_feat.total() * m_feat.elemSize())}}));               LOG4CXX_TRACE(_log,"added user bbox feature to locations");
  }

  if(cfg.classFeatEnabled){                                                     LOG4CXX_TRACE(_log,"processing detected object features with classifications");
    cv::Mat d_bboxes  = _niResult2CVMat(0,"D_BBoxes" ,res);
    cv::Mat d_classes = _niResult2CVMat(0,"D_Classes",res);
    cv::Mat d_scores  = _niResult2CVMat(0,"D_Scores" ,res);
    cv::Mat d_feats   = _niResult2CVMat(0,"D_Feats"  ,res);
    auto classes = _class_labels["ip_irv2_coco"];
    for(int r = 0; r < d_feats.rows; r++){
      float conf = d_scores.at<float>(r,0);
      if(conf >= cfg.confThreshold){
        cv::normalize(d_feats.row(r),d_feats.row(r),1,0,cv::NORM_L2);
        int class_id = ((int)d_classes.at<float>(r,0));
        string label = (class_id <= 90) ? classes[class_id - 1] : "unknown";
        locations.push_back(MPFImageLocation(
           d_bboxes.at<float>(r,1) * (cfg.image_x_max)
          ,d_bboxes.at<float>(r,0) * (cfg.image_y_max)
          ,(d_bboxes.at<float>(r,3)-d_bboxes.at<float>(r,1))*(cfg.image_x_max)
          ,(d_bboxes.at<float>(r,2)-d_bboxes.at<float>(r,0))*(cfg.image_y_max)
          ,conf
          ,{{"FEATURE-TYPE","CLASS"},
            {"FEATURE",string((const char*)d_feats.ptr(r),
                              d_feats.cols * d_feats.elemSize())},
            {"CLASSIFICATION", label }}));                                      LOG4CXX_TRACE(_log,"detected object with class[" << class_id << "] = " << label << " with confidence(" << locations.back().confidence << ")" << " at [" << locations.back().x_left_upper << "," << locations.back().y_left_upper << ","  << locations.back().width        << "," << locations.back().height << "]");
      }
    }                                                                           LOG4CXX_TRACE(_log,"added detected object features to locations");
  }

  if(cfg.extraFeatEnabled){                                                     LOG4CXX_TRACE(_log,"processing extra detected objects without classifications");
    cv::Mat e_bboxes  = _niResult2CVMat(0,"E_BBoxes" ,res);
    cv::Mat e_scores  = _niResult2CVMat(0,"E_Scores" ,res);
    cv::Mat e_feats   = _niResult2CVMat(0,"E_Feats"  ,res);
    for(int r = 0; r < e_feats.rows; r++){
      float conf = e_scores.at<float>(r,0);
      if(conf >= cfg.extraConfThreshold){
        cv::normalize(e_feats.row(r),e_feats.row(r),1,0,cv::NORM_L2);
        locations.push_back(MPFImageLocation(
           e_bboxes.at<float>(r,1) * cfg.image_x_max
          ,e_bboxes.at<float>(r,0) * cfg.image_y_max
          ,(e_bboxes.at<float>(r,3) - e_bboxes.at<float>(r,1)) * cfg.image_x_max
          ,(e_bboxes.at<float>(r,2) - e_bboxes.at<float>(r,0)) * cfg.image_y_max
          ,conf
          ,{{"FEATURE-TYPE","EXTRA"},
            {"FEATURE",string((const char*)e_feats.ptr(r),
                              e_feats.cols * e_feats.elemSize())}}));             LOG4CXX_TRACE(_log,"detected extra object with confidence(" << locations.back().confidence << ")" << " at [" << locations.back().x_left_upper << "," << locations.back().y_left_upper << ","     << locations.back().width        << "," << locations.back().height << "]");
      }
    }
    LOG4CXX_TRACE(_log,"added detected extra features to locations");
  }

  if(cfg.recognitionEnroll){
    // _recognitionFrameworkEnroll(job,locations);
    // * needs to be implemented
    THROW_TRTISEXCEPTION("Recognition Framework Enroll Interface not implemented");
  }

}


/** ****************************************************************************
* base64 encode the feature in the track stop location
*
* \param[in,out] tracks  tracks to perform stop feature decoding on.
*
***************************************************************************** */
void TrtisDetection::_base64EncodeStopFeatures(MPFVideoTrackVec &tracks){
  for(MPFVideoTrack &track : tracks){
    auto dt = track.frame_locations[track.stop_frame].detection_properties;
    auto it = dt.find("FEATURE");
    if(it != dt.end()){                                                         LOG4CXX_TRACE(_log, "base64 encoding track stop_frame[" << track.stop_frame << "] FEATURE");
      dt["FEATURE"] = Base64::Encode(it->second);
    }
  }
}
/** ****************************************************************************
* Compute a similarity score based on Cos / Inner Product Distance
*
* \param p1    1st feature vector to be used in distance calc.
* \param p2    2nd feature vector to be used in distance calc.
* \param size  number of floats in each of the two vectors
*
* \returns  cos similarity of the two vectors (-1...1)
*
* \note  vectors are assumed to be normalized to magnitude of 1.0
***************************************************************************** */
static float ipSimilarity(const float p1[],
                     const float p2[],
                     const size_t size) {
    float res = 0;
    for (size_t i = 0; i < size; i++) {
        res += p1[i] * p2[i];
    }
    return res;
}

/** ****************************************************************************
* Computes distance squared between centers of location bounding boxes
*
* \param l1  1st location to be used in distance calc.
* \param l2  2nd location to be used in distance calc.
*
* \returns  pixel Euclidean distance squared between centers
*
***************************************************************************** */
static float centerDistSq(const MPFImageLocation &l1, const MPFImageLocation &l2){
  float dx = l1.x_left_upper - l2.x_left_upper + (l1.width  - l2.width ) / 2;
  float dy = l1.y_left_upper - l2.y_left_upper + (l1.height - l2.height) / 2;
  return dx*dx + dy*dy;
}

/** ****************************************************************************
* Adds a location to an object track and updates the track's classification
*  and confidence if they exist
*
* \param         location  location to add to track
* \param         frameIdx  frame index of location
* \param[in,out] track     track to which location will be appended
*
***************************************************************************** */
void TrtisDetection::_addToTrack(MPFImageLocation &location,
                                 int               frameIdx,
                                 MPFVideoTrack    &track) {
    track.stop_frame = frameIdx;
    if (location.confidence > track.confidence) {
        track.confidence = location.confidence;
        auto it = location.detection_properties.find("CLASSIFICATION");
        if(it != location.detection_properties.end()){
          track.detection_properties["CLASSIFICATION"] = it->second;            LOG4CXX_TRACE(_log, "updating track class to " << it->second);
        }
    }
    track.frame_locations[frameIdx] = move(location);
}

/** ****************************************************************************
* Determine if and to which track a detection location should be appended, or
* if a new track should be created for it.
*
* \param         cfg       configuration settings for job
* \param         loc       location to add to tracks collection
* \param         frameIdx  frame index of location
* \param[in,out] tracks    collection to which location will be added
*
* \note  Frame and center bounds define a search space of track stop locations
*        from which the one with the smallest feature distance is selected.
*        If no candidates exist, a new track is created.
*        This is just one simplistic tracker, there are a lot of
*        additional possibilities:
*          - could do scene change track breaks based on FRAME feature change
*          - could track things that don't have classifications (EXTRA features)
*          - could select based on some combined time-space-feature distance
*          - could use bbox size and maybe movement ...
*
***************************************************************************** */
void TrtisDetection::_ip_irv2_coco_tracker(
                          const TrtisIpIrv2CocoJobConfig &cfg,
                          MPFImageLocation               &loc,
                          const int                       frameIdx,
                          MPFVideoTrackVec               &tracks){

  const string feature_type = loc.detection_properties["FEATURE-TYPE"];
  if( feature_type == "CLASS" || feature_type == "EXTRA"){                      LOG4CXX_TRACE(_log, "Found detection with feature_type:" << feature_type);
    MPFVideoTrack* bestTrackPtr  = NULL;
    float          minFeatureGap = FLT_MAX;
    for(MPFVideoTrack &track : tracks){
      size_t frameGap = frameIdx - track.stop_frame;
      if( frameGap > 0 && frameGap <= cfg.maxFrameGap){
        MPFImageLocation stopLoc = track.frame_locations[track.stop_frame];
        size_t spaceGapPxSq = centerDistSq(stopLoc,loc);
        if(spaceGapPxSq <= cfg.maxSpaceGapPxSq){
          auto trkFeat = stopLoc.detection_properties.find("FEATURE");
          auto locFeat = loc.detection_properties.find("FEATURE");
          if(   trkFeat !=  stopLoc.detection_properties.end()
             && locFeat !=      loc.detection_properties.end()){
            float* stopFeature=(float*)(&(trkFeat->second[0]));
            float* locFeature =(float*)(&(locFeat->second[0]));
            float featureGap = 1.0f - ipSimilarity(stopFeature,locFeature,1088);
            if(featureGap <= cfg.maxFeatureGap){                                LOG4CXX_TRACE(_log, "featureGap = " << featureGap << " < " << cfg.maxFeatureGap);
              if(featureGap < minFeatureGap){                                   LOG4CXX_TRACE(_log, "bestTrack = " << &track);
                bestTrackPtr  = &track;
                minFeatureGap = featureGap;
              }
            }
          }
        }
      }
    }

    if(bestTrackPtr != NULL){                                                   LOG4CXX_TRACE(_log, "Adding to track(&" << bestTrackPtr << ") from frame[" << frameIdx <<"]");
      int bestStopFrame = bestTrackPtr->stop_frame;
      MPFImageLocation bestStopLoc = bestTrackPtr->frame_locations[bestStopFrame];
     if(bestStopFrame != bestTrackPtr->start_frame){
        bestStopLoc.detection_properties.erase("FEATURE");                      LOG4CXX_TRACE(_log,"Erased previous FEATURE for track(& " << bestTrackPtr << ") from frame[" << frameIdx <<"]");
     }else{ // keep start frame feature for linking ?!
        bestStopLoc.detection_properties["FEATURE"] =
        Base64::Encode(bestStopLoc.detection_properties["FEATURE"]);            LOG4CXX_TRACE(_log,"Re-encoded track start feature");
     }
     _addToTrack(loc, frameIdx, *bestTrackPtr);
    }else{
      tracks.emplace_back(frameIdx, frameIdx);                                  LOG4CXX_TRACE(_log, "Created new track(&" << &(tracks.back()) << ") from frame[" << frameIdx <<"]");
      _addToTrack(loc, frameIdx, tracks.back());
    }
  }
}

/** ****************************************************************************
* Read frames from a video, get object detections and make tracks
*
* \param         job     MPF Video job
*
* \throws an MPF error constant or TRTIS runtime error
*
* \returns Tracks collection to which detections will be added
***************************************************************************** */
std::vector<MPF::COMPONENT::MPFVideoTrack> TrtisDetection::GetDetections(const MPFVideoJob &job) {
  LOG4CXX_INFO(_log, "[" << job.job_name << "] Starting job");

  // if (job.has_feed_forward_track) { do something different ?!}
  try{
    if(job.data_uri.empty()){
      LOG4CXX_ERROR(_log, "[" << job.job_name << "] Input video file path is empty");
      throw MPF_INVALID_DATAFILE_URI;
    }

    MPFVideoCapture video_cap(job);
    if(!video_cap.IsOpened()){
      LOG4CXX_ERROR(_log, "[" << job.job_name << "] Could not initialize capturing");
      throw MPF_COULD_NOT_OPEN_DATAFILE;
    }

    std::vector<MPF::COMPONENT::MPFVideoTrack> tracks;
    cv::Mat frame;
    Properties jpr = job.job_properties;
    string model_name = get<string>(jpr,"MODEL_NAME",  "ip_irv2_coco");

    if(model_name == "ip_irv2_coco"){

      // need to read a frame to get size
      if(!video_cap.Read(frame)) return tracks;
      TrtisIpIrv2CocoJobConfig cfg(job, frame.cols,frame.rows);

      mutex poolMtx, tracksMtx, nextRxFrameMtx;                                 LOG4CXX_TRACE(_log, "Main thread_id:" << std::this_thread::get_id());
      condition_variable cv, cvf;
      auto timeout = chrono::seconds(cfg.contextWaitTimeoutSec);
      vector<uPtrInferCtx*> ctxPool = _niGetInferContexts(cfg);                 LOG4CXX_TRACE(_log,"Retrieved inferencing context pool of size " << ctxPool.size() << " for model '" << cfg.model_name << "' from server " << cfg.trtis_server);
      size_t initialCtxPoolSize = ctxPool.size();
      int frameIdx = 0;
      int nextRxFrameIdx = 0;
      do{                                                                       LOG4CXX_TRACE(_log,"requesting inference from TRTIS server for frame[" <<  frameIdx << "]" );
        uPtrInferCtx* ctx;
        { unique_lock<mutex> lk(poolMtx);
          if(ctxPool.size() < 1){                                               LOG4CXX_TRACE(_log,"wait for an infer context to become available");
            if(!cv.wait_for(lk, timeout,[&ctxPool]{return ctxPool.size() > 0;})){
              THROW_TRTISEXCEPTION("Waited longer than " +
                to_string(timeout.count()) + " sec for an inference context.");
            }
          }
          ctx  = ctxPool.back();
          ctxPool.pop_back();                                                   LOG4CXX_TRACE(_log,"removing ctx["<<ctxPool.size() <<"] from pool");
        }

        LngVec shape;
        BytVec imgDat;
        _ip_irv2_coco_prepImageData(cfg, frame, ctx, shape, imgDat);            LOG4CXX_TRACE(_log,"Loaded data into inference context");
                                                                                LOG4CXX_DEBUG(_log,"frame["<<frameIdx<<"] sending");
        // Send inference request to the inference server.
        NI_CHECK_OK(
          (*ctx)->AsyncRun([&,frameIdx,ctx](nic::InferContext* c,
                                            sPtrInferCtxReq    req){
            { unique_lock<mutex> lk(nextRxFrameMtx);
              if(frameIdx != nextRxFrameIdx){                                   LOG4CXX_TRACE(_log,"out of sequence frame response, expected frame[" << nextRxFrameIdx << "] but received frame[" << frameIdx << "]");
                if(!cvf.wait_for(lk,
                                 timeout,
                                 [&frameIdx, &nextRxFrameIdx]{return frameIdx == nextRxFrameIdx;})){
                  THROW_TRTISEXCEPTION("Waited longer than " +
                    to_string(timeout.count()) + " sec for response to request for frame["+to_string(nextRxFrameIdx)+"].");
                }
              }
            }
            { lock_guard<mutex> lk(nextRxFrameMtx);
              nextRxFrameIdx++;
              cvf.notify_all();
            }
            StrUPtrInferCtxResMap res;                                          LOG4CXX_TRACE(_log, "Callback[" << frameIdx << "] thread_id:" << std::this_thread::get_id());
            bool is_ready = false;
            NI_CHECK_OK(
              c->GetAsyncRunResults(&res,&is_ready,req,true),
              "Failed to retrieve inference results");
            if(!is_ready){
              THROW_TRTISEXCEPTION("Inference results not ready during callback");
            }                                                                   LOG4CXX_TRACE(_log, "inference complete");
            MPFImageLocationVec locations;
            _ip_irv2_coco_getDetections(cfg, res, locations);                   LOG4CXX_TRACE(_log, "inferenced frame[" <<  frameIdx << "]");
            for(MPFImageLocation &loc : locations){
              lock_guard<mutex> lk(tracksMtx);
              _ip_irv2_coco_tracker(cfg, loc, frameIdx, tracks);
            }                                                                   LOG4CXX_TRACE(_log, "tracked objects in frame[" <<  frameIdx << "]");

            { lock_guard<mutex> lk(poolMtx);
              ctxPool.push_back(ctx);
              cv.notify_all();                                                  LOG4CXX_TRACE(_log, "returned ctx[" << ctxPool.size() - 1 << "] to pool");
            }                                                                   LOG4CXX_DEBUG(_log,"frame["<<frameIdx<<"] complete");
          }),
          "unable to inference '" + cfg.model_name + "' ver."
                                  + to_string(cfg.model_version));              LOG4CXX_TRACE(_log,"Inference request sent");
                                                                                LOG4CXX_DEBUG(_log,"frame["<<frameIdx<<"] sent");


        frameIdx++;
      } while (video_cap.Read(frame));

      if(ctxPool.size() < initialCtxPoolSize){                                  LOG4CXX_TRACE(_log,"wait inference context pool size to return to initial size of " << initialCtxPoolSize );
        unique_lock<mutex> lk(poolMtx);
        if(!cv.wait_for(lk, cfg.maxInferConcurrency * timeout
                        ,[&ctxPool, &initialCtxPoolSize]{return ctxPool.size() == initialCtxPoolSize;})){
                           THROW_TRTISEXCEPTION("Waited longer than " +
                             to_string(timeout.count()) +
                             " sec for context pool to return to initial size.");
        }
      }                                                                         LOG4CXX_DEBUG(_log,"all frames complete");

      _base64EncodeStopFeatures(tracks);                                        LOG4CXX_TRACE(_log, "finished (re)encoding track stop_frame FEATUREs to base64");
    }else{
      THROW_TRTISEXCEPTION("Unsupported model type:" + model_name);
    }

    for (MPFVideoTrack &track : tracks) {
      video_cap.ReverseTransform(track);
    }

    LOG4CXX_INFO(_log, "[" << job.job_name << "] Found " << tracks.size() << " tracks.");

    return tracks;

  }catch(...){
    _infCtxs.clear();  // release inference context pool just to be safe
    Utils::LogAndReThrowException(job, _log);
  }

}

/** ****************************************************************************
* Read an image and get object detections and features
*
* \param         job        MPF Image job
*
* \throws an MPF error constant or TRTIS runtime error
*
* \returns locations collection to which detections will be added
*
* \note  prior to returning the features are base64 encoded
*
***************************************************************************** */
std::vector<MPFImageLocation> TrtisDetection::GetDetections(const MPFImageJob   &job) {
  try{                                                                          LOG4CXX_DEBUG(_log, "Data URI = " << job.data_uri);
    LOG4CXX_INFO(_log, "[" << job.job_name << "] Starting job");

    if(job.data_uri.empty()){
      LOG4CXX_ERROR(_log, "Invalid image file");
      throw MPF_INVALID_DATAFILE_URI;
    }

    std::vector<MPFImageLocation> locations;
    MPFImageReader image_reader(job);
    cv::Mat img = image_reader.GetImage();

    if(img.empty()){
      LOG4CXX_ERROR(_log, "Could not read image file: " << job.data_uri);
      throw MPF_IMAGE_READ_ERROR;
    }

    Properties jpr = job.job_properties;
    string model_name = get<string>(jpr,"MODEL_NAME",  "ip_irv2_coco");

    if(model_name == "ip_irv2_coco"){

      TrtisIpIrv2CocoJobConfig cfg(job, img.cols, img.rows);                    LOG4CXX_TRACE(_log,"parsed job configuration settings");
      cfg.maxInferConcurrency = 1;
      uPtrInferCtx* ctx  = _niGetInferContexts(cfg)[0];                         LOG4CXX_TRACE(_log,"retrieved inferencing context for model '" << cfg.model_name << "' from server " << cfg.trtis_server);

      LngVec shape;
      BytVec imgDat;
      _ip_irv2_coco_prepImageData(cfg, img, ctx, shape, imgDat);                LOG4CXX_TRACE(_log,"loaded data into inference context");

      // Send inference request to the inference server.
      StrUPtrInferCtxResMap res;
      NI_CHECK_OK((*ctx)->Run(&res),"unable to inference '" + cfg.model_name
        + "' ver." + to_string(cfg.model_version));                             LOG4CXX_TRACE(_log,"inference complete");

      size_t next_idx = locations.size();
      _ip_irv2_coco_getDetections(cfg, res, locations);
      size_t new_size = locations.size();                                       LOG4CXX_TRACE(_log,"parsed detections into locations vector");

      for(size_t i = next_idx; i < new_size; i++){
        // not sure if base64 encoding is needed here,
        // might make sense for whatever serializes the output to do the encoding?!
        locations[i].detection_properties["FEATURE"] = Base64::Encode(locations[i].detection_properties["FEATURE"]);
        image_reader.ReverseTransform(locations[i]);
      }                                                                         LOG4CXX_TRACE(_log,"base64 encoded new features in locations vector");

      LOG4CXX_INFO(_log, "[" << job.job_name << "] Found " << locations.size() << " detections.");

      return locations;

    }else{
      THROW_TRTISEXCEPTION("Unsupported model type:" + model_name);
    }

  }catch (...) {
    _infCtxs.clear();  // release inference context pool just to be safe
    Utils::LogAndReThrowException(job, _log);
  }

}


//-----------------------------------------------------------------------------

MPF_COMPONENT_CREATOR(TrtisDetection);
MPF_COMPONENT_DELETER();