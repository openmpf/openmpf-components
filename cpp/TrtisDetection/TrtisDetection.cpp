/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
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
#include <mutex>
#include <chrono>
#include <condition_variable>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <log4cxx/logmanager.h>

#include <Utils.h>
#include <detectionComponentUtils.h>
#include <MPFImageReader.h>
#include <MPFVideoCapture.h>

#include "TrtisDetection.h"
#include "S3StorageUtil.h"
#include "S3FeatureStorage.h"
#include "EncodeFeatureStorage.h"
#include "base64.h"

using namespace MPF::COMPONENT;

/** ****************************************************************************
* Macro for throwing exception so we can see where in the code it happened
***************************************************************************** */
#define THROW_TRTISEXCEPTION(X, MSG) {                                        \
    MPFDetectionError e = (X);                                                \
    throw MPFDetectionException(e,                                            \
      /* "Error in TrtisDetection.cpp[" + to_string(__LINE__) + "]: " + */    \
      (MSG));                                                                 \
}

/** ****************************************************************************
* Macro for error checking / logging of inference server client lib
***************************************************************************** */
#define NI_CHECK_OK(X, MSG) {                                                 \
    nic::Error e = (X);                                                       \
    if (!e.IsOk()) {                                                          \
      throw MPFDetectionException(MPF_OTHER_DETECTION_ERROR_TYPE,             \
        string("NVIDIA inference server error")                               \
        /* + " in TrtisDetection.cpp[" + to_string( __LINE__) + "]" */        \
        + ": " + (MSG) + ": " + e.Message());                                 \
    }                                                                         \
}

/** ****************************************************************************
* Convenience << operator template for dumping vectors
***************************************************************************** */
template<typename T>
ostream &operator<<(ostream &out, const vector <T> &v) {
    out << "{";
    size_t last = v.size() - 1;
    for (size_t i = 0; i < v.size(); ++i) {
        out << v[i];
        if (i != last) out << ", ";
    }
    out << "}";
    return out;
}

/** ****************************************************************************
* shorthands for getting configuration from environment variables if not
* provided by job configuration
***************************************************************************** */
template<typename T>
T getEnv(const Properties &p, const string &k, const T &def) {
    auto iter = p.find(k);
    if (iter != p.end()) {
        if (!(*iter).second.empty()) { // only use if something other than the blank string ("") was provided
            return DetectionComponentUtils::GetProperty<T>(p, k, def);
        }
    }
    const char *env_p = getenv(k.c_str());
    if (env_p == nullptr) {
        return def; // use default value if no env. var.
    }
    return DetectionComponentUtils::GetProperty<T>({{k, string(env_p)}}, k, def); // convert env. var. to desired type
}

/** ****************************************************************************
* shorthands for getting MPF properties of various types
***************************************************************************** */
template<typename T>
T get(const Properties &p, const string &k, const T def) {
    return DetectionComponentUtils::GetProperty<T>(p, k, def);
}

/** ****************************************************************************
* Parse TRTIS setting out of MPFJob
***************************************************************************** */
TrtisJobConfig::TrtisJobConfig(const MPFJob &job,
                               const log4cxx::LoggerPtr &log) :
        data_uri(job.data_uri), featureStorage(_getFeatureStorage(job, log)) {
    const Properties jpr = job.job_properties;

    trtis_server = getEnv<string>(jpr, "TRTIS_SERVER", "localhost:8001");
    LOG4CXX_TRACE(log, "TRTIS_SERVER: " << trtis_server);

    model_name = get<string>(jpr, "MODEL_NAME", "ip_irv2_coco");
    LOG4CXX_TRACE(log, "MODEL_NAME: " << model_name);

    model_version = get<int>(jpr, "MODEL_VERSION", -1);
    LOG4CXX_TRACE(log, "MODEL_VERSION: " << model_version);

    maxInferConcurrency = get<int>(jpr, "MAX_INFER_CONCURRENCY", 5);
    LOG4CXX_TRACE(log, "MAX_INFER_CONCURRENCY: " << maxInferConcurrency);
}

/** ****************************************************************************
* Get the feature storage helper
***************************************************************************** */
IFeatureStorage::uPtrFeatureStorage TrtisJobConfig::_getFeatureStorage(const MPFJob &job,
                                                                       const log4cxx::LoggerPtr &log) {
    if (S3StorageUtil::RequiresS3Storage(job)) {
        return IFeatureStorage::uPtrFeatureStorage(new S3FeatureStorage(job, log));
    } else {
        return IFeatureStorage::uPtrFeatureStorage(new EncodeFeatureStorage());
    }
}

/** ****************************************************************************
* Parse ip_irv2_coco model setting out of MPFJob
***************************************************************************** */
TrtisIpIrv2CocoJobConfig::TrtisIpIrv2CocoJobConfig(const MPFJob &job,
                                                   const log4cxx::LoggerPtr &log,
                                                   const int image_width,
                                                   const int image_height)
        : TrtisJobConfig(job, log),
          image_x_max(image_width - 1),
          image_y_max(image_height - 1),
          userBBoxNorm({0.0f, 0.0f, 1.0f, 1.0f}) {
    const Properties jpr = job.job_properties;
    userFeatEnabled = get<bool>(jpr, "USER_FEATURE_ENABLE", false);
    frameFeatEnabled = get<bool>(jpr, "FRAME_FEATURE_ENABLE", true);
    classFeatEnabled = get<bool>(jpr, "CLASS_FEATURE_ENABLE", true);
    extraFeatEnabled = get<bool>(jpr, "EXTRA_FEATURE_ENABLE", true);
    clientScaleEnabled = get<bool>(jpr, "CLIENT_PRESCALING_ENABLE", true);
    recognitionEnroll = get<bool>(jpr, "RECOGNITION_ENROLL_ENABLE", false);
    if (userFeatEnabled) {
        userBBox_x = get<int>(jpr, "USER_FEATURE_X_LEFT_UPPER", 0);
        userBBox_y = get<int>(jpr, "USER_FEATURE_Y_LEFT_UPPER", 0);
        userBBox_width = get<int>(jpr, "USER_FEATURE_WIDTH", image_width);
        if (userBBox_width <= 0) {
            userBBox_width = image_width - userBBox_x;
        }
        userBBox_height = get<int>(jpr, "USER_FEATURE_HEIGHT", image_height);
        if (userBBox_height <= 0) {
            userBBox_height = image_height - userBBox_y;
        }
        userBBox.push_back(userBBox_y);
        userBBox.push_back(userBBox_x);
        userBBox.push_back(userBBox_y + userBBox_height - 1);
        userBBox.push_back(userBBox_x + userBBox_width - 1);
        if (userBBox[0] < 0 || userBBox[0] > image_y_max
            || userBBox[2] < 0 || userBBox[2] > image_y_max
            || userBBox[1] < 0 || userBBox[1] > image_x_max
            || userBBox[3] < 0 || userBBox[3] > image_x_max) {
            THROW_TRTISEXCEPTION(MPF_INVALID_PROPERTY, "Bad USER_FEATURE BBOX specification");
        }
        userBBoxNorm[0] = ((float) userBBox[0]) / image_y_max;
        userBBoxNorm[1] = ((float) userBBox[1]) / image_x_max;
        userBBoxNorm[2] = ((float) userBBox[2]) / image_y_max;
        userBBoxNorm[3] = ((float) userBBox[3]) / image_x_max;
    }

    classConfThreshold = get<float>(jpr, "CLASS_CONFIDENCE_THRESHOLD", 0.0);
    extraConfThreshold = get<float>(jpr, "EXTRA_CONFIDENCE_THRESHOLD", 0.0);
    maxFeatureGap = get<float>(jpr, "TRACK_MAX_FEATURE_GAP", 0.25);
    maxFrameGap = get<int>(jpr, "TRACK_MAX_FRAME_GAP", 30);
    maxSpaceGap = get<float>(jpr, "TRACK_MAX_SPACE_GAP", 0.3);

    int frameDiagSq = image_width * image_width + image_height * image_height;
    maxSpaceGapPxSq = maxSpaceGap * maxSpaceGap * frameDiagSq;
}

/******************************************************************************/
string TrtisDetection::GetDetectionType() {
    return "FEATURE";
}

/******************************************************************************/
bool TrtisDetection::Close() {
    return true;
}

/** ****************************************************************************
* Read in classes for a model from a file, class_id is provided by the line
* number of the class label.
*
* \param model              name of inference server model (as it appears in url)
* \param class_label_file   filename from which to read class labels
* \param class_label_count  highest class_id/line number (as an integrity check)
***************************************************************************** */
void TrtisDetection::_readClassNames(const string &model,
                                     const string &class_label_file,
                                     int class_label_count) {
    ifstream fp(class_label_file);
    if (!fp.is_open()) {
        THROW_TRTISEXCEPTION(MPF_INVALID_PROPERTY, "Could not read class label file: " + class_label_file);
    }

    vector <string> class_labels;
    class_labels.reserve(class_label_count);
    string line;
    while (getline(fp, line)) {
        if (line.length()) { class_labels.push_back(line); }
    }
    fp.close();

    if (class_labels.size() != class_label_count) {
        THROW_TRTISEXCEPTION(MPF_COULD_NOT_READ_DATAFILE,
                             "Read class label count of "
                             + to_string(class_labels.size()) + " in file '"
                             + class_label_file
                             + "' does not match expected count of "
                             + to_string(class_label_count));
    } else {
        _class_labels[model] = class_labels;
    }
}


/** ****************************************************************************
* Convert image colorspace to RGB and turn to bytes blob in prep for
* inference server
*
* \param      img    OpenCV image to prep for inferencing
* \param[out] shape  Shape of the scaled output data
*
* \returns A continuous RGB data vector ready for inference server
***************************************************************************** */
BytVec TrtisDetection::_cvRGBBytes(const cv::Mat &img, LngVec &shape) {
    cv::Mat rgbImg;
    if (img.channels() == 3) {
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);
        LOG4CXX_TRACE(_log, "Converted 3 channel BGR image to RGB");
    } else if (img.channels() == 4) {
        cv::cvtColor(img, rgbImg, cv::COLOR_BGRA2RGB, 3);
        LOG4CXX_TRACE(_log, "Converted 4 channel BGRA image to RGB");
    } else if (img.channels() == 1) {
        cv::cvtColor(img, rgbImg, cv::COLOR_GRAY2RGB);
        LOG4CXX_TRACE(_log, "Converted 1 channel GRAY image to RGB");
    } else {
        THROW_TRTISEXCEPTION(MPF_DETECTION_FAILED, "Image could not be converted to RGB.");
    }

    if (rgbImg.type() != CV_8UC3) {
        rgbImg.convertTo(rgbImg, CV_8UC3);
        LOG4CXX_TRACE(_log, "Converted Image to CV_8U precision");
    }

    // return continuous chunk of image data in a byte vector
    BytVec data;
    size_t img_byte_size = rgbImg.total() * rgbImg.elemSize();
    data.resize(img_byte_size);

    if (rgbImg.isContinuous()) {
        memcpy(&(data[0]), rgbImg.datastart, img_byte_size);
    } else {
        LOG4CXX_TRACE(_log, "Converting image to have continuous data allocation");
        size_t pos = 0;
        size_t row_byte_size = rgbImg.cols * rgbImg.elemSize();
        for (int r = 0; r < rgbImg.rows; ++r) {
            memcpy(&(data[pos]), rgbImg.ptr<uint8_t>(r), row_byte_size);
            pos += row_byte_size;
        }
    }
    shape = {rgbImg.rows, rgbImg.cols, 3};
    return data;
}


/** ****************************************************************************
* Scale image dimensions to ~ [target_width,target_height]
* shorter side equal target_height pixel but keep longer side below
* target_width pixels and preserve aspect ratio.
*
* \param      img            OpenCV image to be resized to fit model
* \param[out] scaleFactor    scaling factor that was used to get to target dims
* \param      target_width   target width
* \param      target_height  target height
*
* \returns A scaled OpenCV image
***************************************************************************** */
cv::Mat TrtisDetection::_cvResize(const cv::Mat &img,
                                  double &scaleFactor,
                                  const int target_width,
                                  const int target_height) {

    if (img.cols > img.rows) {                          // landscape image
        scaleFactor = (float) target_height / img.rows;
        if (scaleFactor * img.cols > target_width) {
            scaleFactor = (float) target_width / img.cols;
        }
    } else {                                             // portrait image
        scaleFactor = (float) target_height / img.cols;
        if (scaleFactor * img.rows > target_width) {
            scaleFactor = (float) target_width / img.rows;
        }
    }
    cv::Mat scaledImg;
    cv::resize(img, scaledImg, cv::Size(), scaleFactor, scaleFactor, cv::INTER_CUBIC);
    LOG4CXX_TRACE(_log, "Scaled image by factor " << scaleFactor << " from [" << img.cols << "," << img.rows << "] to ["
                                                  << scaledImg.cols << "," << scaledImg.rows << "]");
    return scaledImg;
}

/** ****************************************************************************
* Scale image colorspace and dimensions and prep for inference server
*
* \param      cfg     configuration settings
* \param      img     OpenCV image to prep for inferencing
* \param      ctx     shared pointer to an inference context
* \param[out] shape   shape of the imgDat tensor
* \param[out] imgDat  image tensor data
*
* \note shape and imgDat need to persist till inference call
***************************************************************************** */
void TrtisDetection::_ip_irv2_coco_prepImageData(
        const TrtisIpIrv2CocoJobConfig &cfg,
        const cv::Mat &img,
        const sPtrInferCtx &ctx,
        LngVec &shape,
        BytVec &imgDat) {
    double scaleFactor = 1.0;
    LOG4CXX_TRACE(_log, "Preparing image data for inferencing");
    if (cfg.clientScaleEnabled) {
        imgDat = _cvRGBBytes(_cvResize(img, scaleFactor, 1024, 600), shape);
        LOG4CXX_TRACE(_log, "using client side image scaling");
    } else {
        imgDat = _cvRGBBytes(img, shape);
        LOG4CXX_TRACE(_log, "using TRTIS model's image scaling");
    }

    // Initialize the inputs with the data.
    sPtrInferCtxInp inImgDat, inBBox;
    NI_CHECK_OK(ctx->GetInput("image_input", &inImgDat),
                "unable to get image_input");
    NI_CHECK_OK(ctx->GetInput("bbox_input", &inBBox),
                "unable to get bbox_input");
    NI_CHECK_OK(inImgDat->Reset(),
                "unable to reset image_input");
    NI_CHECK_OK(inBBox->Reset(),
                "unable to reset bbox_input");
    NI_CHECK_OK(inImgDat->SetShape(shape),
                "failed setting image_input shape");
    NI_CHECK_OK(inImgDat->SetRaw(imgDat),
                "failed setting image_input");
    NI_CHECK_OK(inBBox->SetRaw((uint8_t * )(&(cfg.userBBoxNorm[0])),
                               cfg.userBBoxNorm.size() * sizeof(float)),
                "failed setting bbox_input");
    LOG4CXX_TRACE(_log, "Prepped data for inferencing");
}

/** ****************************************************************************
* Create an inference context for a model.
*
* \param cfg  job configuration settings containing TRTIS server info
*
* \returns shared pointer to an inferencing context
*          to use for inferencing requests
***************************************************************************** */
sPtrInferCtx TrtisDetection::_niGetInferContext(const TrtisJobConfig &cfg, int ctxId) {
    uPtrInferCtx ctx;
    NI_CHECK_OK(nic::InferGrpcContext::Create(&ctx, ctxId, cfg.trtis_server, cfg.model_name, cfg.model_version),
                "unable to create TRTIS inference context for \"" + cfg.trtis_server + "\"");

    // Configure context for 'batch_size'=1  and return all outputs
    uPtrInferCtxOpt options;
    NI_CHECK_OK(nic::InferContext::Options::Create(&options),
                "failed initializing TRTIS inference options");
    options->SetBatchSize(1);
    for (const auto &output : ctx->Outputs()) {
        options->AddRawResult(output);
    }
    NI_CHECK_OK(ctx->SetRunOptions(*options),
                "failed initializing TRTIS batch size and outputs");

    LOG4CXX_TRACE(_log, "Created context[" << ctx->CorrelationId() << "]");

    return move(ctx);
}

/** ****************************************************************************
* Create inference contexts for a model.
*
* \param cfg  job configuration settings containing TRTIS server info
*
* \returns map of context ids to shared pointers to inferencing contexts
***************************************************************************** */
unordered_map<int, sPtrInferCtx> TrtisDetection::_niGetInferContexts(const TrtisJobConfig &cfg) {
    unordered_map<int, sPtrInferCtx> ctxMap;

    nic::Error err;
    for (int i = 0; i < cfg.maxInferConcurrency; i++) {
        ctxMap[i] = move(_niGetInferContext(cfg, i));
    }

    return ctxMap;
}

/** ****************************************************************************
* convert ni::DataType enum to string for logging
*
* \param dt  NVIDIA DataType enum
*
* \returns string descriptor of enum value
***************************************************************************** */
string TrtisDetection::_niType2Str(ni::DataType dt) {
    switch (dt) {
        case ni::TYPE_INVALID:
            return "INVALID";
        case ni::TYPE_BOOL:
            return "BOOL";
        case ni::TYPE_UINT8:
            return "UINT8";
        case ni::TYPE_UINT16:
            return "UINT16";
        case ni::TYPE_UINT32:
            return "UINT32";
        case ni::TYPE_UINT64:
            return "UINT64";
        case ni::TYPE_INT8:
            return "INT8";
        case ni::TYPE_INT16:
            return "INT16";
        case ni::TYPE_INT32:
            return "INT32";
        case ni::TYPE_INT64:
            return "INT64";
        case ni::TYPE_FP16:
            return "FP16";
        case ni::TYPE_FP32:
            return "FP32";
        case ni::TYPE_FP64:
            return "FP64";
        case ni::TYPE_STRING:
            return "STRING";
        default:
            return "UNKNOWN";
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
* \returns an openCV matrix corresponding to the tensor
***************************************************************************** */
cv::Mat TrtisDetection::_niResult2CVMat(const int batch_idx,
                                        const string &name,
                                        StrUPtrInferCtxResMap &results) {

    // get raw data pointer and size
    const uint8_t *ptrRaw;
    size_t cntRaw;
    uPtrInferCtxRes *res = &results.at(name);
    NI_CHECK_OK((*res)->GetRaw(batch_idx, &ptrRaw, &cntRaw),
                "Failed to get inference server result raw data");

    // get raw data shape
    LngVec shape;
    NI_CHECK_OK((*res)->GetRawShape(&shape),
                "Failed to get inference server result shape");
    size_t ndim = shape.size();
    if (ndim < 2) { // force matrix for vector with single col?!
        ndim = 2;
        shape.push_back(1);
    }

    // calculate num elements from shape
    IntVec iShape;
    int64 numElementsFromShape = 1;
    for (const auto &d: shape) {
        numElementsFromShape *= d;
        iShape.push_back((int) d);
    }

    // determine opencv type and calculate num elements from raw size and data type
    size_t cvType;
    size_t sizeofEl;
    ni::DataType niType = (*res)->GetOutput()->DType();
    switch (niType) {
        case ni::TYPE_UINT8:
            cvType = CV_8UC(ndim - 1);
            sizeofEl = sizeof(uint8_t);
            break;
        case ni::TYPE_UINT16:
            cvType = CV_16UC(ndim - 1);
            sizeofEl = sizeof(uint16_t);
            break;
        case ni::TYPE_INT8:
            cvType = CV_8SC(ndim - 1);
            sizeofEl = sizeof(int8_t);
            break;
        case ni::TYPE_INT16:
            cvType = CV_16SC(ndim - 1);
            sizeofEl = sizeof(int16_t);
            break;
        case ni::TYPE_INT32:
            cvType = CV_32SC(ndim - 1);
            sizeofEl = sizeof(int32_t);
            break;
        case ni::TYPE_FP32:
            cvType = CV_32FC(ndim - 1);
            sizeofEl = sizeof(float);
            break;
        case ni::TYPE_FP64:
            cvType = CV_64FC(ndim - 1);
            sizeofEl = sizeof(double);
            break;
            // OpenCV does not support these types ?!
        case ni::TYPE_UINT32: //cvType = CV_32UC(ndim-1); sizeofEl=sizeof(uint32_t);  break;
        case ni::TYPE_UINT64: //cvType = CV_64UC(ndim-1); sizeofEl=sizeof(uint64_t);  break;
        case ni::TYPE_INT64:  //cvType = CV_64SC(ndim-1); sizeofEl=sizeof(int64_t);   break;
        case ni::TYPE_FP16:   //cvType = CV_16FC(ndim-1); sizeofEl=sizeof(float16_t); break;
        case ni::TYPE_BOOL:
        case ni::TYPE_STRING:
        case ni::TYPE_INVALID:
        default: THROW_TRTISEXCEPTION(MPF_DETECTION_FAILED,
                                      "Unsupported data_type " + _niType2Str(niType) + " in cv:Mat conversion");
    }

    if (cntRaw / sizeofEl == numElementsFromShape) {
        return cv::Mat(ndim, iShape.data(), cvType, (void *) ptrRaw);
    } else {
        stringstream ss("Shape ");
        ss << shape << " and data-type " << _niType2Str(niType) << "are inconsistent with buffer size " << cntRaw;
        THROW_TRTISEXCEPTION(MPF_DETECTION_FAILED, ss.str());
    }
}

/** ****************************************************************************
* Initialize the Trtis detector module by setting up paths and reading configs
*
* \returns true on success
***************************************************************************** */
bool TrtisDetection::Init() {

    // Determine where the executable is running
    string run_dir = GetRunDirectory();
    if (run_dir.empty()) { run_dir = "."; }

    string plugin_path = run_dir + "/TrtisDetection";
    string models_path = plugin_path + "/models";

    // Configure logger
    _log = log4cxx::Logger::getLogger("TrtisDetection");

    try {
        // Read class labels for model(s)
        _readClassNames("ip_irv2_coco", models_path + "/ip_irv2_coco/ip_irv2_coco.labels", 90);
    } catch (const exception &ex) {
        LOG4CXX_ERROR(_log, "Init failed: " << ex.what());
        return false;
    }
    LOG4CXX_DEBUG(_log, "Plugin path: " << plugin_path);

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
*       The user can dictate a bounding box to specify an custom object
*       location for which a feature vector will be returned.  Additionally
*       'extra' detection bounding boxes that did not readily classify to a
*       COCO class can also be returned. Finally a global image/frame feature
*       is provided that consists of a size weighted average of the detection
*       features. This can be useful in similarity searches or possibly
*       scene segmentation.
***************************************************************************** */
void TrtisDetection::_ip_irv2_coco_getDetections(
        const TrtisIpIrv2CocoJobConfig &cfg,
        StrUPtrInferCtxResMap &res,
        MPFImageLocationVec &locations) {

    if (cfg.frameFeatEnabled) {
        LOG4CXX_TRACE(_log, "processing global feature");
        cv::Mat g_feat = _niResult2CVMat(0, "G_Feat", res);
        cv::normalize(g_feat, g_feat, 1, 0, cv::NORM_L2);
        LOG4CXX_TRACE(_log, "normalized global feature");
        locations.push_back(MPFImageLocation(
                0, 0, cfg.image_x_max, cfg.image_y_max, -1.0,
                {{"FEATURE TYPE", "FRAME"},
                 {"FEATURE",      string((const char *) g_feat.data,
                                         g_feat.total() * g_feat.elemSize())}}));
        LOG4CXX_TRACE(_log, "added global feature to locations");
    }

    if (cfg.userFeatEnabled) {
        LOG4CXX_TRACE(_log, "processing user bbox specified feature");
        cv::Mat m_feat = _niResult2CVMat(0, "M_Feat", res);
        cv::normalize(m_feat, m_feat, 1, 0, cv::NORM_L2);
        LOG4CXX_TRACE(_log, "normalized user bbox specified feature");
        locations.push_back(MPFImageLocation(
                cfg.userBBox_x, cfg.userBBox_y, cfg.userBBox_width, cfg.userBBox_height,
                -1.0,
                {{"FEATURE TYPE", "USER"},
                 {"FEATURE",      string((const char *) m_feat.data,
                                         m_feat.total() * m_feat.elemSize())}}));
        LOG4CXX_TRACE(_log, "added user bbox feature to locations");
    }

    if (cfg.classFeatEnabled) {
        LOG4CXX_TRACE(_log, "processing detected object features with classifications");
        cv::Mat d_bboxes = _niResult2CVMat(0, "D_BBoxes", res);
        cv::Mat d_classes = _niResult2CVMat(0, "D_Classes", res);
        cv::Mat d_scores = _niResult2CVMat(0, "D_Scores", res);
        cv::Mat d_feats = _niResult2CVMat(0, "D_Feats", res);
        auto classes = _class_labels["ip_irv2_coco"];
        for (int r = 0; r < d_feats.rows; r++) {
            float conf = d_scores.at<float>(r, 0);
            if (conf >= cfg.classConfThreshold) {
                cv::normalize(d_feats.row(r), d_feats.row(r), 1, 0, cv::NORM_L2);
                int class_id = ((int) d_classes.at<float>(r, 0));
                string label = (class_id <= 90) ? classes[class_id - 1] : "unknown";
                locations.push_back(MPFImageLocation(
                        d_bboxes.at<float>(r, 1) * (cfg.image_x_max), d_bboxes.at<float>(r, 0) * (cfg.image_y_max),
                        (d_bboxes.at<float>(r, 3) - d_bboxes.at<float>(r, 1)) * (cfg.image_x_max),
                        (d_bboxes.at<float>(r, 2) - d_bboxes.at<float>(r, 0)) * (cfg.image_y_max), conf,
                        {{"FEATURE TYPE", "CLASS"},
                         {"FEATURE", string((const char *) d_feats.ptr(r),
                                            d_feats.cols * d_feats.elemSize())},
                         {"CLASSIFICATION", label}}));
                LOG4CXX_TRACE(_log, "detected object with class[" << class_id << "] = " << label << " with confidence("
                                                                  << locations.back().confidence << ")" << " at ["
                                                                  << locations.back().x_left_upper << ","
                                                                  << locations.back().y_left_upper << ","
                                                                  << locations.back().width << ","
                                                                  << locations.back().height
                                                                  << "]");
            }
        }
        LOG4CXX_TRACE(_log, "added detected object features to locations");
    }

    if (cfg.extraFeatEnabled) {
        LOG4CXX_TRACE(_log, "processing extra detected objects without classifications");
        cv::Mat e_bboxes = _niResult2CVMat(0, "E_BBoxes", res);
        cv::Mat e_scores = _niResult2CVMat(0, "E_Scores", res);
        cv::Mat e_feats = _niResult2CVMat(0, "E_Feats", res);
        for (int r = 0; r < e_feats.rows; r++) {
            float conf = e_scores.at<float>(r, 0);
            if (conf >= cfg.extraConfThreshold) {
                cv::normalize(e_feats.row(r), e_feats.row(r), 1, 0, cv::NORM_L2);
                locations.push_back(MPFImageLocation(
                        e_bboxes.at<float>(r, 1) * cfg.image_x_max, e_bboxes.at<float>(r, 0) * cfg.image_y_max,
                        (e_bboxes.at<float>(r, 3) - e_bboxes.at<float>(r, 1)) * cfg.image_x_max,
                        (e_bboxes.at<float>(r, 2) - e_bboxes.at<float>(r, 0)) * cfg.image_y_max, conf,
                        {{"FEATURE TYPE", "EXTRA"},
                         {"FEATURE",      string((const char *) e_feats.ptr(r),
                                                 e_feats.cols * e_feats.elemSize())}}));
                LOG4CXX_TRACE(_log,
                              "detected extra object with confidence(" << locations.back().confidence << ")" << " at ["
                                                                       << locations.back().x_left_upper << ","
                                                                       << locations.back().y_left_upper << ","
                                                                       << locations.back().width << ","
                                                                       << locations.back().height << "]");
            }
        }
        LOG4CXX_TRACE(_log, "added detected extra features to locations");
    }

    if (cfg.recognitionEnroll) {
        // _recognitionFrameworkEnroll(job,locations);
        // * needs to be implemented
        THROW_TRTISEXCEPTION(MPF_INVALID_PROPERTY,
                             "Recognition Framework Enroll Interface not implemented");
    }

}

/** ****************************************************************************
* Compute a similarity score based on Cos / Inner Product Distance
*
* \param p1    1st feature vector to be used in distance calc.
* \param p2    2nd feature vector to be used in distance calc.
* \param size  number of floats in each of the two vectors
*
* \returns cos similarity of the two vectors (-1...1)
*
* \note vectors are assumed to be normalized to magnitude of 1.0
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
* \returns pixel Euclidean distance squared between centers
***************************************************************************** */
static float centerDistSq(const MPFImageLocation &l1, const MPFImageLocation &l2) {
    float dx = l1.x_left_upper - l2.x_left_upper + (float) (l1.width - l2.width) / 2;
    float dy = l1.y_left_upper - l2.y_left_upper + (float) (l1.height - l2.height) / 2;
    return dx * dx + dy * dy;
}

/** ****************************************************************************
* Adds a location to an object track and updates the track's classification
*  and confidence if they exist
*
* \param         location  location to add to track
* \param         frameIdx  frame index of location
* \param[in,out] track     track to which location will be appended
***************************************************************************** */
void TrtisDetection::_addToTrack(MPFImageLocation &location,
                                 int frameIdx,
                                 MPFVideoTrack &track) {
    track.stop_frame = frameIdx;
    if (location.confidence > track.confidence) {
        track.confidence = location.confidence;
        auto it = location.detection_properties.find("CLASSIFICATION");
        if (it != location.detection_properties.end()) {
            track.detection_properties["FEATURE TYPE"] = "CLASS";
            track.detection_properties["CLASSIFICATION"] = it->second;
            LOG4CXX_TRACE(_log, "updating track class to " << it->second);
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
***************************************************************************** */
void TrtisDetection::_ip_irv2_coco_tracker(
        const TrtisIpIrv2CocoJobConfig &cfg,
        MPFImageLocation &loc,
        const int frameIdx,
        MPFVideoTrackVec &tracks) {

    MPFVideoTrack *bestTrackPtr = nullptr;
    float minFeatureGap = FLT_MAX;
    for (MPFVideoTrack &track : tracks) {
        int frameGap = frameIdx - track.stop_frame;
        if (frameGap > 0 && frameGap <= cfg.maxFrameGap) {
            MPFImageLocation stopLoc = track.frame_locations[track.stop_frame];
            float spaceGapPxSq = centerDistSq(stopLoc, loc);
            if (spaceGapPxSq <= cfg.maxSpaceGapPxSq) {
                auto trkFeat = stopLoc.detection_properties.find("FEATURE");
                auto locFeat = loc.detection_properties.find("FEATURE");
                if (trkFeat != stopLoc.detection_properties.end()
                    && locFeat != loc.detection_properties.end()) {
                    auto stopFeature = (float *) (&(trkFeat->second[0]));
                    auto locFeature = (float *) (&(locFeat->second[0]));
                    float featureGap = 1.0f - ipSimilarity(stopFeature, locFeature, 1088);
                    if (featureGap <= cfg.maxFeatureGap) {
                        LOG4CXX_TRACE(_log, "featureGap = " << featureGap << " < " << cfg.maxFeatureGap);
                        if (featureGap < minFeatureGap) {
                            LOG4CXX_TRACE(_log, "bestTrack = " << &track);
                            bestTrackPtr = &track;
                            minFeatureGap = featureGap;
                        }
                    }
                }
            }
        }
    }

    if (bestTrackPtr != nullptr) {
        LOG4CXX_TRACE(_log, "Adding to track(&" << bestTrackPtr << ") from frame[" << frameIdx << "]");
        int bestStopFrame = bestTrackPtr->stop_frame;
        MPFImageLocation bestStopLoc = bestTrackPtr->frame_locations[bestStopFrame];
        if (bestStopFrame != bestTrackPtr->start_frame) {
            bestStopLoc.detection_properties.erase("FEATURE");
            LOG4CXX_TRACE(_log,
                          "Erased previous FEATURE for track(& " << bestTrackPtr << ") from frame[" << frameIdx
                                                                 << "]");
        } else { // keep start frame feature for linking ?!
            bestStopLoc.detection_properties["FEATURE"] =
                    Base64::Encode(bestStopLoc.detection_properties["FEATURE"]);
            LOG4CXX_TRACE(_log, "Re-encoded track start feature");
        }
        _addToTrack(loc, frameIdx, *bestTrackPtr);
    } else {
        // track is EXTRA until first CLASS detection is added
        MPFVideoTrack newTrack(frameIdx, frameIdx, -1, { {"FEATURE TYPE", "EXTRA"} });
        tracks.push_back(move(newTrack));
        LOG4CXX_TRACE(_log, "Created new track(&" << &(tracks.back()) << ") from frame[" << frameIdx << "]");
        _addToTrack(loc, frameIdx, tracks.back());
    }
}

/** ****************************************************************************
* Read frames from a video, get object detections and make tracks
*
* \param job  MPF Video job
*
* \throws MPFDetectionException
*
* \returns Tracks collection to which detections will be added
***************************************************************************** */
vector <MPFVideoTrack> TrtisDetection::GetDetections(const MPFVideoJob &job) {
    try {
        LOG4CXX_INFO(_log, "Starting job");

        // if (job.has_feed_forward_track) { do something different ?!}

        MPFVideoCapture video_cap(job);

        vector <MPFVideoTrack> class_extra_tracks;
        MPFVideoTrack frame_track(0, 0, -1, { {"FEATURE TYPE", "FRAME"} });
        MPFVideoTrack user_track(0, 0, -1, { {"FEATURE TYPE", "USER"} });

        cv::Mat frame;
        Properties jpr = job.job_properties;
        string model_name = get<string>(jpr, "MODEL_NAME", "ip_irv2_coco");

        if (model_name != "ip_irv2_coco") {
            THROW_TRTISEXCEPTION(MPF_INVALID_PROPERTY, "Unsupported model type:" + model_name);
        }

        // need to read a frame to get size
        if (!video_cap.Read(frame)) return {};
        TrtisIpIrv2CocoJobConfig cfg(job, _log, frame.cols, frame.rows);

        // frames per milli-sec if available
        double fp_ms = get<double>(job.media_properties, "FPS", 0.0) / 1000.0;

        unordered_map<int, sPtrInferCtx> ctxMap = _niGetInferContexts(cfg);
        size_t initialCtxPoolSize = ctxMap.size();
        LOG4CXX_TRACE(_log, "Retrieved inferencing context pool of size " << initialCtxPoolSize << " for model '"
                                                                          << cfg.model_name << "' from server "
                                                                          << cfg.trtis_server);

        unordered_set<int> freeCtxPool;
        for (const auto &pair : ctxMap) {
            freeCtxPool.insert(pair.first);
        }

        mutex freeCtxMtx, nextRxFrameMtx, tracksMtx, errorMtx;
        condition_variable freeCtxCv, nextRxFrameCv;

        exception_ptr eptr; // first exception thrown

        try {
            LOG4CXX_TRACE(_log, "Main thread_id:" << this_thread::get_id());

            int frameIdx = 0;
            int nextRxFrameIdx = 0;

            do {
                // Wait for an available inference context.
                LOG4CXX_TRACE(_log, "requesting inference from TRTIS server for frame[" << frameIdx << "]");

                int ctxId;
                {
                    unique_lock <mutex> lk(freeCtxMtx);
                    if (freeCtxPool.empty()) {
                        LOG4CXX_TRACE(_log, "wait for an infer context to become available");
                        freeCtxCv.wait(lk, [&freeCtxPool] { return !freeCtxPool.empty(); });
                    }
                    {
                        lock_guard <mutex> lk(errorMtx);
                        if (eptr) {
                            break; // stop processing frames
                        }
                    }
                    auto it = freeCtxPool.begin();
                    ctxId = *it;
                    freeCtxPool.erase(it);
                    LOG4CXX_TRACE(_log, "removing context[" << ctxId << "] from pool");
                }

                sPtrInferCtx &ctx = ctxMap[ctxId];

                LngVec shape;
                BytVec imgDat;
                _ip_irv2_coco_prepImageData(cfg, frame, ctx, shape, imgDat);
                LOG4CXX_TRACE(_log, "Loaded data into inference context");
                LOG4CXX_DEBUG(_log, "frame[" << frameIdx << "] sending");

                // Send inference request to the inference server.
                NI_CHECK_OK(
                        ctx->AsyncRun([frameIdx, &job, &cfg, &freeCtxCv, &nextRxFrameCv,
                                              &nextRxFrameMtx, &tracksMtx, &freeCtxMtx, &errorMtx,
                                              &nextRxFrameIdx, &freeCtxPool, &eptr,
                                              &class_extra_tracks, &frame_track, &user_track,
                                              this](nic::InferContext *c, sPtrInferCtxReq req) {
                            // NOTE: When this callback is invoked, the frame has already been processed by the TRTIS server.
                            LOG4CXX_DEBUG(_log, "Async run callback for frame[" << frameIdx << "] with context["
                                                                                << c->CorrelationId()
                                                                                << "] and thread_id:"
                                                                                << this_thread::get_id());

                            // Ensure tracking is performed on the frames in the proper order.
                            {
                                unique_lock <mutex> lk(nextRxFrameMtx);
                                if (frameIdx != nextRxFrameIdx) {
                                    LOG4CXX_TRACE(_log, ">> Out of sequence frame response, waiting to process frame["
                                            << frameIdx
                                            << "], but nextRxFrameIdx["
                                            << nextRxFrameIdx
                                            << "]");
                                    nextRxFrameCv.wait(lk,
                                                       [frameIdx, &nextRxFrameIdx] {
                                                           return frameIdx == nextRxFrameIdx;
                                                       });
                                }
                            }

                            // Retrieve the results from the TRTIS server and update tracks.
                            try {
                                StrUPtrInferCtxResMap res;
                                bool is_ready = false;
                                NI_CHECK_OK(
                                        c->GetAsyncRunResults(&res, &is_ready, req, true),
                                        "Failed to retrieve inference results for context " +
                                        to_string(c->CorrelationId()));
                                if (!is_ready) {
                                    THROW_TRTISEXCEPTION(MPF_DETECTION_FAILED,
                                                         "Inference results not ready during callback for context " +
                                                         to_string(c->CorrelationId()));
                                }

                                LOG4CXX_TRACE(_log, "inference complete");
                                MPFImageLocationVec locations;
                                _ip_irv2_coco_getDetections(cfg, res, locations);
                                LOG4CXX_TRACE(_log, "inferenced frame[" << frameIdx << "]");
                                {
                                    lock_guard <mutex> lk(tracksMtx);
                                    for (MPFImageLocation &loc : locations) {
                                        const string feature_type = loc.detection_properties["FEATURE TYPE"];
                                        LOG4CXX_TRACE(_log, "Found detection with feature_type:" << feature_type);
                                        if (feature_type == "CLASS" || feature_type == "EXTRA") {
                                            _ip_irv2_coco_tracker(cfg, loc, frameIdx, class_extra_tracks);
                                        } else if (feature_type == "FRAME") {
                                            frame_track.stop_frame = frameIdx;
                                            frame_track.frame_locations[frameIdx] = loc;
                                        } else if (feature_type == "USER") {
                                            user_track.stop_frame = frameIdx;
                                            user_track.frame_locations[frameIdx] = loc;
                                        }
                                    }
                                    LOG4CXX_TRACE(_log, "tracked objects in frame[" << frameIdx << "]");
                                }
                            } catch (...) {
                                try {
                                    Utils::LogAndReThrowException(job, _log);
                                } catch (MPFDetectionException &e) {
                                    {
                                        lock_guard <mutex> lk(errorMtx);
                                        if (!eptr) {
                                            eptr = current_exception();
                                        }
                                    }
                                }
                            }

                            // Tracking for this frame is complete. Allow tracking on the next frame.
                            {
                                lock_guard <mutex> lk(nextRxFrameMtx);
                                nextRxFrameIdx++;
                                LOG4CXX_TRACE(_log, "nextRxFrameIdx++ to " << nextRxFrameIdx);
                                nextRxFrameCv.notify_all();
                            }

                            // We're done with the context. Add it back to the pool so it can be used for another frame.
                            {
                                lock_guard <mutex> lk(freeCtxMtx);
                                freeCtxPool.insert(c->CorrelationId());
                                freeCtxCv.notify_all();
                                LOG4CXX_TRACE(_log, "returned context[" << c->CorrelationId() << "] to pool");
                            }
                            LOG4CXX_DEBUG(_log, "frame[" << frameIdx << "] complete");
                        }),
                        "unable to inference '" + cfg.model_name + "' ver."
                        + to_string(cfg.model_version));
                LOG4CXX_DEBUG(_log, "Inference request sent for frame[" << frameIdx << "] sent");

                frameIdx++;
                LOG4CXX_TRACE(_log, "frameIdx++ to " << frameIdx);
            } while (video_cap.Read(frame));

        } catch (...) {
            try {
                Utils::LogAndReThrowException(job, _log);
            } catch (MPFDetectionException &e) {
                {
                    lock_guard <mutex> lk(errorMtx);
                    if (!eptr) {
                        eptr = current_exception();
                    }
                }
            }
        }

        // Always wait for async threads to complete.
        if (freeCtxPool.size() < initialCtxPoolSize) {
            LOG4CXX_TRACE(_log,
                          "wait for inference context pool size to return to initial size of " << initialCtxPoolSize);
            unique_lock <mutex> lk(freeCtxMtx);
            freeCtxCv.wait(lk,
                           [&freeCtxPool, &initialCtxPoolSize] { return freeCtxPool.size() == initialCtxPoolSize; });
        }

        // Abort now if an error occurred.
        if (eptr) {
            LOG4CXX_ERROR(_log, "An error occurred. Aborting job.")
            rethrow_exception(eptr);
        }

        LOG4CXX_DEBUG(_log, "all frames complete");

        vector <MPFVideoTrack> tracks = move(class_extra_tracks);
        if (!frame_track.frame_locations.empty()) {
            tracks.push_back(move(frame_track));
        }
        if (!user_track.frame_locations.empty()) {
            tracks.push_back(move(user_track));
        }

        for (MPFVideoTrack &track : tracks) {
            video_cap.ReverseTransform(track);
        }

        // Only record features (potentially to S3) if the job is otherwise successful.
        for (auto &track : tracks) {
            for (auto &pair : track.frame_locations) {
                cfg.featureStorage->Store(cfg.data_uri, cfg.model_name, track, pair.second, fp_ms);
            }
        }

        LOG4CXX_INFO(_log, "Found " << tracks.size() << " tracks.");

        return tracks;
    } catch (...) {
        Utils::LogAndReThrowException(job, _log);
    }
}

/** ****************************************************************************
* Read an image and get object detections and features
*
* \param job  MPF Image job
*
* \throws MPFDetectionException
*
* \returns locations collection to which detections will be added
*
* \note prior to returning the features are base64 encoded
*
***************************************************************************** */
vector <MPFImageLocation> TrtisDetection::GetDetections(const MPFImageJob &job) {
    try {
        LOG4CXX_INFO(_log, "Starting job");
        LOG4CXX_DEBUG(_log, "Data URI = " << job.data_uri);

        vector <MPFImageLocation> locations;
        MPFImageReader image_reader(job);
        cv::Mat img = image_reader.GetImage();

        Properties jpr = job.job_properties;
        string model_name = get<string>(jpr, "MODEL_NAME", "ip_irv2_coco");

        if (model_name != "ip_irv2_coco") {
            THROW_TRTISEXCEPTION(MPF_INVALID_PROPERTY, "Unsupported model type: " + model_name);
        }

        TrtisIpIrv2CocoJobConfig cfg(job, _log, img.cols, img.rows);
        LOG4CXX_TRACE(_log, "parsed job configuration settings");

        cfg.maxInferConcurrency = 1;
        sPtrInferCtx ctx = _niGetInferContext(cfg);
        LOG4CXX_TRACE(_log, "retrieved inferencing context for model '" << cfg.model_name << "' from server "
                                                                        << cfg.trtis_server);

        LngVec shape;
        BytVec imgDat;
        _ip_irv2_coco_prepImageData(cfg, img, ctx, shape, imgDat);
        LOG4CXX_TRACE(_log, "loaded data into inference context");

        // Send inference request to the inference server.
        StrUPtrInferCtxResMap res;
        NI_CHECK_OK(ctx->Run(&res), "unable to inference '" + cfg.model_name
                                    + "' ver." + to_string(cfg.model_version));
        LOG4CXX_TRACE(_log, "inference complete");

        size_t next_idx = locations.size();
        _ip_irv2_coco_getDetections(cfg, res, locations);
        LOG4CXX_TRACE(_log, "parsed detections into locations vector");

        for (auto &location : locations) {
            image_reader.ReverseTransform(location);
        }

        for (auto &location : locations) {
            cfg.featureStorage->Store(cfg.data_uri, cfg.model_name, location);
        }

        LOG4CXX_INFO(_log, "Found " << locations.size() << " detections.");

        return locations;
    } catch (...) {
        Utils::LogAndReThrowException(job, _log);
    }
}

//-----------------------------------------------------------------------------

MPF_COMPONENT_CREATOR(TrtisDetection);

MPF_COMPONENT_DELETER();
