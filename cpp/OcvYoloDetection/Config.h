/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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

#ifndef OPENMPF_COMPONENTS_CONFIG_H
#define OPENMPF_COMPONENTS_CONFIG_H

#include <ostream>
#include <string>

#include <log4cxx/logger.h>
#include <opencv2/opencv.hpp>

#include <MPFDetectionObjects.h>



/** ****************************************************************************
* logging shorthand macros
****************************************************************************** */
#ifdef DEBUG_LINE_NUMBERS
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define LOG_PREFIX "." <<  std::setw(20) << std::string(__FUNCTION__).substr(0,20) << ":" << std::setw(4) << __LINE__ <<" - "
#else
#define LOG_PREFIX ""
#endif

#define LOG_TRACE(MSG){ LOG4CXX_TRACE(Config::log, LOG_PREFIX << MSG) }
#define LOG_DEBUG(MSG){ LOG4CXX_DEBUG(Config::log, LOG_PREFIX << MSG) }
#define LOG_INFO(MSG){ LOG4CXX_INFO (Config::log, LOG_PREFIX << MSG) }
#define LOG_WARN(MSG){ LOG4CXX_WARN (Config::log, LOG_PREFIX << MSG) }
#define LOG_ERROR(MSG){ LOG4CXX_ERROR(Config::log, LOG_PREFIX << MSG) }
#define LOG_FATAL(MSG){ LOG4CXX_FATAL(Config::log, LOG_PREFIX << MSG) }


class Config {
public:
    /// detection confidence threshold
    float confidenceThreshold;

    /// non-maximum suppression threshold to remove redundant bounding boxes
    float nmsThresh;

    /// number of class labels and confidence scores to return for a bbox
    int numClassPerRegion;

    int netInputImageSize;

    /// number of frames to batch inference when processing video
    int frameBatchSize;

    /// maximum class feature scores above which detections will not be considered for the same track
    float maxClassDist;

    /// maximum feature distance to maintain track continuity
    float maxFeatureDist;

    /// maximum spatial distance normalized by diagonal to maintain track continuity
    float maxCenterDist;

    /// maximum temporal distance (frames) to maintain track continuity
    long maxFrameGap;

    /// maximum for (1 - Intersection/Union) to maintain track continuity
    float maxIOUDist;

    /// distance as a fraction of image dimensions within which bboxes are snapped to frame edge
    float edgeSnapDist;

    /// size of dft used for bbox alignment
    int dftSize;

    /// use Hanning windowing with dft
    bool dftHannWindowEnabled;

    /// disable builtin OCV MOSSE tracking
    bool mosseTrackerDisabled;

    /// maximum residual for valid detection to track assignment
    float maxKFResidual;

    /// if true Kalman filtering is disabled
    bool kfDisabled;

    /// Kalman filter measurement noise matrix
    cv::Mat1f RN;

    /// Kalman filter process noise variances (i.e. unknown accelerations)
    cv::Mat1f QN;

    /// fallback to cpu if there is a gpu problem
    bool fallback2CpuWhenGpuProblem;

    /// gpu device id to use for cuda
    int cudaDeviceId;

    std::string classWhiteListPath;

    bool enableDebug;

    /// enable inference server use
    bool tritonEnabled;

    /// triton inference server to use
    std::string tritonServer;

    /// triton inference server model to use
    std::string tritonModelName;

    /// version of model (e.g. "" for latest)
    std::string tritonModelVersion;

    /// number of classes returned by model
    int tritonNumClasses;

    /// inference server maximum number of concurrent video frame inferencing request
    int tritonMaxInferConcurrency;

    /// inference server client request timeout in micro-seconds
    uint32_t tritonClientTimeout;

    /// max setup retries for inference server connection
    int tritonMaxConnectionSetupRetries;

    /// initial delay before attempting inference server connection again
    int tritonConnectionSetupRetryInitialDelay;

    /// verbose inference server client mode
    bool tritonVerboseClient;

    /// use ssl encryption with inference server client
    bool tritonUseSSL;

    /// use shared memory for client-server communication
    bool tritonUseShm;

    /// shared log object
    static const log4cxx::LoggerPtr log;


    explicit Config(const MPF::COMPONENT::Properties& jobProps);
};

/// Dump Config to a stream
std::ostream &operator<<(std::ostream &out, const Config &cfg);


#endif //OPENMPF_COMPONENTS_CONFIG_H
