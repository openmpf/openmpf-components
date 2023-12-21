/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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

#include "Config.h"

#include <detectionComponentUtils.h>

#include "util.h"

using namespace MPF::COMPONENT;

using DetectionComponentUtils::GetProperty;

const log4cxx::LoggerPtr Config::log = log4cxx::Logger::getLogger("OcvYoloDetection");


namespace {
    cv::Mat1f loadCovarianceMat(const std::string &serializedMat) {
        auto result = fromString(serializedMat, 4, 1, "f");
        //convert stdev to variances
        return result.mul(result);
    }

    std::string toLower(std::string str) {
        std::transform(str.begin(), str.end(), str.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        return str;
    }
}

Config::Config(const Properties &jobProps)
        : confidenceThreshold(std::max(GetProperty(jobProps, "CONFIDENCE_THRESHOLD", 0.5), 0.0))
        , nmsThresh(GetProperty(jobProps, "DETECTION_NMS_THRESHOLD", 0.3))
        , numClassPerRegion(GetProperty(jobProps, "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5))
        , netInputImageSize(GetProperty(jobProps, "NET_INPUT_IMAGE_SIZE", 416))
        , frameBatchSize(GetProperty(jobProps, "DETECTION_FRAME_BATCH_SIZE", 16))
        , maxClassDist(GetProperty(jobProps, "TRACKING_MAX_CLASS_DIST", 0.99))
        , maxFeatureDist(GetProperty(jobProps, "TRACKING_MAX_FEATURE_DIST", 0.1))
        // TODO: Center-to-center distance is currently disabled. Expose it as optional behavior or remove it.
        , maxCenterDist(GetProperty(jobProps, "TRACKING_MAX_CENTER_DIST", 0.0))
        , maxFrameGap(GetProperty(jobProps, "TRACKING_MAX_FRAME_GAP", 4)
                        / GetProperty(jobProps, "FRAME_INTERVAL", 1))
        , maxIOUDist(GetProperty(jobProps, "TRACKING_MAX_IOU_DIST", 0.3))
        , edgeSnapDist(GetProperty(jobProps, "TRACKING_EDGE_SNAP_DIST", 0.005))
        , dftSize(GetProperty(jobProps, "TRACKING_DFT_SIZE", 128))
        , dftHannWindowEnabled(GetProperty(jobProps, "TRACKING_DFT_USE_HANNING_WINDOW", true))
        , mosseTrackerDisabled(GetProperty(jobProps, "TRACKING_DISABLE_MOSSE_TRACKER", true))
        , maxKFResidual(GetProperty(jobProps, "KF_MAX_ASSIGNMENT_RESIDUAL", 2.5))
        , kfDisabled(GetProperty(jobProps, "KF_DISABLED", false))
        , RN(loadCovarianceMat(
                GetProperty(jobProps, "KF_RN", "[ 10.0, 10.0, 100.0, 100.0 ]")))
        , QN(loadCovarianceMat(
                GetProperty(jobProps, "KF_QN", "[ 1000.0, 1000.0, 1000.0, 1000.0 ]")))
        , fallback2CpuWhenGpuProblem(GetProperty(jobProps, "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", false))
        , cudaDeviceId(GetProperty(jobProps, "CUDA_DEVICE_ID", -1))
        , classAllowListPath(GetProperty(jobProps, "CLASS_ALLOW_LIST_FILE", ""))
        , enableDebug(GetProperty(jobProps, "ENABLE_DEBUG", false))
        , tritonEnabled(GetProperty(jobProps, "ENABLE_TRITON", false))
        , tritonServer(GetProperty(jobProps, "TRITON_SERVER", "ocv-yolo-detection-server:8001"))
        , tritonModelName(toLower(GetProperty(jobProps, "MODEL_NAME", "tiny yolo")))
        , tritonModelVersion(GetProperty(jobProps, "TRITON_MODEL_VERSION", ""))
        , tritonNumClasses(GetProperty(jobProps, "TRITON_MODEL_NUM_CLASSES", 80))
        , tritonMaxInferConcurrency(GetProperty(jobProps, "TRITON_MAX_INFER_CONCURRENCY", 4))
        , tritonClientTimeout(GetProperty(jobProps, "TRITON_INFER_TIMEOUT_US", 0))
        , tritonMaxConnectionSetupRetries(GetProperty(jobProps, "TRITON_MAX_CONNECTION_SETUP_RETRIES", 5))
        , tritonConnectionSetupRetryInitialDelay(
                GetProperty(jobProps, "TRITON_CONNECTION_SETUP_RETRY_INITIAL_DELAY", 5))
        , tritonVerboseClient(GetProperty(jobProps, "TRITON_VERBOSE_CLIENT", false))
        , tritonUseSSL(GetProperty(jobProps, "TRITON_USE_SSL", false))
        , tritonUseShm(GetProperty(jobProps, "TRITON_USE_SHM", false)) {
}


/****************************************************************************
*   Dump config to a stream
****************************************************************************/
std::ostream &operator<<(std::ostream &out, const Config &cfg) {
    out << "{"
        << "\"confThresh\":" << cfg.confidenceThreshold << ","
        << "\"nmsThresh\":" << cfg.nmsThresh << ","
        << "\"frameBatchSize\":" << cfg.frameBatchSize << ","
        << "\"numClassPerRegion\":" << cfg.numClassPerRegion << ","
        << "\"maxClassDist\":" << cfg.maxClassDist << ","
        << "\"maxFeatureDist\":" << cfg.maxFeatureDist << ","
        << "\"maxFrameGap\":" << cfg.maxFrameGap << ","
        << "\"maxCenterDist\":" << cfg.maxCenterDist << ","
        << "\"maxIOUDist\":" << cfg.maxIOUDist << ","
        << "\"edgeSnapDist\":" << cfg.edgeSnapDist << ","
        << "\"dftSize\":" << cfg.dftSize << ","
        << "\"dftHannWindow\":" << (cfg.dftHannWindowEnabled ? "1" : "0") << ","
        << "\"maxKFResidual\":" << cfg.maxKFResidual << ","
        << "\"kfDisabled\":" << (cfg.kfDisabled ? "1" : "0") << ","
        << "\"mosseTrackerDisabled\":" << (cfg.mosseTrackerDisabled ? "1" : "0") << ","
        << "\"fallback2CpuWhenGpuProblem\":" << (cfg.fallback2CpuWhenGpuProblem ? "1" : "0") << ","
        << "\"cudaDeviceId\":" << cfg.cudaDeviceId << ","
        << "\"classAllowListPath\":" << cfg.classAllowListPath << ","
        << "\"enabledDebug\":" << cfg.enableDebug << ","
        << "\"tritonServer\":" << cfg.tritonModelVersion << ","
        << "\"tritonModelName\":" << cfg.tritonModelName << ","
        << "\"tritonModelVersion\":" << cfg.tritonModelVersion << ","
        << "\"tritonNumClasses\":" << cfg.tritonNumClasses << ","
        << "\"tritonMaxInferConcurrency\":" << cfg.tritonMaxInferConcurrency << ","
        << "\"tritonClientTimeout\":" << cfg.tritonClientTimeout << ","
        << "\"tritonMaxConnectionsSetupRetries\":" << cfg.tritonMaxConnectionSetupRetries << ","
        << "\"tritonConnectionSetupRetryInitialDelay\":" << cfg.tritonConnectionSetupRetryInitialDelay << ","
        << "\"tritonVerboseClient\":" << cfg.tritonVerboseClient << ","
        << "\"tritonUseSSL\":" << cfg.tritonUseSSL << ","
        << "\"tritonUseShm\":" << cfg.tritonUseShm << ","
        << "\"kfProcessVar\":" << format(cfg.QN) << ","
        << "\"kfMeasurementVar\":" << format(cfg.RN)
        << "}";
    return out;
}
