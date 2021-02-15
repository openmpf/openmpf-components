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

#include "Config.h"

#include <detectionComponentUtils.h>

#include "util.h"

using namespace MPF::COMPONENT;

using DetectionComponentUtils::GetProperty;

// Shared static members (might need mutex locks and condition variable if multithreading... )
log4cxx::LoggerPtr Config::log = log4cxx::Logger::getRootLogger();


namespace {
    cv::Mat1f LoadCovarianceMat(const std::string &serializedMat) {
        auto result = fromString(serializedMat, 4, 1, "f");
        //convert stdev to variances
        return result.mul(result);
    }
}

Config::Config(const Properties &jobProps)
        : confidenceThreshold(GetProperty(jobProps, "CONFIDENCE_THRESHOLD", 0.5))
        , nmsThresh(GetProperty(jobProps, "DETECTION_NMS_THRESHOLD", 0.3))
        , numClassPerRegion(GetProperty(jobProps, "NUMBER_OF_CLASSIFICATIONS_PER_REGION", 5))
        , netInputImageSize(GetProperty(jobProps, "NET_INPUT_IMAGE_SIZE", 416))
        , frameBatchSize(GetProperty(jobProps, "DETECTION_FRAME_BATCH_SIZE", 16))
        , maxClassDist(GetProperty(jobProps, "TRACKING_MAX_CLASS_DIST", 0.99))
        , maxFeatureDist(GetProperty(jobProps, "TRACKING_MAX_FEATURE_DIST", 0.1))
        , maxCenterDist(GetProperty(jobProps, "TRACKING_MAX_CENTER_DIST", 0.0))
        , maxFrameGap(GetProperty(jobProps, "TRACKING_MAX_FRAME_GAP", 4))
        , maxIOUDist(GetProperty(jobProps, "TRACKING_MAX_IOU_DIST", 0.3))
        , edgeSnapDist(GetProperty(jobProps, "TRACKING_EDGE_SNAP_DIST", 0.005))
        , dftSize(GetProperty(jobProps, "TRACKING_DFT_SIZE", 256))
        , dftHannWindowEnabled(GetProperty(jobProps, "TRACKING_DFT_USE_HANNING_WINDOW", true))
        , mosseTrackerDisabled(GetProperty(jobProps, "TRACKING_DISABLE_MOSSE_TRACKER", false))
        , maxKFResidual(GetProperty(jobProps, "KF_MAX_ASSIGNMENT_RESIDUAL", 2.5))
        , kfDisabled(GetProperty(jobProps, "KF_DISABLED", false))
        , RN(LoadCovarianceMat(
                GetProperty(jobProps, "KF_RN", "[ 10.0, 10.0, 100.0, 100.0 ]")))
        , QN(LoadCovarianceMat(
                GetProperty(jobProps, "KF_QN", "[ 1000.0, 1000.0, 1000.0, 1000.0 ]")))
        , fallback2CpuWhenGpuProblem(
                GetProperty(jobProps, "FALLBACK_TO_CPU_WHEN_GPU_PROBLEM", false))
        , cudaDeviceId(GetProperty(jobProps, "CUDA_DEVICE_ID", -1))
        , classWhiteListPath(GetProperty(jobProps, "CLASS_WHITELIST_FILE", ""))
{
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
        << "\"kfProcessVar\":" << format(cfg.QN) << ","
        << "\"kfMeasurementVar\":" << format(cfg.RN)
        << "}";
    return out;
}

