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

#include <fstream>
#include <utility>
#include <vector>

#include <MPFDetectionException.h>

#include "../util.h"

#include <grpc_client.h>
#include "../triton/TritonTensorMeta.h"
#include "../triton/TritonClient.h"
#include "../triton/TritonInferencer.h"

#include "BaseYoloNetworkImpl.h"

using namespace MPF::COMPONENT;

// match Triton yololayer.h constant
static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
static constexpr int OUTPUT_BLOB_DIM_1 = MAX_OUTPUT_BBOX_COUNT * 7 + 1;

class YoloNetwork::YoloNetworkImpl : public BaseYoloNetworkImpl {
public:
    YoloNetworkImpl(ModelSettings model_settings, const Config &config)
            : BaseYoloNetworkImpl(model_settings, config),
              tritonInferencer_(std::move(ConnectTritonInferencer(config))) {}

    ~YoloNetworkImpl() = default;

    void GetDetections(
            std::vector<Frame> &frames,
            ProcessFrameDetectionsCallback processFrameDetectionsCallback,
            const Config &config) {
        if (!config.tritonEnabled) {
            processFrameDetectionsCallback(GetDetectionsCvdnn(frames, config), frames.begin(), frames.end());
        } else {
            GetDetectionsTriton(frames, processFrameDetectionsCallback, config);
        }
    }

    // Determines if the cached YoloNetwork should be reused or not.
    bool IsCompatible(const ModelSettings &modelSettings, const Config &config) const {
        if (config.tritonEnabled) {
            return tritonInferencer_
                   && config.tritonServer == tritonInferencer_->serverUrl()
                   && config.tritonModelName == tritonInferencer_->modelName()
                   && config.tritonModelVersion == tritonInferencer_->modelVersion()
                   && config.tritonUseShm == tritonInferencer_->useShm()
                   && config.tritonUseSSL == tritonInferencer_->useSSL()
                   && config.tritonVerboseClient == tritonInferencer_->verboseClient()
                   && config.netInputImageSize == tritonInferencer_->inputsMeta.at(0).shape[2]
                   // common settings with local yolo network
                   && modelSettings_.namesFile == modelSettings.namesFile
                   && modelSettings_.confusionMatrixFile == modelSettings.confusionMatrixFile
                   && config.classWhiteListPath == classWhiteListPath_;
        } else {
            return !tritonInferencer_
                   && modelSettings_.ocvDnnNetworkConfigFile == modelSettings.ocvDnnNetworkConfigFile
                   && modelSettings_.ocvDnnWeightsFile == modelSettings.ocvDnnWeightsFile
                   && modelSettings_.namesFile == modelSettings.namesFile
                   && modelSettings_.confusionMatrixFile == modelSettings.confusionMatrixFile
                   && config.cudaDeviceId == cudaDeviceId_
                   && config.classWhiteListPath == classWhiteListPath_;
        }
    }


    void Finish() {
        if (tritonInferencer_) {
            // wait for clients and check for client exception at the end of the job
            tritonInferencer_->waitTillAllClientsReleased();
            frameIdxComplete_ = -1;
            tritonInferencer_->rethrowClientException();
        }
    }


    void Reset() noexcept {
        if (tritonInferencer_) {
            // wait for clients but don't check for client exception; it's too late to care
            tritonInferencer_->waitTillAllClientsReleased();
            frameIdxComplete_ = -1;
            tritonInferencer_->reset();
        }
    }

private:
    int frameIdxComplete_ = -1;
    std::mutex frameIdxCompleteMtx_;
    std::condition_variable frameIdxCompleteCv_;
    std::unique_ptr<TritonInferencer> tritonInferencer_;


    std::unique_ptr<TritonInferencer> ConnectTritonInferencer(const Config &config) {
        if (!config.tritonEnabled) {
            return nullptr;
        }

        std::unique_ptr<TritonInferencer> tritonInferencer(new TritonInferencer(config));
        std::string modelNameAndVersion = tritonInferencer->getModelNameAndVersion();

        if (tritonInferencer->inputsMeta.size() != 1) {
            std::stringstream ss;
            ss << "Configured Triton inference server model " << modelNameAndVersion << " has "
               << tritonInferencer->inputsMeta.size() << " inputs, but only one is expected.";
            throw MPFDetectionException(MPFDetectionError::MPF_INVALID_PROPERTY, ss.str());
        }

        if (tritonInferencer->inputsMeta.at(0).shape[0] != 3
            || tritonInferencer->inputsMeta.at(0).shape[1] != tritonInferencer->inputsMeta.at(0).shape[2]
            || tritonInferencer->inputsMeta.at(0).shape[2] != config.netInputImageSize) {
            std::stringstream ss;
            ss << "Configured Triton inference server model " << modelNameAndVersion
               << " has first input shape "
               << tritonInferencer->inputsMeta.at(0).shape << ", but data has shape "
               << std::vector<int>{3, config.netInputImageSize, config.netInputImageSize} << ".";
            throw MPFDetectionException(MPFDetectionError::MPF_INVALID_PROPERTY, ss.str());
        }

        if (tritonInferencer->outputsMeta.at(0).shape[0] != OUTPUT_BLOB_DIM_1
            || tritonInferencer->outputsMeta.at(0).shape[1] != 1
            || tritonInferencer->outputsMeta.at(0).shape[2] != 1) {
            std::stringstream ss;
            ss << "Configured Triton inference server model " << modelNameAndVersion
               << " has first output shape "
               << tritonInferencer->outputsMeta.at(0).shape << ", but "
               << std::vector<int>{OUTPUT_BLOB_DIM_1, 1, 1} << " was expected.";
            throw MPFDetectionException(MPFDetectionError::MPF_INVALID_PROPERTY, ss.str());
        }

        return tritonInferencer;
    }


    void GetDetectionsTriton(
            const std::vector<Frame> &frames,
            ProcessFrameDetectionsCallback processFrameDetectionsCallback,
            const Config &config) {

        // Send async request to Triton using this batch of frames to get output blobs.
        tritonInferencer_->infer(frames, tritonInferencer_->inputsMeta.at(0),

                                 // LAMBDA: This callback will extract detections from output blobs.
                                 // Also, it will invoke processFrameDetectionsCallback to process detections (e.g. tracking).
                                 [this, &config, processFrameDetectionsCallback]
                                         (std::vector<cv::Mat> outBlobs,
                                          std::vector<Frame>::const_iterator begin,
                                          std::vector<Frame>::const_iterator end) {

                                     cv::Mat outBlob = outBlobs.at(0); // yolo only has one output tensor
                                     int numFrames = end - begin;

                                     LOG_TRACE("frameCount: " << numFrames << " outBlob.size(): "
                                               << std::vector<int>(outBlob.size.p, outBlob.size.p + outBlob.dims));
                                     assert(("Blob's first dim should equal number of frames.",
                                             outBlob.size[0] == numFrames));

                                     LOG_TRACE("Received outBlob["
                                               << outBlob.size[0] << "," << outBlob.size[1] << ","
                                               << outBlob.size[2] << "," << outBlob.size[3] << "].");
                                     assert(("Output blob shape should be [frames, detections, 1, 1].",
                                             outBlob.size[0] <= tritonInferencer_->maxBatchSize()
                                             && outBlob.size[1] == OUTPUT_BLOB_DIM_1
                                             && outBlob.size[2] == 1
                                             && outBlob.size[3] == 1));

                                     // parse output blob into detections
                                     std::vector<std::vector<DetectionLocation>> detectionsGroupedByFrame;
                                     detectionsGroupedByFrame.reserve(numFrames);

                                     LOG_TRACE("Extracting detections for frames["
                                               << begin->idx << ".." << (end - 1)->idx << "].");
                                     int i = 0;
                                     for (auto frameIt = begin; frameIt != end; ++i, ++frameIt) {
                                         detectionsGroupedByFrame.push_back(
                                                 ExtractFrameDetectionsTriton(*frameIt, outBlob.ptr<float>(i, 0),
                                                                              config));
                                     }

                                     { // exact frame sequencing needed from here on due to tracking...
                                         int frameIdxToWaitFor = begin->idx - 1;
                                         int frameIdxLast = (end - 1)->idx;
                                         std::unique_lock<std::mutex> lk(frameIdxCompleteMtx_);
                                         LOG_TRACE("Waiting for frame[" << frameIdxToWaitFor << "] to complete.");
                                         frameIdxCompleteCv_.wait(lk,
                                                                  [this, frameIdxToWaitFor] {
                                                                      return frameIdxComplete_ == frameIdxToWaitFor;
                                                                  });
                                         LOG_TRACE("Done waiting for frame[" << frameIdxToWaitFor << "].");

                                         processFrameDetectionsCallback(std::move(detectionsGroupedByFrame), begin, end);

                                         frameIdxComplete_ = frameIdxLast;

                                         LOG_TRACE("Completed frames["
                                                   << begin->idx << ".." << frameIdxComplete_ << "].");
                                     }
                                     frameIdxCompleteCv_.notify_all();
                                 });
    }


    std::vector<DetectionLocation> ExtractFrameDetectionsTriton(
            const Frame &frame, float *data, const Config &config) const {

        float maxFrameDim = std::max(frame.data.cols, frame.data.rows);
        int horizontalPadding = (maxFrameDim - frame.data.cols) / 2;
        int verticalPadding = (maxFrameDim - frame.data.rows) / 2;
        cv::Vec2f paddingPerSide(horizontalPadding, verticalPadding);

        std::vector<cv::Rect2d> boundingBoxes;
        std::vector<float> topConfidences;
        std::vector<int> classifications;

        int numDetections = static_cast<int>(data[0]);
        // dmat[d,0...6] = [x_center, y_center, width, height, det_score, class, class_score]
        cv::Mat dmat(OUTPUT_BLOB_DIM_1 - 1, 7, CV_32F, &data[1]);

        float rescale2Frame = maxFrameDim / config.netInputImageSize;
        for (int det = 0; det < numDetections; ++det) {
            float maxConfidence = dmat.at<float>(det, 4);
            int classIdx = static_cast<int>(dmat.at<float>(det, 5));
            const std::string &maxClass = names_.at(classIdx);

            if (maxConfidence >= config.confidenceThreshold && classFilter_(maxClass)) {
                auto center = cv::Vec2f(dmat.at<float>(det, 0),
                                        dmat.at<float>(det, 1)) * rescale2Frame;
                auto size = cv::Vec2f(dmat.at<float>(det, 2),
                                      dmat.at<float>(det, 3)) * rescale2Frame;
                auto topLeft = (center - size / 2.0) - paddingPerSide;

                boundingBoxes.emplace_back(topLeft(0), topLeft(1), size(0), size(1));
                topConfidences.push_back(maxConfidence);
                classifications.push_back(classIdx);
            }
        }

        std::vector<int> keepIndecies;
        cv::dnn::NMSBoxes(boundingBoxes, topConfidences, config.confidenceThreshold,
                          config.nmsThresh, keepIndecies);

        std::vector<DetectionLocation> detections;
        detections.reserve(keepIndecies.size());
        for (int keepIdx: keepIndecies) {
            detections.push_back(
                    CreateDetectionLocationTriton(frame, boundingBoxes.at(keepIdx),
                                                  topConfidences.at(keepIdx), classifications.at(keepIdx), config));

            // always calc DFT in callback threads for performance reasons
            detections.back().getDFTFeature();
        }
        return detections;
    }


    DetectionLocation CreateDetectionLocationTriton(
            const Frame &frame,
            const cv::Rect2d &boundingBox,
            const float score,
            const int classIdx,
            const Config &config) const {

        assert(("classIdx: " + std::to_string(classIdx) + " >= " + std::to_string(names_.size()),
                classIdx < names_.size()));

        cv::Mat1f classFeature(1, names_.size(), 0.0);
        classFeature.at<float>(0, classIdx) = 1.0;
        if (!confusionMatrix_.empty()) {
            classFeature = classFeature * confusionMatrix_;
        }

        DetectionLocation detection(config, frame, boundingBox, score,
                                    std::move(classFeature), cv::Mat());
        detection.detection_properties.emplace("CLASSIFICATION", names_.at(classIdx));
        detection.detection_properties.emplace("CLASSIFICATION LIST", names_.at(classIdx));
        detection.detection_properties.emplace("CLASSIFICATION CONFIDENCE LIST", std::to_string(score));

        return detection;
    }
};


YoloNetwork::YoloNetwork(ModelSettings model_settings, const Config &config)
        : pimpl_(new YoloNetworkImpl(model_settings, config)) {}

YoloNetwork::~YoloNetwork() = default;

void YoloNetwork::GetDetections(
        std::vector<Frame> &frames,
        ProcessFrameDetectionsCallback processFrameDetectionsFun,
        const Config &config) {
    pimpl_->GetDetections(frames, processFrameDetectionsFun, config);
}

bool YoloNetwork::IsCompatible(const ModelSettings &modelSettings, const Config &config) const {
    return pimpl_->IsCompatible(modelSettings, config);
}

void YoloNetwork::Finish() {
    return pimpl_->Finish();
}

void YoloNetwork::Reset() noexcept {
    return pimpl_->Reset();
}
