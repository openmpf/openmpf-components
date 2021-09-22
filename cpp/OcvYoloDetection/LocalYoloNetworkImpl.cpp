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

#include <algorithm>
#include <fstream>
#include <list>
#include <utility>
#include <MPFDetectionException.h>
#include <Utils.h>

#include "util.h"
#include "Config.h"
#include "Frame.h"
#include "YoloNetwork.h"
#include "WhitelistFilter.h"

using namespace MPF::COMPONENT;


namespace {

    int ConfigureCudaDeviceIfNeeded(const Config &config, log4cxx::LoggerPtr& log) {
        if (config.cudaDeviceId < 0 || config.tritonEnabled) {
            if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
                // A previous job may have been configured to use CUDA, but this job wasn't.
                // We call cv::cuda::resetDevice() so that GPU memory used by the previous job
                // can be released.
                cv::cuda::resetDevice();
            }
            return -1;
        }

        try {
            if (cv::cuda::getDevice() != config.cudaDeviceId) {
                cv::cuda::resetDevice();
                cv::cuda::setDevice(config.cudaDeviceId);
            }
            return config.cudaDeviceId;
        }
        catch (const cv::Exception &e) {
            if (e.code != cv::Error::GpuApiCallError && e.code != cv::Error::GpuNotSupported) {
                throw;
            }

            std::string message = "An error occurred while trying to set CUDA device: " + e.msg;
            if (config.fallback2CpuWhenGpuProblem) {
                LOG4CXX_WARN(log, message << ". Job will run on CPU instead.")
                return -1;
            }
            else {
                throw MPFDetectionException(MPFDetectionError::MPF_GPU_ERROR, message);
            }
        }
    }


    cv::dnn::Net LoadNetwork(const ModelSettings &modelSettings, int cudaDeviceId,
                             log4cxx::LoggerPtr& log) {

        LOG4CXX_INFO(log, "Attempting to load network using network config file from "
                << modelSettings.networkConfigFile << " and weights from "
                << modelSettings.weightsFile);

        cv::dnn::Net net;
        try {
            net = cv::dnn::readNetFromDarknet(modelSettings.networkConfigFile,
                                              modelSettings.weightsFile);
        }
        catch (const cv::Exception& e) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    std::string("Failed to load model due to: ") + e.what());
        }

        if (cudaDeviceId >= 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }

        LOG4CXX_INFO(log, "Successfully loaded network.");
        return net;
    }


    int GetNumClasses(const cv::dnn::Net &net, const Config &config) {
        int outLayerId = net.getUnconnectedOutLayers().front();
        std::vector<cv::dnn::MatShape> inShapes;
        std::vector<cv::dnn::MatShape> outShapes;
        net.getLayerShapes(
                {1, 3, config.netInputImageSize, config.netInputImageSize},
                outLayerId,
                inShapes,
                outShapes);

        // outputFeatures = x, y, width, height, objectness, ...confidences
        int numOutputFeatures = outShapes.front().back();
        return numOutputFeatures - 5;
    }


    std::vector<std::string> LoadNames(const cv::dnn::Net &net,
                                       const ModelSettings &modelSettings,
                                       const Config &config) {
        std::ifstream namesFile(modelSettings.namesFile);
        if (!namesFile.good()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_OPEN_DATAFILE,
                    "Failed to open names file at: " + modelSettings.namesFile);
        }

        int expectedNumClasses = config.tritonEnabled ? config.tritonNumClasses : GetNumClasses(net, config);
        std::vector<std::string> names;
        names.reserve(expectedNumClasses);

        std::string line;
        while (std::getline(namesFile, line)) {
            Utils::trim(line);
            names.push_back(std::move(line));
            line = "";
        }

        // Remove trailing blank lines
        while (!names.empty() && names.back().empty()) {
            names.pop_back();
        }

        if (names.size() == expectedNumClasses) {
            return names;
        }

        std::stringstream error;
        error << "The network config file at " << modelSettings.networkConfigFile
              << " specifies " << expectedNumClasses << " classes, but the names file at "
              << modelSettings.namesFile << " contains " << names.size()
              << " classes. This is probably because given names file does not correspond to the "
              << "given network configuration file.";
        throw MPFDetectionException(MPF_COULD_NOT_READ_DATAFILE, error.str());
    }


    cv::Mat1f LoadConfusionMatrix(const std::string &path, int numNames) {
        if (path.empty()) {
            return {};
        }

        cv::FileStorage fileStorage;
        cv::Mat1f confusionMatrix;
        try {
            fileStorage.open(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
            if (!fileStorage.isOpened()) {
                throw MPFDetectionException(
                        MPF_COULD_NOT_OPEN_DATAFILE,
                        "Could not open confusion matrix file at: " + path);
            }
        }
        catch (const cv::Exception &ex) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "Could not read the confusion matrix file at \"" + path + "\" due to: "
                    + ex.what());
        }

        fileStorage["confusion"] >> confusionMatrix;
        if (confusionMatrix.empty()) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "Could not read the confusion matrix from the file at \"" + path
                    + R"(" because it doesn't contains a "confusion" entry or it was invalid.)");
        }
        if (confusionMatrix.rows != confusionMatrix.cols) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "Expected the confusion matrix from the file at \"" + path
                    + "\" to be square but it was " + std::to_string(confusionMatrix.rows)
                    + " X " + std::to_string(confusionMatrix.cols) + '.');
        }
        if (confusionMatrix.rows != numNames) {
            throw MPFDetectionException(
                    MPF_COULD_NOT_READ_DATAFILE,
                    "Expected the confusion matrix from the file at \"" + path
                    + "\" to be " + std::to_string(numNames) + " X "
                    + std::to_string(numNames) + ", but it was "
                    + std::to_string(confusionMatrix.rows) + " X "
                    + std::to_string(confusionMatrix.cols) + '.');

        }

        // transpose for use on score row vectors
        cv::transpose(confusionMatrix, confusionMatrix);
        return confusionMatrix;
    }


    std::function<bool(const std::string&)> GetClassFilter(
            const std::string& whiteListPath, const std::vector<std::string> &names) {
        if (whiteListPath.empty()) {
            return [](const std::string&) { return true; };
        }
        else {
            return WhitelistFilter(whiteListPath, names);
        }
    }


    cv::Mat ConvertToBlob(std::vector<Frame>::const_iterator start, std::vector<Frame>::const_iterator stop, const int netInputImageSize) {
        std::vector<cv::Mat> resizedImages;
        resizedImages.reserve(stop - start);
        while(start != stop){
            resizedImages.push_back(
                    start->getDataAsResizedFloat(
                            cv::Size2i(netInputImageSize, netInputImageSize),
                            cv::BORDER_CONSTANT, {127,127,127}));
            start++;
        }
        return cv::dnn::blobFromImages(
                resizedImages,
                1.0,              // no pixel scaleing
                cv::Size(),       // no resizing
                cv::Scalar(),     // no mean subtraction
                true,             // swapRB
                false,            // no cropping
                CV_32F            // make float blob
        );
    }


    std::vector<int> GetTopScoreIndicesDesc(const cv::Mat1f &scores, int numScoresToGet,
                                            float confidenceThreshold) {
        auto scoreIsGreater = [&scores](int i1, int i2) {
            return scores(0, i1) > scores(0, i2);
        };

        std::vector<int> results;
        for (int i = 0; i < scores.cols; ++i) {
            if (scores(0, i) < confidenceThreshold) {
                continue;
            }
            results.push_back(i);
            std::push_heap(results.begin(), results.end(), scoreIsGreater);
            if (results.size() > numScoresToGet) {
                std::pop_heap(results.begin(), results.end(), scoreIsGreater);
                results.pop_back();
            }
        }
        std::sort_heap(results.begin(), results.end(), scoreIsGreater);
        return results;
    }

} // end anonymous namespace


class YoloNetwork::YoloNetworkImpl {
public:
    YoloNetworkImpl(ModelSettings model_settings, const Config &config)
            : modelSettings_(std::move(model_settings))
            , cudaDeviceId_(ConfigureCudaDeviceIfNeeded(config, log_))
            , net_(config.tritonEnabled ?  cv::dnn::Net() : LoadNetwork(modelSettings_, cudaDeviceId_, log_))
            , names_(LoadNames(net_, modelSettings_, config))
            , confusionMatrix_(LoadConfusionMatrix(modelSettings_.confusionMatrixFile, names_.size()))
            , classWhiteListPath_(config.classWhiteListPath)
            , classFilter_(GetClassFilter(classWhiteListPath_, names_)) {}

    ~YoloNetworkImpl() = default;

    // YoloNetworkImpl(const YoloNetworkImpl& other) = default;

    // YoloNetworkImpl& operator=(YoloNetworkImpl rhs) = default;

    void GetDetections(
            std::vector<Frame> &frames,
            ProcessFrameDetectionsFunc processFrameDetectionsFun,
            const Config &config){
        LOG_TRACE("start");
        processFrameDetectionsFun(GetDetectionsCvdnn(frames, config), frames.begin(), frames.end());
        LOG_TRACE("end");
    }

    bool IsCompatible(const ModelSettings &modelSettings, const Config &config) const {
        return modelSettings_.networkConfigFile == modelSettings.networkConfigFile
               && modelSettings_.namesFile == modelSettings.namesFile
               && modelSettings_.weightsFile == modelSettings.weightsFile
               && modelSettings_.confusionMatrixFile == modelSettings.confusionMatrixFile
               && config.cudaDeviceId == cudaDeviceId_
               && config.classWhiteListPath == classWhiteListPath_;
    }

private:
    log4cxx::LoggerPtr log_ = log4cxx::Logger::getLogger("OcvYoloDetection");

    ModelSettings modelSettings_;
    int cudaDeviceId_;
    cv::dnn::Net net_;
    std::vector<std::string> names_;
    cv::Mat1f confusionMatrix_;
    std::string classWhiteListPath_;
    std::function<bool(const std::string&)> classFilter_;


    std::vector<std::vector<DetectionLocation>> GetDetectionsCvdnn(
            const std::vector<Frame> &frames, const Config &config) {

        net_.setInput(ConvertToBlob(frames.begin(),frames.end(), config.netInputImageSize));

        // There are different output layers for different scales, e.g. yolo_82, yolo_94, yolo_106 for yolo v3.
        // Each result is a row vector like: [center_x, center_y, width, height, objectness, ...class_scores]
        // When multiple frames dimensions are: layerOutputs[output_layer][frame][box][feature]
        // When single frame dimensions are: layerOutputs[output_layer][box][feature]
        std::vector<cv::Mat> layerOutputs;
        net_.forward(layerOutputs, net_.getUnconnectedOutLayersNames());

        std::vector<std::vector<DetectionLocation>> detectionsGroupedByFrame;
        detectionsGroupedByFrame.reserve(frames.size());
        for (int frameIdx = 0; frameIdx < frames.size(); ++frameIdx)  {
            detectionsGroupedByFrame.push_back(
                    ExtractFrameDetectionsCvdnn(frameIdx, frames.at(frameIdx), layerOutputs, config));
        }
        return detectionsGroupedByFrame;
    }


    std::vector<DetectionLocation> ExtractFrameDetectionsCvdnn(
            int frameIdx, const Frame &frame, const std::vector<cv::Mat> &layerOutputs,
            const Config &config) const {

        int maxFrameDim = std::max(frame.data.cols, frame.data.rows);
        int horizontalPadding = (maxFrameDim - frame.data.cols) / 2;
        int verticalPadding = (maxFrameDim - frame.data.rows) / 2;
        cv::Vec2f paddingPerSide(horizontalPadding, verticalPadding);

        // cv::dnn::NMSBoxes requires a std::vector<cv::Rect2d> and a std::vector<float>
        std::vector<cv::Rect2d> boundingBoxes;
        std::vector<float> topConfidences;
        std::vector<cv::Mat1f> scoreMats;

        for (const cv::Mat& layerOutput : layerOutputs) {
            cv::Mat frameDetections;
            if (layerOutput.dims == 2) {
                // When a single frame is passed to the network, the output only has two dimensions:
                // (boxes X features)
                frameDetections = layerOutput;
            }
            else {
                // When multiple frames are passed to the network, the output has three dimensions:
                // (frames X boxes X features)
                frameDetections = layerOutput
                        // Get current frame results
                        .row(frameIdx)
                                // Reshape from (1 X boxes X features) to (boxes X features)
                        .reshape(0, layerOutput.size[1]);
            }

            for (int detectionIdx = 0; detectionIdx < frameDetections.rows; ++detectionIdx) {
                cv::Mat1f detectionFeatures = frameDetections.row(detectionIdx);
                cv::Mat1f scores = detectionFeatures.colRange(5, detectionFeatures.cols);

                double maxConfidence = 0;
                cv::Point maxLoc;
                cv::minMaxLoc(scores, nullptr, &maxConfidence, nullptr, &maxLoc);
                const std::string &maxClass = names_.at(maxLoc.x);

                if (maxConfidence >= config.confidenceThreshold && classFilter_(maxClass)) {
                    auto center = cv::Vec2f(detectionFeatures.colRange(0, 2)) * maxFrameDim;
                    auto size = cv::Vec2f(detectionFeatures.colRange(2, 4)) * maxFrameDim;
                    auto topLeft = (center - size / 2.0) - paddingPerSide;

                    boundingBoxes.emplace_back(topLeft(0), topLeft(1),
                                               size(0), size(1));
                    topConfidences.push_back(maxConfidence);
                    scoreMats.push_back(scores);
                }
            }
        }

        std::vector<int> keepIndices;
        cv::dnn::NMSBoxes(boundingBoxes, topConfidences, config.confidenceThreshold,
                          config.nmsThresh, keepIndices);

        std::vector<DetectionLocation> detections;
        detections.reserve(keepIndices.size());
        for (int keepIdx : keepIndices) {
            detections.push_back(CreateDetectionLocationCvdnn(frame, boundingBoxes.at(keepIdx),
                                                              scoreMats.at(keepIdx), config));
        }
        return detections;
    }


    DetectionLocation CreateDetectionLocationCvdnn(
            const Frame &frame,
            const cv::Rect2d &boundingBox,
            const cv::Mat1f &scores,
            const Config &config) const {

        std::vector<int> topScoreIndices = GetTopScoreIndicesDesc(scores, config.numClassPerRegion,
                                                                  config.confidenceThreshold);
        auto topScoreIdxIter = topScoreIndices.begin();
        float topScore = scores(0, *topScoreIdxIter);
        std::string topClass = names_.at(*topScoreIdxIter);
        ++topScoreIdxIter;

        std::ostringstream scoreList;
        scoreList << topScore;
        std::string classList = topClass;

        for (; topScoreIdxIter != topScoreIndices.end(); ++topScoreIdxIter) {
            scoreList << "; " << scores(0, *topScoreIdxIter);
            classList += "; ";
            classList += names_.at(*topScoreIdxIter);
        }
        cv::Mat1f classFeature;
        if (confusionMatrix_.empty()) {
            cv::normalize(scores, classFeature);
        }
        else {
            cv::normalize(scores * confusionMatrix_, classFeature);
        }
        DetectionLocation detection(config, frame, boundingBox, topScore,
                                    std::move(classFeature), cv::Mat());
        detection.detection_properties.emplace("CLASSIFICATION", topClass);
        detection.detection_properties.emplace("CLASSIFICATION LIST", std::move(classList));
        detection.detection_properties.emplace("CLASSIFICATION CONFIDENCE LIST", scoreList.str());
        return detection;
    }
};


YoloNetwork::YoloNetwork(ModelSettings model_settings, const Config &config)
        : pimpl_(new YoloNetworkImpl(model_settings, config)) {}

YoloNetwork::~YoloNetwork() = default;

/*
YoloNetwork::YoloNetwork(const YoloNetwork& other)
        : pimpl_(new YoloNetworkImpl(*other.pimpl_)) {}

YoloNetwork& YoloNetwork::operator=(YoloNetwork rhs) {
    swap(pimpl_, rhs.pimpl_);
    return *this;
}
*/

void YoloNetwork::GetDetections(
        std::vector<Frame> &frames,
        ProcessFrameDetectionsFunc processFrameDetectionsFun,
        const Config &config){
    pimpl_->GetDetections(frames, processFrameDetectionsFun, config);
}

bool YoloNetwork::IsCompatible(const ModelSettings &modelSettings, const Config &config) const {
    return pimpl_->IsCompatible(modelSettings, config);
}
