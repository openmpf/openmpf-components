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

#ifndef OPENMPF_COMPONENTS_TRITON_INFERENCER_H
#define OPENMPF_COMPONENTS_TRITON_INFERENCER_H

#include <functional>
#include <string>
#include <vector>

#include "../Config.h"
#include "../Frame.h"

#include <triton/core/tritonbackend.h>

class TritonInferencer {

  public:
    std::vector<TritonTensorMeta> inputsMeta;

    std::vector<TritonTensorMeta> outputsMeta;

    // TODO: Move definitions to *.cpp. Prefix with get*.

    const triton::client::InferOptions& inferOptions() const {return inferOptions_;}

    const triton::client::SslOptions& sslOptions() const {return sslOptions_;}

    const std::string& serverUrl() const {return serverUrl_;}

    const std::string& modelName() const {return modelName_;}

    const std::string& modelVersion() const {return modelVersion_;}

    const bool useShm() const {return useShm_;}

    const bool verboseClient() const {return verboseClient_;}

    const bool useSSL() const {return useSSL_;}

    const int maxBatchSize() const {return maxBatchSize_;}

    using ExtractDetectionsFunc =
      std::function<void(std::vector<cv::Mat> outBlobs,
                         std::vector<Frame>::const_iterator begin,
                         std::vector<Frame>::const_iterator end)>;

    void infer(const std::vector<Frame> &frames,
               const TritonTensorMeta &inputMeta,
               ExtractDetectionsFunc extractDetectionsFun);

    int acquireClientIdBlocking();

    void releaseClientId(int clientId);

    void waitTillAllClientsReleased();

    std::string getModelNameAndVersion() const;

    TritonInferencer(const Config &cfg);

  private:

    std::string serverUrl_;

    std::string modelName_;

    std::string modelVersion_;

    bool useShm_;

    bool useSSL_;

    bool verboseClient_;

    int maxBatchSize_;

    triton::client::SslOptions sslOptions_;

    triton::client::InferOptions inferOptions_;

    std::unique_ptr<triton::client::InferenceServerGrpcClient> statusClient_;

    std::vector<std::unique_ptr<TritonClient>> clients_;

    std::unordered_set<int> freeClientIds_;

    std::mutex freeClientIdsMtx_;

    std::condition_variable freeClientIdsCv_;

    void checkServerIsAlive(int maxRetries, int initialDelaySeconds) const;

    void checkServerIsReady(int maxRetries, int initialDelaySeconds) const;

    void checkModelIsReady(int maxRetries, int initialDelaySeconds) const;

    void getModelInputOutputMetaData();

    void removeAllShmRegions(const std::string prefix);
};

#endif // OPENMPF_COMPONENTS_TRITON_INFERENCER_H