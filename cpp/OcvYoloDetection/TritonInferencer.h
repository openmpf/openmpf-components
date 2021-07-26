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

class TritonInferencer {

  public:

    TritonInferencer(const Config &cfg);
    ~TritonInferencer(){LOG_TRACE("~TritonInferencer");}

    const std::string& serverUrl;
    const std::string  modelName;
    const std::string  modelVersion;
    int maxBatchSize;
    std::vector<TritonTensorMeta> inputsMeta;
    std::vector<TritonTensorMeta> outputsMeta;
    nvidia::inferenceserver::client::InferOptions inferOptions;
    nvidia::inferenceserver::client::SslOptions sslOptions;

    //std::vector<cv::Mat> infer(const std::vector<cv::Mat> &inputBlobs);

    using ExtractDetectionsFunc =
      std::function<void(std::vector<cv::Mat> outBlobs,
                         std::vector<Frame>::const_iterator begin,
                         std::vector<Frame>::const_iterator end)>;

    void infer(const std::vector<Frame> &frames,
               const std::vector<cv::Mat> &inputBlobs,
               ExtractDetectionsFunc extractDetectionsFun);



    int aquireClientIdBlocking();
    void releaseClientId(int clientId);
    void waitTillAllClientsReleased();

  private:

    std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> statusClient_;
    std::mutex freeClientIdxsMtx_;
    std::condition_variable freeClientIdxCv_;
    std::unordered_set<int> freeClientIdxs_;
    std::vector<std::unique_ptr<TritonClient>> clients_;

    //int batchCompleted_;
    //std::mutex batchCompletedMtx_;
    //std::condition_variable batchCompletedCv_;

    int frameIdxComplete_;
    std::mutex frameIdxCompleteMtx_;
    std::condition_variable frameIdxCompleteCv_;


    void checkServerIsAlive(int maxAttempts) const;
    void checkServerIsReady(int maxAttempts) const;
    void checkModelIsReady(int maxAttempts) const;
    void getModelInputOutputMetaData();
    void removeAllShmRegions(const std::string prefix);

};


#endif //OPENMPF_COMPONENTS_TRITON_CLIENT_H