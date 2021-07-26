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

#ifndef OPENMPF_COMPONENTS_TRITON_CLIENT_H
#define OPENMPF_COMPONENTS_TRITON_CLIENT_H

class TritonInferencer;

class TritonClient {

  public:
    const int id;

    TritonClient(
      const int id,
      const Config& cfg,
      const TritonInferencer *inferencer);

    ~TritonClient();

    void infer(const std::vector<cv::Mat> &inputBlobs);

    using CallbackFunc = std::function<void()>;
    void inferAsync(const std::vector<cv::Mat> &inputBlobs, CallbackFunc inferencerLambda);

    cv::Mat getOutput(const TritonTensorMeta& om);
    static const std::string& shm_key_prefix();

    //void check(){LOG_TRACE("client[" << id << "]->inferencer: " << std::hex << inferencer_);}

    void setWait();

  private:

    std::mutex mtx_;
    std::condition_variable cv_;
    int dependantOnClientId;

    //const Config& cfg_;
    const TritonInferencer *inferencer_;
    const size_t inputs_byte_size_;
    const size_t outputs_byte_size_;

    const std::string inputs_shm_key_;
    const std::string outputs_shm_key_;
    uint8_t* inputs_shm_;
    uint8_t* outputs_shm_;

    std::vector<std::unique_ptr<nvidia::inferenceserver::client::InferInput>> inferInputs_;
    std::vector<std::unique_ptr<const nvidia::inferenceserver::client::InferRequestedOutput>> inferReqestedOutputs_;

    std::unique_ptr<nvidia::inferenceserver::client::InferResult> inferResult_;
    std::unique_ptr<nvidia::inferenceserver::client::InferenceServerGrpcClient> grpc_;

    void setupShmRegions();
    void removeShmRegions();
    void prepareInferInputs();
    void prepareInferRequestedOutputs();
    void setInferInputsData(const std::vector<cv::Mat> &blobs);

};

#endif