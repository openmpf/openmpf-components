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

    const size_t inputs_byte_size;

    const size_t outputs_byte_size;

    const std::string inputs_shm_key;

    const std::string outputs_shm_key;

    TritonClient(
      const int id,
      const TritonInferencer *inferencer);

    ~TritonClient();

    using CallbackFunc = std::function<void()>;

    void inferAsync(int inferInputIdx, const cv::Mat& shmBlob, CallbackFunc inferencerLambda);

    cv::Mat getOutput(const TritonTensorMeta& om);

    static const std::string& shm_key_prefix();

    // TODO: Move definitions to *.cpp. Prefix with get*.

    const bool usingShmInput() const {return !inputs_shm_key.empty();}

    const bool usingShmOutput() const {return !outputs_shm_key.empty();}

    const uint8_t* inputs_shm() const {return inputs_shm_;}

  private:

    const TritonInferencer *inferencer_;

    uint8_t* inputs_shm_;

    uint8_t* outputs_shm_;

    std::vector<std::unique_ptr<triton::client::InferInput>>
      inferInputs_;

    std::vector<std::unique_ptr<const triton::client::InferRequestedOutput>>
      inferRequestedOutputs_;

    std::unique_ptr<triton::client::InferResult>
      inferResult_;

    std::unique_ptr<triton::client::InferenceServerGrpcClient>
      grpc_;

    void setupShmRegion(const std::string shm_key,
                        const size_t byte_size,
                        uint8_t* &shm_addr);

    void removeShmRegion(const std::string shm_key,
                         const size_t byte_size,
                        uint8_t* shm_addr);

    void prepareInferInputs();

    void prepareInferRequestedOutputs();

    void inferAsync_(CallbackFunc inferencerLambda);

};

#endif // OPENMPF_COMPONENTS_TRITON_CLIENT_H
