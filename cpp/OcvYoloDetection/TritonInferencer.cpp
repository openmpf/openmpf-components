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

#include <Utils.h>
#include <grpc_client.h>
#include <MPFDetectionException.h>

#include "util.h"
#include "Config.h"
#include "Frame.h"
#include "TritonTensorMeta.h"
#include "TritonClient.h"
#include "TritonInferencer.h"
#include "TritonExceptionMacros.h"

using namespace MPF::COMPONENT;

void TritonInferencer::checkServerIsAlive(int maxAttempts) const {
  bool ok = false;
  int attempt = 0;
  while(!ok && attempt < maxAttempts){
    TR_CHECK_OK(statusClient_->IsServerLive(&ok),
      "failed to contact TRTIS inference server \"" + serverUrl + "\"");
    attempt++;
  }
  if(!ok){
    THROW_TRITON_EXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
      "unable to verify that TRTIS inference server \""
       + serverUrl + "\" is alive.");
  }else{
    LOG_INFO("Found inference server \"" + serverUrl + "\"");
  }
}


void TritonInferencer::checkServerIsReady(int maxAttempts) const {
  bool ok = false;
  int attempt = 0;
  while(!ok && attempt < maxAttempts){
    TR_CHECK_OK(statusClient_->IsServerReady(&ok),
      "failed to check if TRTIS inference server \""
       + serverUrl + "\" is ready");
    attempt++;
  }
  if(!ok){
    THROW_TRITON_EXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
      "TRTIS inference server \"" + serverUrl + "\" is not ready");
  }else{
    LOG_INFO("Inference server \"" + serverUrl + "\" is ready");
  }
}


void TritonInferencer::checkModelIsReady(int maxAttempts) const{
  bool ok = false;
  int attempt = 0;
  while(!ok && attempt < maxAttempts){
    TR_CHECK_OK(statusClient_->IsModelReady(&ok, modelName, modelVersion),
       "unable to check if TRTIS inference server model \""
        + modelName + "\" ver. " + modelVersion + " is ready");
    if(!ok){
      TR_CHECK_OK(statusClient_->LoadModel(modelName),
        "failed to explicitly load TRTIS inference server model \"" + modelName
         + "\" ver. " + modelName + " on server \"" + serverUrl + "\"");
    }
    attempt++;
  }
  if(!ok){
    THROW_TRITON_EXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
     "TRTIS inference server model \"" + modelName
     + "\" is not ready and could not be loaded explicitly");
  }else{
    LOG_INFO("Inference server model \"" << modelName << " ver. \""
     << modelVersion << "\"" << "\" is loaded and ready for inferencing.");
  }
}


void TritonInferencer::getModelInputOutputMetaData(){

  // get model configuration
  inference::ModelConfigResponse modelConfigResponse;
  TR_CHECK_OK(statusClient_->ModelConfig(
    &modelConfigResponse, modelName,modelVersion),
    "unable to get \"" + modelName + "\" model configuration from server \""
     + serverUrl + "\"");

  // get model inputs and outputs data from server
  maxBatchSize = modelConfigResponse.config().max_batch_size();
  //maxBatchSize = 3;  //for testing
  LOG_INFO("model max supported batch size = " << std::to_string(maxBatchSize));

  // get inputs meta data in inputsMeta TensorMeta vector
  inputsMeta.reserve(modelConfigResponse.config().input_size());
  size_t input_shm_offset = 0;
  for(int i = 0; i < modelConfigResponse.config().input_size(); i++){
    inputsMeta.emplace_back(
      modelConfigResponse.config().input(i), input_shm_offset);
    input_shm_offset += inputsMeta.back().byte_size * maxBatchSize;
    LOG_INFO("input[" << i << "]  = \"" << inputsMeta.back().name << "\" "
              << inputsMeta.back().type << inputsMeta.back().shape
              << " bytes:" << inputsMeta.back().byte_size
              << " shm_offset:" << inputsMeta.back().shm_offset);
  }

  // get outputs meta data in outputsMeta TensorMeta vector
  size_t output_shm_offset = 0;
  outputsMeta.reserve(modelConfigResponse.config().output_size());
  for(int o = 0; o < modelConfigResponse.config().output_size(); o++){
    outputsMeta.emplace_back(
      modelConfigResponse.config().output(o), output_shm_offset);
    output_shm_offset += outputsMeta.back().byte_size * maxBatchSize;
    LOG_INFO("output[" << o << "] = \"" << outputsMeta.back().name << "\" "
              << outputsMeta.back().type << outputsMeta.back().shape
              << " bytes:" << outputsMeta.back().byte_size
              << " shm_offset:" << outputsMeta.back().shm_offset);
  }

}


void TritonInferencer::removeAllShmRegions(const std::string prefix){

    inference::SystemSharedMemoryStatusResponse shm_status;
    statusClient_->SystemSharedMemoryStatus(&shm_status);
    for(const auto& p  : shm_status.regions()){
      std::string region_name = p.second.name();
      if(!region_name.compare(0,prefix.size(),prefix)){
        // found existing mapping with same prefix, so delete it for clean start
        TR_CHECK_OK(statusClient_->UnregisterSystemSharedMemory(region_name),
            "unable to unregister system shared memory region \"" + region_name + "\" from \"" + serverUrl + "\"");
        LOG_TRACE("removed existing registered shm region " << region_name << " of size:" << p.second.byte_size() << " with key:" << p.second.key());
      }
    }
}


void TritonInferencer::infer(
  const std::vector<Frame> &frames,
  const std::vector<cv::Mat> &inputBlobs,
  ExtractDetectionsFunc extractDetectionsFun){

  std::vector<Frame>::const_iterator begin;
  std::vector<Frame>::const_iterator end(frames.begin());

  while(end != frames.end()){

    begin = end;
    int size = std::min(maxBatchSize, static_cast<int>(frames.end() - begin));
    end = end + size;

    // create ocv matrix headers as window into inputBlobs allocated data
    std::vector<cv::Mat> batchInputBlobs;
    for(auto& inputBlob : inputBlobs){
      std::vector<int> shape(inputBlob.size.p, inputBlob.size.p + inputBlob.dims);
      shape[0] = size;
      batchInputBlobs.emplace_back(shape.size(), shape.data(),
        inputBlob.type(), (void*)inputBlob.ptr(begin - frames.begin()));
    }

    int clientId =  acquireClientIdBlocking();
    LOG_TRACE("inferencing frames[" << begin->idx << ".."
      << (end - 1)->idx << "] with client[" << clientId << "]");

    clients_[clientId]->inferAsync(
      batchInputBlobs,

      [this, extractDetectionsFun, clientId, begin, end](){
        std::vector<cv::Mat> results;
        for(int i = 0; i < outputsMeta.size(); ++i){
          results.push_back(clients_[clientId]->getOutput(outputsMeta[i]));
        }
        extractDetectionsFun(results, begin, end);
        releaseClientId(clientId);
      }

    );
  }
}


void TritonInferencer::releaseClientId(int clientId){
  {
    std::lock_guard<std::mutex> lk(freeClientIdsMtx_);
    freeClientIds_.insert(clientId);
    LOG_TRACE("freeing client["<< clientId <<"]");
  }
  freeClientIdsCv_.notify_all();
}


void TritonInferencer::waitTillAllClientsReleased(){
  if(freeClientIds_.size() != clients_.size()){
    std::unique_lock<std::mutex> lk(freeClientIdsMtx_);
    LOG_TRACE("waiting till all clients freed");
    freeClientIdsCv_.wait(lk, [this] {
       return freeClientIds_.size() == clients_.size(); });
  }
  LOG_TRACE("all clients were freed");

}


int TritonInferencer::acquireClientIdBlocking(){

  std::unique_lock<std::mutex> lk(freeClientIdsMtx_);
  if(freeClientIds_.empty()){
    LOG_TRACE("wait for a free client");
    freeClientIdsCv_.wait(lk, [this] { return !freeClientIds_.empty(); });
  }
  auto it = freeClientIds_.begin();
  int id = *it;
  freeClientIds_.erase(it);
  return id;
}


TritonInferencer::TritonInferencer(const Config &cfg)
  : serverUrl(cfg.trtisServer)
  , modelName(cfg.trtisModelName + "-" + std::to_string(cfg.netInputImageSize))
  , modelVersion((cfg.trtisModelVersion > 0) ? std::to_string(cfg.trtisModelVersion) : "")
  , inferOptions(modelName){

  // setup remaining inferencing options
  inferOptions.model_version_ = modelVersion;
  inferOptions.client_timeout_ = cfg.trtisClientTimeout;
  LOG_TRACE("Created inference options for " << modelName << " ver."
    << modelVersion << " and a client timeout of " << std::fixed
    << std::setprecision(6) << inferOptions.client_timeout_ / 1e6 << " sec.");

  // initialize client for server status requests
  TR_CHECK_OK(triton::client::InferenceServerGrpcClient::Create(
    &statusClient_,serverUrl, cfg.trtisVerboseClient, cfg.trtisUseSSL, sslOptions),
    "unable to create TRTIS inference client for \"" + serverUrl + "\"");

  // do some check on server and model
  checkServerIsAlive(cfg.trtisMaxConnectionSetupAttempts);
  checkServerIsReady(cfg.trtisMaxConnectionSetupAttempts);
  checkModelIsReady(cfg.trtisMaxConnectionSetupAttempts);

  // get model configuration
  getModelInputOutputMetaData();

  // clean up any existing shared memory regions from client host
  if(cfg.trtisUseShm){
    removeAllShmRegions(TritonClient::shm_key_prefix());
  }

  // create clients for concurrent inferencing
  LOG_TRACE("Creating " << cfg.trtisMaxInferConcurrency << " clients for concurrent inferencing");
  for(int i = 0; i < cfg.trtisMaxInferConcurrency; i++){
    clients_.emplace_back(std::unique_ptr<TritonClient>(
      new TritonClient(i, cfg, this )));
    freeClientIds_.insert(i);

  }
}
