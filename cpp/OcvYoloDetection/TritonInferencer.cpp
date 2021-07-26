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


/** ***************************************************************************
* check the server is alive
***************************************************************************** */
void TritonInferencer::checkServerIsAlive(int maxAttempts) const {
  bool ok = false;
  int attempt = 0;
  while(!ok && attempt < maxAttempts){
    NI_CHECK_OK(statusClient_->IsServerLive(&ok),
      "failed to contact TRTIS inference server \"" + serverUrl + "\"");
    attempt++;
  }
  if(!ok){
    THROW_TRTISEXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
      "unable to verify that TRTIS inference server \""
       + serverUrl + "\" is alive.");
  }else{
    LOG_INFO("Found inference server \"" + serverUrl + "\"");
  }
}

/** ***************************************************************************
* check server is ready
***************************************************************************** */
void TritonInferencer::checkServerIsReady(int maxAttempts) const {
  // check the server is ready
  bool ok = false;
  int attempt = 0;
  while(!ok && attempt < maxAttempts){
    NI_CHECK_OK(statusClient_->IsServerReady(&ok),
      "failed to check if TRTIS inference server \""
       + serverUrl + "\" is ready");
    attempt++;
  }
  if(!ok){
    THROW_TRTISEXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
      "TRTIS inference server \"" + serverUrl + "\" is not ready");
  }else{
    LOG_INFO("Inference server \"" + serverUrl + "\" is ready");
  }
}

/** ***************************************************************************
* check the model is ready or to load it
***************************************************************************** */
void TritonInferencer::checkModelIsReady(int maxAttempts) const{
  bool ok = false;
  int attempt = 0;
  while(!ok && attempt < maxAttempts){
    NI_CHECK_OK(statusClient_->IsModelReady(&ok, modelName, modelVersion),
       "unable to check if TRTIS inference server model \""
        + modelName + "\" ver. " + modelVersion + " is ready");
    if(!ok){
      NI_CHECK_OK(statusClient_->LoadModel(modelName),
        "failed to explicitly load TRTIS inference server model \"" + modelName
         + "\" ver. " + modelName + " on server \"" + serverUrl + "\"");
    }
    attempt++;
  }
  if(!ok){
    THROW_TRTISEXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
     "TRTIS inference server model \"" + modelName
     + "\" is not ready and could not be loaded explicitly");
  }else{
    LOG_INFO("Inference server model \"" << modelName << " ver. \""
     << modelVersion << "\"" << "\" is loaded and ready for inferencing.");
  }
}

/** ***************************************************************************
* get meta data about model inputs and outputs, batch size, shapes, types etc.
***************************************************************************** */
void TritonInferencer::getModelInputOutputMetaData(){

  // get model configuration
  inference::ModelConfigResponse modelConfigResponse;
  NI_CHECK_OK(statusClient_->ModelConfig(
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

/** ***************************************************************************
* remove and unregister shared memory regions for clients on this host
***************************************************************************** */
void TritonInferencer::removeAllShmRegions(const std::string prefix){

   // Clean up registered shared memory regions on server
    inference::SystemSharedMemoryStatusResponse shm_status;
    statusClient_->SystemSharedMemoryStatus(&shm_status);
    for(const auto& p  : shm_status.regions()){
      std::string region_name = p.second.name();
      if(!region_name.compare(0,prefix.size(),prefix)){
        // found existing mapping with same prefix, so delete it for clean start
        // todo: should check and remove regions from client docker containers that are no longer running as well
        NI_CHECK_OK(statusClient_->UnregisterSystemSharedMemory(region_name),
            "unable to unregister system shared memory region \"" + region_name + "\" from \"" + serverUrl + "\"");
        LOG_TRACE("removed existing registered shm region " << region_name << " of size:" << p.second.byte_size() << " with key:" << p.second.key());
      }
    }
}


/** ****************************************************************************
 * inference input specified in vector of 4D data blobs
***************************************************************************** */
void TritonInferencer::infer(
  const std::vector<Frame> &frames,
  const std::vector<cv::Mat> &inputBlobs,
  ExtractDetectionsFunc extractDetectionsFun){

  #define MYASYNC
  #ifdef MYASYNC
    LOG_TRACE("start async");
    //int total = inputBlobs.at(0).size[0];
    //int batchBegin = 0;
    //int batchEnd = 0;
    //batchCompleted_ = -1;

    frameIdxComplete_ = frames.front().idx - 1;
    std::vector<Frame>::const_iterator begin;
    std::vector<Frame>::const_iterator end(frames.begin());

    while(end != frames.end()){

      begin = end;
      int size = std::min(maxBatchSize, static_cast<int>(frames.end() - begin));
      end = end + size;

      // create matrix headers as window into inputBlobs allocated data
      std::vector<cv::Mat> batchInputBlobs;
      for(auto& inputBlob : inputBlobs){
        std::vector<int> shape(inputBlob.size.p, inputBlob.size.p + inputBlob.dims);
        shape[0] = size;
        //LOG_TRACE("TritonInferencer::infer inferencing batch blob with shape "<< shape);
        batchInputBlobs.emplace_back(shape.size(), shape.data(),
                                     inputBlob.type(),
                                     (void*)inputBlob.ptr(begin - frames.begin()));
      }

      int clientId =  aquireClientIdBlocking();
      LOG_TRACE("inferencing frames[" << begin->idx << ".." << (end - 1)->idx << "] with client[" << clientId << "]");

      clients_[clientId]->inferAsync(
        batchInputBlobs,
        [this, extractDetectionsFun, clientId, begin, end](){
          std::vector<cv::Mat> results; // popoulate in loop
          // copy to results
          for(int i = 0; i < outputsMeta.size(); ++i){
            cv::Mat result = clients_[clientId]->getOutput(outputsMeta[i]);
            results.push_back(result);
            //LOG_TRACE("TritonInferencer::infer  got result of shape " << std::vector<int>(result.size.p,result.size.p+result.dims));
            //std::memcpy(results[i].ptr(batchBegin), result.ptr(0), result.total() * result.elemSize());
          }

          { // block till prior blobs/frames have been processed
            int frameIdxToWaitFor = begin->idx - 1;
            std::unique_lock<std::mutex> lk(frameIdxCompleteMtx_);
            LOG_TRACE("waiting for frame[" << frameIdxToWaitFor << "] to complete");
            frameIdxCompleteCv_.wait(lk,
              //[this, frameIdxToWaitFor]{return frameIdxComplete_ >= frameIdxToWaitFor;});
              [this, frameIdxToWaitFor]{return frameIdxComplete_ >= frameIdxToWaitFor;});
            LOG_TRACE("done waiting for frame[" << frameIdxToWaitFor << "]");
          //}

          // extract detections and move frames (invalidates iterators)
          int firstFrameIdx = begin->idx;
          int lastFrameIdx = (end - 1)->idx;
          extractDetectionsFun(results, begin, end);

          releaseClientId(clientId);

          //{ // update frameIdxComplete
          //  std::lock_guard<std::mutex> lk(frameIdxCompleteMtx_);
            frameIdxComplete_ = lastFrameIdx;
            LOG_TRACE("completed frames["<< firstFrameIdx << ".." << lastFrameIdx << "] with client[" << clientId << "]");
          }
          frameIdxCompleteCv_.notify_all();

        });

    }
    LOG_TRACE("end async");
  #else
    // pre allocate results vector
    std::vector<cv::Mat> results;
    for(TritonTensorMeta& om : outputsMeta){
      std::vector<int> shape({inputBlobs[0].size[0]});
      shape.insert(shape.end(), om.shape.begin(), om.shape.end());
      LOG_TRACE("Preallocation results ocv matrix of type \"" << om.type << "\" and size " << shape);
      results.emplace_back(shape, om.cvType);
    }

    int total = inputBlobs.at(0).size[0];
    int batchBegin = 0;
    int batchEnd = 0;

    while(batchEnd < total){
      batchBegin = batchEnd;
      batchEnd = std::min(batchBegin + maxBatchSize, total);
      int batchSize = batchEnd - batchBegin;

      // create matrix headers as window into inputBlobs allocated data
      std::vector<cv::Mat> batchInputBlobs;
      for(auto& inputBlob : inputBlobs){
        std::vector<int> shape(inputBlob.size.p, inputBlob.size.p + inputBlob.dims);
        shape[0] = batchSize;
        LOG_TRACE("inferencing batch blob with shape "<< shape);
        batchInputBlobs.emplace_back(shape.size(), shape.data(),
          inputBlob.type(), (void*)inputBlob.ptr(batchBegin));
      }

      // inference batch
        LOG_DEBUG("inferencing: batch [" << batchBegin << "..." << batchEnd << "]");
        clients_[0]->infer(batchInputBlobs);
        for(int i = 0; i < results.size(); ++i){
          cv::Mat result = clients_[0]->getOutput(outputsMeta[i]);
          std::memcpy(results[i].ptr(batchBegin), result.ptr(0), result.total() * result.elemSize());
        }
    }

    dFun(results);
  #endif

}

/** ****************************************************************************
***************************************************************************** */
void TritonInferencer::releaseClientId(int clientId){
  {
    std::lock_guard<std::mutex> lk(freeClientIdxsMtx_);
    freeClientIdxs_.insert(clientId);
    LOG_TRACE("freeing client["<< clientId <<"]");
  }
  freeClientIdxCv_.notify_all();
}

/** ****************************************************************************
***************************************************************************** */
void TritonInferencer::waitTillAllClientsReleased(){
  if(freeClientIdxs_.size() != clients_.size()){
    std::unique_lock<std::mutex> lk(freeClientIdxsMtx_);
    LOG_TRACE("waiting till all clients freed");
    freeClientIdxCv_.wait(lk,
      [this]{ return freeClientIdxs_.size() == clients_.size();});
  }
  LOG_TRACE("all clients were freed");

}

/** ****************************************************************************
***************************************************************************** */
int TritonInferencer::aquireClientIdBlocking(){

  std::unique_lock<std::mutex> lk(freeClientIdxsMtx_);
  if(freeClientIdxs_.empty()){
    LOG_TRACE("wait for a free client");
    freeClientIdxCv_.wait(lk, [this] { return !freeClientIdxs_.empty(); });
  }
  auto it = freeClientIdxs_.begin();
  int id = *it;
  freeClientIdxs_.erase(it);
  //LOG_TRACE("using triton client[" << id << "]");
  return id;
}

/** ****************************************************************************
 * Constructor creates a pool of clients for a inference server.
***************************************************************************** */
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
  NI_CHECK_OK(nvidia::inferenceserver::client::InferenceServerGrpcClient::Create(&statusClient_,
      serverUrl, cfg.trtisVerboseClient, cfg.trtisUseSSL, sslOptions),
    "unable to create TRTIS inference client for \"" + serverUrl + "\"");

  // do some check on server and model
  checkServerIsAlive(cfg.trtisMaxConnectionSetupAttempts);
  checkServerIsReady(cfg.trtisMaxConnectionSetupAttempts);
  checkModelIsReady(cfg.trtisMaxConnectionSetupAttempts);

  // get model configuration
  getModelInputOutputMetaData();

  // clean up existing shared memory regions from client host
  if(cfg.trtisUseShm){
    removeAllShmRegions(TritonClient::shm_key_prefix());
  }

  // create max number of clients for concurrent inferencing
  LOG_TRACE("Creating " << cfg.trtisMaxInferConcurrency << " clients for concurrent inferencing");
  for(int i = 0; i < cfg.trtisMaxInferConcurrency; i++){
    clients_.emplace_back(std::unique_ptr<TritonClient>(
      new TritonClient(i, cfg, this )));
    freeClientIdxs_.insert(i);

  }

}

/**/