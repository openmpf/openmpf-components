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

#include <grpc_client.h>
#include <MPFDetectionException.h>


#include "util.h"
#include "shm_utils.h"
#include "Config.h"
#include "Frame.h"
#include "TritonTensorMeta.h"
#include "TritonExceptionMacros.h"
#include "TritonClient.h"
#include "TritonInferencer.h"

#define LOG_TRACE(MSG){ LOG4CXX_TRACE(Config::log,LOG_PREFIX << MSG) }

using namespace MPF::COMPONENT;

/** ****************************************************************************
* get vector of raw pointers from vector of unique pointers
***************************************************************************** */
template<typename T>
std::vector<T*> getRaw(const std::vector<std::unique_ptr<T>> &v){
  std::vector<T*> raw;
  for(const auto& i : v){
    raw.push_back(i.get());
  }
  return raw;
}

/** ***************************************************************************
* generate fixed prefix for share memory keys
***************************************************************************** */
const std::string& TritonClient::shm_key_prefix(){
  static std::string prefix = "\\" + hostname();
  return prefix;
}

/** ***************************************************************************
* setup shared memory regions for client inputs and outputs
***************************************************************************** */
void TritonClient::setupShmRegions(){
  int shm_fd;
  if(!inputs_shm_key_.empty()){
    LOG_TRACE("Setting up shared memory input blocks for clients on host: " << shm_key_prefix());

    NI_CHECK_OK(nic::CreateSharedMemoryRegion(inputs_shm_key_, inputs_byte_size_, &shm_fd),
      "unable to create shared memory region " + inputs_shm_key_ + " on host");
    NI_CHECK_OK(nic::MapSharedMemory(shm_fd, 0, inputs_byte_size_, (void**) &inputs_shm_),
      "unable to map shared memory region " + inputs_shm_key_ + " to client address space");
    NI_CHECK_OK(nic::CloseSharedMemory(shm_fd),
        "failed ot close shared memory region " + inputs_shm_key_ + " on host");
    // Register shared memory region with server
    NI_CHECK_OK(grpc_->RegisterSystemSharedMemory(inputs_shm_key_, inputs_shm_key_, inputs_byte_size_),
      "unable to register " + inputs_shm_key_ + " shared memory with server \"" + inferencer_->serverUrl + "\"");
    LOG_TRACE("registered shared memory with key " << inputs_shm_key_ << " of size " << inputs_byte_size_ << " at address " << std::hex << (void*)inputs_shm_);
  }
  if(!outputs_shm_key_.empty()){
    LOG_TRACE("Setting up shared memory output blocks for clients on host: " << shm_key_prefix());
    NI_CHECK_OK(nic::CreateSharedMemoryRegion(outputs_shm_key_, outputs_byte_size_, &shm_fd),
      "unable to create shared memory region " + outputs_shm_key_ + " on host");
    NI_CHECK_OK(nic::MapSharedMemory(shm_fd, 0, outputs_byte_size_, (void**) &outputs_shm_),
      "unable to map shared memory region " + outputs_shm_key_ + " to client address space");
    NI_CHECK_OK(nic::CloseSharedMemory(shm_fd),
        "failed ot close shared memory region " + outputs_shm_key_ + " on host");
    // Register shared memory region with server
    NI_CHECK_OK(grpc_->RegisterSystemSharedMemory(outputs_shm_key_, outputs_shm_key_, outputs_byte_size_),
      "unable to register " + outputs_shm_key_ + " shared memory with server \"" + inferencer_->serverUrl + "\"");
    LOG_TRACE("registered shared memory with key " <<  outputs_shm_key_ << " of size " << outputs_byte_size_ << " at address " << std::hex << (void*)outputs_shm_);
  }
}

/** ***************************************************************************
* remove shared memory regions for client inputs and outputs
***************************************************************************** */
void TritonClient::removeShmRegions(){
  if(!inputs_shm_key_.empty()){
    NI_CHECK_OK(grpc_->UnregisterSystemSharedMemory(inputs_shm_key_),
      "unable to unregister shared memory region " + inputs_shm_key_ + " from server \"" + inferencer_->serverUrl + "\"");
    NI_CHECK_OK(nic::UnmapSharedMemory((void*) inputs_shm_, inputs_byte_size_),
      "unable to unmap shared memory region " + inputs_shm_key_ + " from client address space");
    NI_CHECK_OK(nic::UnlinkSharedMemoryRegion(inputs_shm_key_),
      "unable to remove shared memory region " + inputs_shm_key_ + " on host");
  }
  if(!outputs_shm_key_.empty()){
    NI_CHECK_OK(grpc_->UnregisterSystemSharedMemory(outputs_shm_key_),
      "unable to unregister shared memory region " + outputs_shm_key_ + " from server \"" + inferencer_->serverUrl + "\"");
    NI_CHECK_OK(nic::UnmapSharedMemory((void*) outputs_shm_, outputs_byte_size_),
      "unable to unmap shared memory region " + outputs_shm_key_ + " from client address space");
    NI_CHECK_OK(nic::UnlinkSharedMemoryRegion(outputs_shm_key_),
      "unable to remove shared memory region " + outputs_shm_key_ + " on host");
  }
}


/** ****************************************************************************
***************************************************************************** */
cv::Mat TritonClient::getOutput(const TritonTensorMeta& om) {

    // get raw data shape
    std::vector<int64_t> shape;
    NI_CHECK_OK(inferResult_->Shape(om.name, &shape),
                "Failed to get inference server result shape for '" + om.name +"'");
    size_t ndim = shape.size();
    if (ndim < 2) { // force matrix for vector with single col?!
        ndim = 2;
        shape.push_back(1);
    }

    // get raw data pointer and size
    const uint8_t *ptrRaw;
    size_t cntRaw;
    if(!outputs_shm_key_.empty()){
      // calc some values since RawData() doesn't seem to work for shm
      ptrRaw = outputs_shm_ + om.shm_offset;
      cntRaw = inferInputs_[0]->Shape()[0] * om.byte_size;  // batch size * byte_size
      LOG_TRACE("output \"" << om.name << "\" uses shared memory starting at address " << std::hex << (void*)ptrRaw);
    }else{
      NI_CHECK_OK(inferResult_->RawData(om.name, &ptrRaw, &cntRaw),
            "Failed to get inference server result raw data  for '" + om.name +"'");
    }

    // calculate num elements from shape
    std::vector<int> iShape;
    int64 numElementsFromShape = 1;
    for (const auto &d: shape) {
        numElementsFromShape *= d;
        iShape.push_back(static_cast<int>(d));
    }

    // determine opencv type and calculate num elements from raw size and data type
    LOG_TRACE("Expecting " << numElementsFromShape << " elements in output buffer size: " << cntRaw << " with element size:" << om.element_byte_size);
    if (cntRaw / om.element_byte_size != numElementsFromShape) {
        std::stringstream ss("Shape ");
        ss << shape << " and data-type '" << om.type << "' are inconsistent with buffer size " << cntRaw;
        THROW_TRTISEXCEPTION(MPF_DETECTION_FAILED, ss.str());
    }

    return cv::Mat(ndim, iShape.data(), om.cvType, (void *) ptrRaw);
}


/** ****************************************************************************
 * prepare input data blob for a inferencing
***************************************************************************** */
void TritonClient::prepareInferRequestedOutputs(){

  for(auto &om : inferencer_->outputsMeta){
    nic::InferRequestedOutput *tmp;
    NI_CHECK_OK(nic::InferRequestedOutput::Create(&tmp, om.name),
      "unable to create requested output '" + om.name + "'");

    if(!outputs_shm_key_.empty()){
      NI_CHECK_OK(tmp->SetSharedMemory(outputs_shm_key_, om.byte_size * inferencer_->maxBatchSize , om.shm_offset),
        "unable to associate output \"" + om.name+ "\" with shared memory at offset" + std::to_string(om.shm_offset));
    }

    inferReqestedOutputs_.push_back(std::unique_ptr<const nic::InferRequestedOutput>(tmp));
  }

}

/** ****************************************************************************
 * prepare input data blob for a inferencing
***************************************************************************** */
void TritonClient::prepareInferInputs(){

    inferInputs_.clear();
    for(auto& im : inferencer_->inputsMeta){
      nic::InferInput *tmp;
      NI_CHECK_OK(nic::InferInput::Create(&tmp, im.name, im.shape, im.type),
                 "unable to create input '" + im.name + "'");
      inferInputs_.push_back(std::unique_ptr<nic::InferInput>(tmp));

    }
}

/** ****************************************************************************
 * set input data blob for a inferencing
***************************************************************************** */
void TritonClient::setInferInputsData(const std::vector<cv::Mat> &blobs){
    // LOG_TRACE("Preparing data for inferencing");
    // LOG_TRACE("blobs size: " << blobs.size());
    // LOG_TRACE("blobs[0] shape " << std::vector<int>(blobs[0].size.p, blobs[0].size.p + blobs[0].dims));
    // LOG_TRACE("inferencer: " << std::hex << (void*) inferencer_);
    // LOG_TRACE("inpusMeta size: " << inferencer_->inputsMeta.size());

    assert(("All inputs have to be specified" , blobs.size() == inferencer_->inputsMeta.size()));

    for(int i = 0; i < blobs.size(); ++i){

      // check batch size is ok for model
      int64_t inputBatchSize = *(blobs[i].size.p);
      if(inputBatchSize > inferencer_->maxBatchSize){
          THROW_TRTISEXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
          "input \"" + inferencer_->inputsMeta[i].name + "\" blob's batch dimension of "
          + std::to_string(inputBatchSize)
          + "is greater than the maximum of "+ std::to_string(inferencer_->maxBatchSize)
          + " supported by the model");
      }

      // check ocv matrix data is contrinuous in memory
      if(!blobs[i].isContinuous()){
        THROW_TRTISEXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
         "blob is not stored in continuous memory for conversion to inference client input \""
          + inferencer_->inputsMeta[i].name + "\".");
      }

      // clear out input
      NI_CHECK_OK(inferInputs_.at(i)->Reset(),
        "unable to reset input \"" + inferencer_->inputsMeta.at(i).name + "\" to receive new tensor data");

      // set input shape
      std::vector<int64_t> shape;
      shape.assign(blobs[i].size.p, blobs[i].size.p + blobs[i].dims);
      NI_CHECK_OK(inferInputs_[i]->SetShape(shape),
        "unable to set shape" +
        [&shape]{std::stringstream ss; ss << shape; return ss.str();}()
        + " for input \"" + inferencer_->inputsMeta[i].name + "\"");

      // set input data
      size_t num_bytes = blobs[i].total() * blobs[i].elemSize();
      if(!inputs_shm_key_.empty()){
        if(num_bytes <= inferencer_->inputsMeta[i].byte_size * inferencer_->maxBatchSize){

          LOG_TRACE("memcpy " << num_bytes << " bytes from blob of size " << shape << " to shm at address " << std::hex << (void*)(inputs_shm_ + inferencer_->inputsMeta[i].shm_offset));

          std::memcpy(inputs_shm_ + inferencer_->inputsMeta[i].shm_offset, blobs[i].data , num_bytes);

          NI_CHECK_OK(inferInputs_.at(i)->SetSharedMemory(inputs_shm_key_, num_bytes , inferencer_->inputsMeta[i].shm_offset),
            "unable to associate input \"" + inferencer_->inputsMeta[i].name+ "\" with shared memory at offset" + std::to_string(inferencer_->inputsMeta[i].shm_offset));

        }else{
          THROW_TRTISEXCEPTION(MPF_OTHER_DETECTION_ERROR_TYPE,
            "attempted to set shared input memory buffer with "
             + std::to_string(num_bytes) + " but there is only room for "
             + std::to_string(inferencer_->inputsMeta[i].byte_size * inferencer_->maxBatchSize) + " bytes.");
        }
      }else{
        NI_CHECK_OK(inferInputs_.at(i)->AppendRaw(blobs[i].data, num_bytes),
              "unable to set data for \"" + inferencer_->inputsMeta[i].name + "\"");
      }

    }


}

/** ****************************************************************************
 * inference input specified in vector of 4D data blobs
***************************************************************************** */
void TritonClient::infer(const std::vector<cv::Mat> &inputBlobs){

  setInferInputsData(inputBlobs);

  nic::InferResult* tmp;
  NI_CHECK_OK( grpc_->Infer(&tmp, inferencer_->inferOptions, getRaw(inferInputs_), getRaw(inferReqestedOutputs_) ),
    "unable to inference on server");
  inferResult_.reset(tmp);
}


/** ****************************************************************************
 * inference input specified in vector of 4D data blobs
***************************************************************************** */
void TritonClient::inferAsync(const std::vector<cv::Mat> &inputBlobs, CallbackFunc inferencerLambda){
  LOG_TRACE("inferAsync start");
  setInferInputsData(inputBlobs);

  NI_CHECK_OK(
    grpc_->AsyncInfer(
    [inferencerLambda, this](nic::InferResult* tmp){
      LOG_TRACE("inferAsync lambda start");
      inferResult_.reset(tmp);
      inferencerLambda();
      LOG_TRACE("inferAsync lambda end");
    },
    inferencer_->inferOptions,
    getRaw(inferInputs_),
    getRaw(inferReqestedOutputs_)),
    "unable to async inference on server");
    LOG_TRACE("inferAsync end");
}


/** ****************************************************************************
 * inference input specified in vector of 4D data blobs
***************************************************************************** */
// void TritonClient::AsyncInfer(const std::vector<cv::Mat> &inputBlobs,
//                          nic::InferenceServerClient::OnCompleteFn callback){

//   setInferInputsData(inputBlobs);

//   nic::InferResult* tmp;
//   grpc_->AsyncInfer(
//   NI_CHECK_OK( grpc_->Infer(&tmp, inferencer_->inferOptions, getRaw(inferInputs_), getRaw(inferReqestedOutputs_) ),
//     "unable to inference on server");

//   inferResult_.reset(tmp);
// }

/** ****************************************************************************
 * Destructor to cleanup share memory regions
***************************************************************************** */
TritonClient::~TritonClient(){
  LOG_TRACE("~TritonClient " << id);
  removeShmRegions();

}

/** ****************************************************************************
 * Constructor for nested Client_ class encapsulating nv grpc client
***************************************************************************** */
TritonClient::TritonClient(
  const int id,
  const Config& cfg,
  const TritonInferencer *inferencer)
 : id(id)
 , inferencer_(inferencer)
 , inputs_byte_size_(inferencer->inputsMeta.back().shm_offset + inferencer->inputsMeta.back().byte_size * inferencer->maxBatchSize)
 , outputs_byte_size_(inferencer->outputsMeta.back().shm_offset + inferencer->outputsMeta.back().byte_size * inferencer->maxBatchSize)
 , inputs_shm_key_(cfg.trtisUseShm ? shm_key_prefix() + "_" + std::to_string(id) + "_inputs" : "")
 , outputs_shm_key_(cfg.trtisUseShm ? shm_key_prefix() + "_" + std::to_string(id) + "_outputs" : "")
{
  NI_CHECK_OK(nic::InferenceServerGrpcClient::Create(
    &grpc_, inferencer->serverUrl, cfg.trtisVerboseClient, cfg.trtisUseSSL, inferencer->sslOptions),
      "unable to create TRTIS inference client for \"" + cfg.trtisServer + "\"");

  setupShmRegions();
  prepareInferInputs();
  prepareInferRequestedOutputs();

}