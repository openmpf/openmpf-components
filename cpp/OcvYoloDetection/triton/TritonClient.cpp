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

#include <triton/core/tritonbackend.h>
#include <grpc_client.h>
#include <MPFDetectionException.h>

#include "shm_utils.h"

#include "../util.h"
#include "../Config.h"
#include "../Frame.h"

#include "TritonTensorMeta.h"
#include "TritonUtils.h"
#include "TritonClient.h"
#include "TritonInferencer.h"

using namespace MPF::COMPONENT;

// TODO: Seems dangerous.
template<typename T>
std::vector<T*> getRaw(const std::vector<std::unique_ptr<T>> &v){
  std::vector<T*> raw;
  for(const auto& i : v){
    raw.push_back(i.get());
  }
  return raw;
}


const std::string& TritonClient::shm_key_prefix(){
  static std::string prefix = "\\" + hostname();
  return prefix;
}


cv::Mat TritonClient::getOutput(const TritonTensorMeta& om) {

    // get raw data shape
    std::vector<int64_t> shape;
    tritonCheckOk(inferResult_->Shape(om.name, &shape),
                MPF_DETECTION_FAILED,
                "Failed to get inference server result shape for \"" + om.name + "\".");
    size_t ndim = shape.size();
    if (ndim < 2) { // force matrix for vector with single col?!
        ndim = 2;
        shape.push_back(1);
    }

    // get raw data pointer and size
    const uint8_t *ptrRaw;
    size_t cntRaw;
    if(usingShmOutput()){
      // calc some values since RawData() doesn't seem to work for shm
      ptrRaw = outputs_shm_ + om.shm_offset;
      cntRaw = inferInputs_[0]->Shape()[0] * om.byte_size;  // batch size * byte_size
      LOG_TRACE("Output \"" << om.name << "\" uses shared memory starting at address "
                << std::hex << (void*)ptrRaw);
    }else{
      tritonCheckOk(inferResult_->RawData(om.name, &ptrRaw, &cntRaw),
            MPF_DETECTION_FAILED,
            "Failed to get inference server result raw data for \"" + om.name + "\".");
    }

    // calculate num elements from shape
    std::vector<int> iShape;
    int64 numElementsFromShape = 1;
    for (const auto &d: shape) {
        numElementsFromShape *= d;
        iShape.push_back(static_cast<int>(d));
    }

    // determine opencv type and calculate num elements from raw size and data type
    LOG_TRACE("Expecting " << numElementsFromShape << " elements in output buffer size: "
              << cntRaw << " with element size: " << om.element_byte_size);
    if (cntRaw / om.element_byte_size != numElementsFromShape) {
        std::stringstream ss("Shape ");
        ss << shape << " and data-type \"" << om.type
           << "\" are inconsistent with buffer size " << cntRaw << ".";
        throw createTritonException(MPF_DETECTION_FAILED, ss.str());
    }

    return cv::Mat(ndim, iShape.data(), om.cvType, (void *) ptrRaw);
}


void TritonClient::prepareInferRequestedOutputs(){

  for(auto &om : inferencer_->outputsMeta){
    triton::client::InferRequestedOutput *tmp;
    tritonCheckOk(triton::client::InferRequestedOutput::Create(&tmp, om.name),
      MPF_DETECTION_FAILED,
      "Unable to create requested output \"" + om.name + "\".");

    if(usingShmOutput()){
      tritonCheckOk(tmp->SetSharedMemory(outputs_shm_key,
                                       om.byte_size * inferencer_->maxBatchSize(),
                                       om.shm_offset),
       MPF_MEMORY_ALLOCATION_FAILED,
       "Unable to associate output \"" + om.name+ "\" with shared memory at offset "
       + std::to_string(om.shm_offset));
    }

    inferRequestedOutputs_.push_back(
      std::unique_ptr<const triton::client::InferRequestedOutput>(tmp));
  }
}


void TritonClient::prepareInferInputs(){

    inferInputs_.clear();
    for(auto& im : inferencer_->inputsMeta){
      triton::client::InferInput *tmp;
      tritonCheckOk(triton::client::InferInput::Create(&tmp, im.name, im.shape, im.type),
                  MPF_DETECTION_FAILED,
                  "Unable to create input \"" + im.name + "\".");
      inferInputs_.push_back(std::unique_ptr<triton::client::InferInput>(tmp));
    }
}


void TritonClient::setInferInputsData(const std::vector<cv::Mat> &blobs){

    assert(("All model inputs have to be specified." , blobs.size() == inferencer_->inputsMeta.size()));

    for(int i = 0; i < blobs.size(); ++i){

      // check batch size is ok for model
      int64_t inputBatchSize = *(blobs[i].size.p);
      if(inputBatchSize > inferencer_->maxBatchSize()){
          throw createTritonException(MPF_INVALID_PROPERTY,
                                      "Input \"" + inferencer_->inputsMeta[i].name + "\" blob's batch dimension of "
                                      + std::to_string(inputBatchSize)
                                      + " is greater than the maximum of " + std::to_string(inferencer_->maxBatchSize())
                                      + " supported by the model.");
      }

      // check ocv matrix data is contiguous in memory
      if(!blobs[i].isContinuous()){
          throw createTritonException(MPF_MEMORY_ALLOCATION_FAILED,
                                      "Blob is not stored in continuous memory for conversion to inference client input \""
                                      + inferencer_->inputsMeta[i].name + "\".");
      }

      // clear out input
        tritonCheckOk(inferInputs_.at(i)->Reset(),
                      MPF_DETECTION_FAILED,
                      "Unable to reset input \"" + inferencer_->inputsMeta.at(i).name
                      + "\" to receive new tensor data.");

      // set input shape
      std::vector<int64_t> shape;
      shape.assign(blobs[i].size.p, blobs[i].size.p + blobs[i].dims);
      tritonCheckOk(inferInputs_[i]->SetShape(shape),
        MPF_DETECTION_FAILED,
        "Unable to set shape " +
        [&shape]{std::stringstream ss; ss << shape; return ss.str();}()
        + " for input \"" + inferencer_->inputsMeta[i].name + "\".");

      // set input data
      size_t num_bytes = blobs[i].total() * blobs[i].elemSize();
      if (usingShmInput()){
        if (num_bytes <= inferencer_->inputsMeta[i].byte_size * inferencer_->maxBatchSize()) {
          std::memcpy(inputs_shm_ + inferencer_->inputsMeta[i].shm_offset, blobs[i].data , num_bytes);
            tritonCheckOk(inferInputs_.at(i)->SetSharedMemory(inputs_shm_key,
                                                                     num_bytes,
                                                                     inferencer_->inputsMeta[i].shm_offset),
                          MPF_MEMORY_ALLOCATION_FAILED,
                          "Unable to associate input \"" + inferencer_->inputsMeta[i].name
                          + "\" with shared memory at offset "
                          + std::to_string(inferencer_->inputsMeta[i].shm_offset) + ".");
        } else {
            throw createTritonException(MPF_MEMORY_ALLOCATION_FAILED,
                                        "Attempted to set shared input memory buffer with "
                                        + std::to_string(num_bytes) + " bytes but there is only room for "
                                        + std::to_string(
                                                inferencer_->inputsMeta[i].byte_size * inferencer_->maxBatchSize()) +
                                        " bytes.");
        }
      } else {
        tritonCheckOk(inferInputs_.at(i)->AppendRaw(blobs[i].data, num_bytes),
                    MPF_MEMORY_ALLOCATION_FAILED,
                    "Unable to set data for \"" + inferencer_->inputsMeta[i].name + "\".");
      }
    }
}


void TritonClient::inferAsync(int inferInputIdx, const cv::Mat& blob, CallbackFunc inferencerLambda){

  // clear out input
    tritonCheckOk(inferInputs_.at(inferInputIdx)->Reset(),
                  MPF_DETECTION_FAILED,
                  "Unable to reset input \"" + inferencer_->inputsMeta.at(inferInputIdx).name
                  + "\" to receive new tensor data.");

  // set input shape
  std::vector<int64_t> shape;
  shape.assign(blob.size.p, blob.size.p + blob.dims);
  if(inferInputs_[inferInputIdx]->Shape()[0] != shape[0]){
    tritonCheckOk(inferInputs_[inferInputIdx]->SetShape(shape),
      MPF_DETECTION_FAILED,
      "Unable to set shape" +
      [&shape]{std::stringstream ss; ss << shape; return ss.str();}()
      + " for input \"" + inferencer_->inputsMeta[inferInputIdx].name + "\".");
  }
  // set input data
  size_t numBytes = blob.total() * blob.elemSize();
  if(usingShmInput()){
    tritonCheckOk(inferInputs_.at(inferInputIdx)->SetSharedMemory(
      inputs_shm_key, numBytes, inferencer_->inputsMeta[inferInputIdx].shm_offset),
      MPF_MEMORY_ALLOCATION_FAILED,
      "Unable to associate input \"" + inferencer_->inputsMeta[inferInputIdx].name
      + "\" with shared memory at offset "
      + std::to_string(inferencer_->inputsMeta[inferInputIdx].shm_offset) + ".");
  }else{
    tritonCheckOk(inferInputs_.at(inferInputIdx)->AppendRaw(blob.data, numBytes),
                MPF_MEMORY_ALLOCATION_FAILED,
                "Unable to set data for \"" + inferencer_->inputsMeta[inferInputIdx].name + "\".");
  }

  inferAsync_(inferencerLambda);
}


void TritonClient::inferAsync_(CallbackFunc inferencerLambda){
  tritonCheckOk(
      grpc_->AsyncInfer(
      [inferencerLambda, this](triton::client::InferResult* tmp) {
        inferResult_.reset(tmp);
        inferencerLambda();
      },
      inferencer_->inferOptions(),
      getRaw(inferInputs_),
      getRaw(inferRequestedOutputs_)),
      MPF_DETECTION_FAILED,
      "Unable to perform async inference on server.");
}


void TritonClient::setupShmRegion(const std::string shm_key, const size_t byte_size, uint8_t* &shm_addr){

    int shm_fd;
    tritonCheckOk(triton::client::CreateSharedMemoryRegion(shm_key, byte_size, &shm_fd),
      MPF_MEMORY_ALLOCATION_FAILED,
      "Unable to create shared memory region " + shm_key + " on host.");
    tritonCheckOk(triton::client::MapSharedMemory(shm_fd, 0, byte_size, (void**) &shm_addr),
      MPF_MEMORY_ALLOCATION_FAILED,
      "Unable to map shared memory region " + shm_key + " to client address space.");
    tritonCheckOk(triton::client::CloseSharedMemory(shm_fd),
      MPF_MEMORY_ALLOCATION_FAILED,
      "Failed to close shared memory region " + shm_key + " on host.");
    tritonCheckOk(grpc_->RegisterSystemSharedMemory(shm_key, shm_key, byte_size),
      MPF_MEMORY_ALLOCATION_FAILED,
      "Unable to register " + shm_key + " shared memory with Triton inference server " + inferencer_->serverUrl() + ".");

    LOG_TRACE("Registered shared memory with key " << shm_key << " of size " << byte_size << " bytes at address " << std::hex << (void*) shm_addr);
}


void TritonClient::removeShmRegion(const std::string shm_key, const size_t byte_size, uint8_t* shm_addr){

  LOG_TRACE("Removing shared memory with key " << shm_key << " of size " << byte_size << " bytes at address " << std::hex << (void*) shm_addr);

  tritonCheckOk(grpc_->UnregisterSystemSharedMemory(shm_key),
    MPF_MEMORY_ALLOCATION_FAILED,
    "Unable to unregister shared memory region " + shm_key + " from Triton inference server " + inferencer_->serverUrl() + ".");
  tritonCheckOk(triton::client::UnmapSharedMemory((void*) shm_addr, byte_size),
    MPF_MEMORY_ALLOCATION_FAILED,
    "Unable to unmap shared memory region " + shm_key + " from client address space.");
  tritonCheckOk(triton::client::UnlinkSharedMemoryRegion(shm_key),
    MPF_MEMORY_ALLOCATION_FAILED,
    "Unable to remove shared memory region " + shm_key + " on host.");
}


TritonClient::~TritonClient(){
  LOG_TRACE("~TritonClient " << id);
  if(usingShmInput()){
    removeShmRegion(inputs_shm_key,  inputs_byte_size,  inputs_shm_);
  }
  if(usingShmOutput()){
    removeShmRegion(outputs_shm_key, outputs_byte_size, outputs_shm_);
  }
}


TritonClient::TritonClient(
  const int id,
  const TritonInferencer *inferencer)
 : id(id)
 , inferencer_(inferencer)
 , inputs_byte_size(inferencer->inputsMeta.back().shm_offset
                   + inferencer->inputsMeta.back().byte_size * inferencer->maxBatchSize())
 , outputs_byte_size(inferencer->outputsMeta.back().shm_offset
                    + inferencer->outputsMeta.back().byte_size * inferencer->maxBatchSize())
 , inputs_shm_key(inferencer->useShm() ? shm_key_prefix() + "_" + std::to_string(id) + "_inputs" : "")
 , outputs_shm_key(inferencer->useShm() ? shm_key_prefix() + "_" + std::to_string(id) + "_outputs" : "")
{
  tritonCheckOk(triton::client::InferenceServerGrpcClient::Create(
    &grpc_,
    inferencer->serverUrl(),
    inferencer->verboseClient(),
    inferencer->useSSL(),
    inferencer->sslOptions()),
      MPF_NETWORK_ERROR,
      "Unable to create Triton inference client for " + inferencer->serverUrl() + ".");

  if(usingShmInput()){
    setupShmRegion(inputs_shm_key,  inputs_byte_size,  inputs_shm_);
  }
  if(usingShmOutput()){
    setupShmRegion(outputs_shm_key, outputs_byte_size, outputs_shm_);
  }
  prepareInferInputs();
  prepareInferRequestedOutputs();
}