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

#include "../Config.h"

#include "TritonUtils.h"
#include "TritonTensorMeta.h"

using namespace MPF::COMPONENT;

namespace {

    std::string tritonType2Str(const inference::DataType &dt) {
        switch (dt) {
            case inference::TYPE_INVALID:
                return "INVALID";
            case inference::TYPE_BOOL:
                return "BOOL";
            case inference::TYPE_UINT8:
                return "UINT8";
            case inference::TYPE_UINT16:
                return "UINT16";
            case inference::TYPE_UINT32:
                return "UINT32";
            case inference::TYPE_UINT64:
                return "UINT64";
            case inference::TYPE_INT8:
                return "INT8";
            case inference::TYPE_INT16:
                return "INT16";
            case inference::TYPE_INT32:
                return "INT32";
            case inference::TYPE_INT64:
                return "INT64";
            case inference::TYPE_FP16:
                return "FP16";
            case inference::TYPE_FP32:
                return "FP32";
            case inference::TYPE_FP64:
                return "FP64";
            case inference::TYPE_STRING:
                return "STRING";
            default:
                return "UNKNOWN";
        }
    }

    size_t tritonTypeSizeOf(const inference::DataType &dt) {
        switch (dt) {
            case inference::TYPE_BOOL:
                return sizeof(bool);
            case inference::TYPE_UINT8:
                return sizeof(u_int8_t);
            case inference::TYPE_UINT16:
                return sizeof(u_int16_t);
            case inference::TYPE_UINT32:
                return sizeof(u_int32_t);
            case inference::TYPE_UINT64:
                return sizeof(u_int64_t);
            case inference::TYPE_INT8:
                return sizeof(int8_t);
            case inference::TYPE_INT16:
                return sizeof(int16_t);
            case inference::TYPE_INT32:
                return sizeof(int32_t);
            case inference::TYPE_INT64:
                return sizeof(int64_t);
            case inference::TYPE_FP16:
                return sizeof(int16_t);
            case inference::TYPE_FP32:
                return sizeof(float);
            case inference::TYPE_FP64:
                return sizeof(double);
            default:
                return -1;
        }
    }

    size_t tritonTypeName2OCVType(const inference::DataType &dt) {
        switch (dt) {
            case inference::TYPE_FP32:
                return CV_32FC1;
            case inference::TYPE_UINT8:
                return CV_16UC1;
            case inference::TYPE_INT8:
                return CV_8SC1;
            case inference::TYPE_INT16:
                return CV_16SC1;
            case inference::TYPE_INT32:
                return CV_32SC1;
            case inference::TYPE_FP64:
                return CV_64FC1;
            default:// OpenCV does not support these types
                //   UINT32, UINT64, INT64, FP16, BOOL, BYTES:
            THROW_TRITON_EXCEPTION(MPF_DETECTION_FAILED,
                                   "Unsupported inference::DataType = "
                                   + std::to_string(dt)
                                   + " in cv:Mat conversion.");
        }
    }

}

TritonTensorMeta::TritonTensorMeta(const inference::ModelInput &mi, size_t shm_offset)
        : name(mi.name()), type(tritonType2Str(mi.data_type())), cvType(tritonTypeName2OCVType(mi.data_type())),
          shape(mi.dims().begin(), mi.dims().end()),
          element_count(std::accumulate(mi.dims().begin(), mi.dims().end(), 1, std::multiplies<int64_t>())),
          element_byte_size(tritonTypeSizeOf(mi.data_type())), byte_size(element_count * element_byte_size),
          shm_offset(shm_offset) {}

TritonTensorMeta::TritonTensorMeta(const inference::ModelOutput &mo, size_t shm_offset)
        : name(mo.name()), type(tritonType2Str(mo.data_type())), cvType(tritonTypeName2OCVType(mo.data_type())),
          shape(mo.dims().begin(), mo.dims().end()),
          element_count(std::accumulate(mo.dims().begin(), mo.dims().end(), 1, std::multiplies<int64_t>())),
          element_byte_size(tritonTypeSizeOf(mo.data_type())), byte_size(element_count * element_byte_size),
          shm_offset(shm_offset) {}
