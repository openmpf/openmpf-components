/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2024 The MITRE Corporation                                       *
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

#ifndef OPENMPF_COMPONENTS_TRITON_TENSOR_META_H
#define OPENMPF_COMPONENTS_TRITON_TENSOR_META_H

#include <string>
#include <vector>

#include <grpc_client.h>

class TritonTensorMeta {
public:
    const std::string name;
    const std::string type;
    const size_t cvType;
    const std::vector<int64_t> shape;
    const size_t element_count;
    const size_t element_byte_size;
    const size_t byte_size;
    const size_t shm_offset;

    TritonTensorMeta(const inference::ModelInput &mi, size_t shm_offset);

    TritonTensorMeta(const inference::ModelOutput &mo, size_t shm_offset);
};

#endif // OPENMPF_COMPONENTS_TRITON_TENSOR_META_H
