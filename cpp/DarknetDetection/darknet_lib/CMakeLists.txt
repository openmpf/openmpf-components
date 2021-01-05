#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

cmake_minimum_required(VERSION 3.6)

project(darknet_lib)

set(CMAKE_C_STANDARD 11)

set(DARKNET_LIB_SRC_FILES
    include/darknet.h
    src/gemm.c
    src/utils.c
    src/cuda.c
    src/deconvolutional_layer.c
    src/convolutional_layer.c
    src/list.c
    src/image.c
    src/activations.c
    src/im2col.c
    src/col2im.c
    src/blas.c
    src/crop_layer.c
    src/dropout_layer.c
    src/maxpool_layer.c
    src/softmax_layer.c
    src/data.c
    src/matrix.c
    src/network.c
    src/connected_layer.c
    src/cost_layer.c
    src/parser.c
    src/option_list.c
    src/detection_layer.c
    src/route_layer.c
    src/upsample_layer.c
    src/box.c
    src/normalization_layer.c
    src/avgpool_layer.c
    src/layer.c
    src/local_layer.c
    src/shortcut_layer.c
    src/logistic_layer.c
    src/activation_layer.c
    src/rnn_layer.c
    src/gru_layer.c
    src/crnn_layer.c
    src/demo.c
    src/batchnorm_layer.c
    src/region_layer.c
    src/reorg_layer.c
    src/tree.c
    src/lstm_layer.c
    src/l2norm_layer.c
    src/yolo_layer.c
    src/iseg_layer.c)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Darknet fails to compile when all optimizations are disabled, which is the default for debug builds.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # -Og enables optimizations that do not interfere with debugging.
    add_compile_options(-Og)
else()
    add_compile_options(-Ofast)
endif()

add_library(darknet_lib ${DARKNET_LIB_SRC_FILES})
target_include_directories(darknet_lib PUBLIC include)
target_link_libraries(darknet_lib m pthread)


if (DARKNET_BUILD_CUDA)
    SET(DARKNET_CUDA_SRC_FILES
        src/activation_kernels.cu
        src/avgpool_layer_kernels.cu
        src/blas_kernels.cu
        src/col2im_kernels.cu
        src/convolutional_kernels.cu
        src/crop_layer_kernels.cu
        src/deconvolutional_kernels.cu
        src/dropout_layer_kernels.cu
        src/im2col_kernels.cu
        src/maxpool_layer_kernels.cu
        )

    find_package(CUDA REQUIRED)
    set(CUDA_VERBOSE_BUILD ON)

    # By default, we build Darknet for maximum portability across GPU
    # architectures. This means that we create PTX code for the minimum compute
    # capability only. For the darknet component, we have not seen a performance
    # benefit to creating to ELF code in addition to PTX code. However, for other
    # components, this may not be the case.
    set(CUDA_NVCC_FLAGS --compiler-options -fPIC -gencode arch=compute_30,code=compute_30)
    message(STATUS "CUDA_NVCC_FLAGS = ${CUDA_NVCC_FLAGS}")

    include_directories(include)

    CUDA_ADD_LIBRARY(darknet_lib_cuda ${DARKNET_LIB_SRC_FILES} ${DARKNET_CUDA_SRC_FILES})
    CUDA_ADD_CUBLAS_TO_TARGET(darknet_lib_cuda)
    target_compile_definitions(darknet_lib_cuda PUBLIC -DGPU)

    target_include_directories(darknet_lib_cuda PUBLIC include ${CUDA_INCLUDE_DIRS})
    target_link_libraries(darknet_lib_cuda m pthread ${CUDA_curand_LIBRARY})
endif()

