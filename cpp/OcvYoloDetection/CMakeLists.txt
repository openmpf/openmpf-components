#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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
project(ocv-yolo-detection)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-Wfatal-errors)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)  # build release version with debug symbols
set(CMAKE_BUILD_TYPE Release)          # build release version
#set(CMAKE_BUILD_TYPE Debug)           # build debug version

# Uncomment to enable line numbers in log messages and exceptions. Should only be used when debugging, not production.
# add_compile_definitions(DEBUG_LINE_NUMBERS)

include(../ComponentSetup.cmake)

if ($ENV{TRITON_SUPPORT})
    message("Building with Triton support")
else ()
    message("Building without Triton support")
endif ()

# Uncomment if not specifying dlib source directory.
# Warning: may not work with dlib release installation.
#find_package(dlib REQUIRED)

set(USE_AVX_INSTRUCTIONS ON)
set(DLIB_USE_CUDA OFF)
set(DLIB_DIR /apps/source/dlib-sources/dlib-19.20/dlib)
include(${DLIB_DIR}/cmake)
set_property(TARGET dlib PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV 4.9.0 EXACT REQUIRED COMPONENTS opencv_dnn opencv_videoio opencv_tracking)

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)

if ($ENV{TRITON_SUPPORT})
    include_directories(/opt/triton/include)
    find_library(grpc_client grpcclient REQUIRED PATHS /opt/triton/lib)
    find_library(protobuf protobuf REQUIRED PATHS /opt/triton/lib)
endif ()


set(COMMON_OCV_YOLO_DETECTION_SOURCE_FILES
        util.cpp util.h
        Config.cpp Config.h
        Frame.cpp Frame.h
        DetectionLocation.cpp DetectionLocation.h
        Track.cpp Track.h
        KFTracker.cpp KFTracker.h
        OcvYoloDetection.cpp OcvYoloDetection.h
        ocv_phasecorr.cpp ocv_phasecorr.h
        AllowListFilter.cpp AllowListFilter.h
        yolo_network/BaseYoloNetworkImpl.cpp yolo_network/BaseYoloNetworkImpl.h)

set(LOCAL_OCV_YOLO_DETECTION_SOURCE_FILES
        yolo_network/LocalYoloNetworkImpl.cpp yolo_network/YoloNetwork.h)

set(TRITON_OCV_YOLO_DETECTION_SOURCE_FILES
        yolo_network/TritonYoloNetworkImpl.cpp yolo_network/YoloNetwork.h
        triton/TritonUtils.h
        triton/TritonInferencer.cpp triton/TritonInferencer.h
        triton/TritonClient.cpp triton/TritonClient.h
        triton/TritonTensorMeta.cpp triton/TritonTensorMeta.h
        triton/shm_utils.cc triton/shm_utils.h)

if ($ENV{TRITON_SUPPORT})
    add_library(mpfOcvYoloDetection SHARED ${COMMON_OCV_YOLO_DETECTION_SOURCE_FILES} ${TRITON_OCV_YOLO_DETECTION_SOURCE_FILES})
    target_link_libraries(mpfOcvYoloDetection
            mpfComponentInterface
            mpfDetectionComponentApi
            mpfComponentUtils
            ${grpc_client}
            ${protobuf}
            ${OpenCV_LIBS}
            dlib)
else ()
    add_library(mpfOcvYoloDetection SHARED ${COMMON_OCV_YOLO_DETECTION_SOURCE_FILES} ${LOCAL_OCV_YOLO_DETECTION_SOURCE_FILES})
    target_link_libraries(mpfOcvYoloDetection
            mpfComponentInterface
            mpfDetectionComponentApi
            mpfComponentUtils
            ${OpenCV_LIBS}
            dlib)
endif ()

configure_mpf_component(OcvYoloDetection TARGETS mpfOcvYoloDetection)

add_subdirectory(test)
