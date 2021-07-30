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
project(ocv-yolo-detection)

set(CMAKE_CXX_STANDARD 11)
#ADD_DEFINITIONS("-std=c++11")
add_definitions(-Wfatal-errors)

set(CMAKE_BUILD_TYPE RelWithDebInfo)  # build release version with debug symbols
#set(CMAKE_BUILD_TYPE Release)         # build release version
#set(CMAKE_BUILD_TYPE Debug)           # build debug version

include(../ComponentSetup.cmake)


# Uncomment if not specifying dlib source directory.
# Warning: may not work with dlib release installation.
#find_package(dlib REQUIRED)

set(USE_AVX_INSTRUCTIONS ON)
set(DLIB_USE_CUDA OFF)
set(DLIB_DIR /apps/source/dlib-sources/dlib-19.20/dlib)
include(${DLIB_DIR}/cmake)
set_property(TARGET dlib PROPERTY POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV 4.5.0 EXACT REQUIRED COMPONENTS opencv_dnn opencv_videoio opencv_tracking)

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)


find_package(gRPC REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)


set(OCV_YOLO_DETECTION_SOURCE_FILES
        TritonExceptionMacros.h
        TritonInferencer.cpp TritonInferencer.h
        TritonClient.cpp TritonClient.h
        TritonTensorMeta.cpp TritonTensorMeta.h
        shm_utils.cc shm_utils.h
        util.cpp util.h
        Config.cpp Config.h
        Frame.h
        DetectionLocation.cpp DetectionLocation.h
        Track.cpp Track.h
        KFTracker.cpp KFTracker.h
        OcvYoloDetection.cpp OcvYoloDetection.h
        ocv_phasecorr.cpp ocv_phasecorr.h
        YoloNetwork.cpp YoloNetwork.h
        WhitelistFilter.cpp WhitelistFilter.h)

add_library(mpfOcvYoloDetection SHARED ${OCV_YOLO_DETECTION_SOURCE_FILES})
target_link_libraries(mpfOcvYoloDetection
                      mpfComponentInterface
                      mpfDetectionComponentApi
                      mpfComponentUtils
                      TritonClient::grpcclient_static
                      ${OpenCV_LIBS}
                      dlib
                      explain)

configure_mpf_component(OcvYoloDetection TARGETS mpfOcvYoloDetection)

add_subdirectory(test)

#add_executable(sample_ocv_yolo_detector sample_ocv_yolo_detector.cpp)
#target_link_libraries(sample_ocv_yolo_detector mpfOcvYoloDetection})
