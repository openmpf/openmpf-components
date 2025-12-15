#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
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
project(trtis-detection)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-Wfatal-errors)

set(CMAKE_BUILD_TYPE RelWithDebInfo)  # build release version with debug symbols
# set(CMAKE_BUILD_TYPE Debug)           # build debug version

include(../ComponentSetup.cmake)

find_package(OpenCV 4.13.0 EXACT REQUIRED PATHS /opt/opencv-4.13.0
    COMPONENTS opencv_core)
find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(CURL REQUIRED)

set(BUILD_SHARED_LIBS ON)  # make AWS use shared linking
find_package(AWSSDK REQUIRED COMPONENTS core s3)

include_directories(${PROTOBUF_INCLUDE_DIR} ${CURL_INCLUDE_DIR})
set(TRTIS_DETECTION_SOURCE_FILES
        TrtisDetection.cpp TrtisDetection.h IFeatureStorage.h EncodeFeatureStorage.h EncodeFeatureStorage.cpp
        S3FeatureStorage.cpp S3FeatureStorage.h S3StorageUtil.cpp S3StorageUtil.h base64.h uri.h)

add_library(mpfTrtisDetection SHARED ${TRTIS_DETECTION_SOURCE_FILES})
target_link_libraries(mpfTrtisDetection request mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
        ${OpenCV_LIBS} ${PROTOBUF_LIBRARY} ${CURL_LIBRARIES} ${AWSSDK_LINK_LIBRARIES})

configure_mpf_component(TrtisDetection TARGETS mpfTrtisDetection)

add_subdirectory(test)

# Build sample executable
add_executable(sample_trtis_detector sample_trtis_detector.cpp)
target_link_libraries(sample_trtis_detector mpfTrtisDetection)
