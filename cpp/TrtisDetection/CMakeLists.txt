#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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
project(caffe-detection)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-Wfatal-errors)

include(../ComponentSetup.cmake)

find_package(OpenCV 3.4.7 EXACT REQUIRED PATHS /opt/opencv-3.4.7
    COMPONENTS opencv_core)
#find_package(Protobuf 3.11.2 REQUIRED)
find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)
find_package(request REQUIRED)
find_package(CURL REQUIRED)

include_directories(${PROTOBUF_INCLUDE_DIR} ${CURL_INCLUDE_DIR})
set(TRTIS_DETECTION_SOURCE_FILES TrtisDetection.cpp TrtisDetection.hpp)

add_library(trtisDetection SHARED ${TRTIS_DETECTION_SOURCE_FILES})
target_link_libraries(trtisDetection request mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils ${OpenCV_LIBS} ${PROTOBUF_LIBRARY} ${CURL_LIBRARIES})

configure_mpf_component(TrtisDetection TARGETS trtisDetection)

enable_testing()
add_subdirectory(test)

# Build sample executable
add_executable(sample_trtis_detector sample_trtis_detector.cpp)
target_link_libraries(sample_trtis_detector trtisDetection)
