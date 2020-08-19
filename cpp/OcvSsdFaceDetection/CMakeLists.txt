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
project(ocv-ssd-face-detection)

set(CMAKE_CXX_STANDARD 11)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)    # build release version with debug symbols
set(CMAKE_BUILD_TYPE Debug)            # build debug version

add_definitions(-Wfatal-errors)
include(../ComponentSetup.cmake)

# Specify dlib source directory here.
# Otherwise, disable the following three lines if using find_package(dlib) instead.
#set(DLIB_DIR /apps/source/dlib-sources/dlib-19.19/dlib)
#include(${DLIB_DIR}/cmake)
#set_property(TARGET dlib PROPERTY POSITION_INDEPENDENT_CODE ON)

# Uncomment if not specifying dlib source directory.
# Warning: may not work with dlib release installation.
# find_package(dlib REQUIRED)

find_package(OpenCV
  REQUIRED
  COMPONENTS
     opencv_dnn
     opencv_video
     opencv_face
     opencv_tracking
)
if(${OpenCV_VERSION} VERSION_LESS 3.4.7)
  message(FATAL_ERROR "OpenCV ${OpenCV_VERSION} is below minimum 3.4.7")
endif()

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)

IF(CUDA_FOUND)
    MESSAGE( STATUS "I found CUDA !" )
    SET(HAVE_CUDA ${CUDA_FOUND} CACHE BOOL "Set to TRUE if CUDA is found, FALSE otherwise")
    set (CMAKE_CXX_FLAGS "-DHAVE_CUDA=1")
ENDIF(CUDA_FOUND)

find_package(GTest)
if (${GTEST_FOUND})
    message("-- found GTest, configuring tests")
    enable_testing()
    include(Dart)
endif()

set(OCV_SSD_FACE_DETECTION_SOURCE_FILES
        JobConfig.cpp
        DetectionLocation.cpp
        Track.cpp
        KFTracker.cpp
        OcvSsdFaceDetection.cpp)

add_library(mpfOcvSsdFaceDetection SHARED ${OCV_SSD_FACE_DETECTION_SOURCE_FILES})
target_link_libraries(mpfOcvSsdFaceDetection
                      mpfComponentInterface
                      mpfDetectionComponentApi
                      mpfComponentUtils
                      ${OpenCV_LIBS}
                      dlib)

configure_mpf_component(OcvSsdFaceDetection TARGETS mpfOcvSsdFaceDetection)

add_subdirectory(test)

add_executable(       sample_ocv_ssd_face_detector sample_ocv_ssd_face_detector.cpp)
target_link_libraries(sample_ocv_ssd_face_detector mpfOcvSsdFaceDetection)

# update configs if they change
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  plugin-files/config/mpfOcvSsdFaceDetection.ini
  plugin-files/config/Log4cxxConfig.xml
  plugin-files/descriptor/descriptor.json)