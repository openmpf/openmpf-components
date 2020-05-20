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

set(CMAKE_BUILD_TYPE RelWithDebInfo)    # build release version with debug symbols
#set(CMAKE_BUILD_TYPE Debug)            # build debug version

add_definitions(-Wfatal-errors)
include(../ComponentSetup.cmake)

find_package(OpenCV 3.4.7 EXACT
    REQUIRED PATHS /opt/opencv-3.4.7
    COMPONENTS
               opencv_dnn
               opencv_video
               opencv_face
               opencv_tracking
               )

find_package(dlib REQUIRED)
find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)

find_package(GTest)
if (${GTEST_FOUND})
    message("-- found GTest, configuring tests")
    enable_testing()
    include(Dart)
endif()

set(OCV_SSD_FACE_DETECTION_SOURCE_FILES
        JobConfig.cpp
        DetectionLocation.cpp
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