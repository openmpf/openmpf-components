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
project(dlib-face-detection)

set(CMAKE_CXX_STANDARD 11)

set(USE_AVX_INSTRUCTIONS ON)
set(DLIB_USE_CUDA OFF)
set(DLIB_DIR /apps/source/dlib-sources/dlib-19.20/dlib)
include(${DLIB_DIR}/cmake)
set_property(TARGET dlib PROPERTY POSITION_INDEPENDENT_CODE ON)

include(../ComponentSetup.cmake)

find_package(OpenCV 4.5.0 EXACT REQUIRED PATHS /opt/opencv-4.5.0 COMPONENTS opencv_highgui)
find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)


set(DLIB_FACE_SOURCE_FILES DlibFaceDetection.cpp DlibFaceDetection.h)

add_library(mpfDlibFaceDetection SHARED ${DLIB_FACE_SOURCE_FILES})

target_link_libraries(mpfDlibFaceDetection mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
    dlib ${OpenCV_LIBS})

configure_mpf_component(DlibFaceDetection TARGETS mpfDlibFaceDetection)

add_subdirectory(test)

add_executable(sample_dlib_face_detector sample_dlib_face_detector.cpp)
target_link_libraries(sample_dlib_face_detector mpfDlibFaceDetection)
