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
project(ocv-face-detection)

set(CMAKE_CXX_STANDARD 17)

include(../ComponentSetup.cmake)

find_package(OpenCV 4.9.0 EXACT REQUIRED PATHS /opt/opencv-4.9.0
    COMPONENTS opencv_highgui opencv_objdetect opencv_features2d opencv_ml opencv_flann opencv_video)

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)


set(OCV_FACE_DETECTION_SOURCE_FILES
        OcvDetection.cpp OcvDetection.h
        OcvFaceDetection.cpp OcvFaceDetection.h)

add_library(mpfOcvFaceDetection SHARED ${OCV_FACE_DETECTION_SOURCE_FILES})
target_link_libraries(mpfOcvFaceDetection mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils ${OpenCV_LIBS})


configure_mpf_component(OcvFaceDetection TARGETS mpfOcvFaceDetection)


add_subdirectory(test)

add_executable(sample_ocv_face_detector sample_ocv_face_detector.cpp)
target_link_libraries(sample_ocv_face_detector mpfOcvFaceDetection)

