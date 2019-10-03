#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2019 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2019 The MITRE Corporation                                      #
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
project(ocv-dnn-detection)

set(CMAKE_CXX_STANDARD 11)

include(../ComponentSetup.cmake)

find_package(OpenCV 3.4.7 EXACT REQUIRED PATHS /opt/opencv-3.4.7
    COMPONENTS opencv_dnn)

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)


set(OCVDNN_SOURCE_FILES OcvDnnDetection.cpp OcvDnnDetection.h)

add_library(ocvdnn SHARED ${OCVDNN_SOURCE_FILES})
target_link_libraries(ocvdnn mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils ${OpenCV_LIBS})

configure_mpf_component(OcvDnnDetection TARGETS ocvdnn)

add_subdirectory(test)

# Build sample executable
add_executable(sample_ocv_dnn_classifier sample_ocv_dnn_classifier.cpp)
target_link_libraries(sample_ocv_dnn_classifier ocvdnn)
