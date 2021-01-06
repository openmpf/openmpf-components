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
project(darknet-detection)

set(CMAKE_CXX_STANDARD 11)

include(../ComponentSetup.cmake)

find_package(OpenCV 4.5.0 EXACT REQUIRED PATHS /opt/opencv-4.5.0 COMPONENTS opencv_core opencv_highgui)


find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)

get_plugin_build_location(DarknetDetection pluginLocation)

if (DEFINED ENV{MPF_DISABLE_CUDA_BUILD})
set(DARKNET_BUILD_CUDA OFF)
    message(
    "The CUDA version of Darknet will not be built because the MPF_DISABLE_CUDA_BUILD environment variable was set.")
else()
    find_package(CUDA)
    if (CUDA_FOUND)
        set(DARKNET_BUILD_CUDA ON)
    else()
        set(DARKNET_BUILD_CUDA OFF)
        message("The CUDA version of Darknet will not be built because the CUDA library was not found on this machine")
    endif()
endif()

add_subdirectory(darknet_lib)
add_subdirectory(darknet_wrapper)

set(DARKNET_COMPONENT_SOURCE_FILES
    DarknetDetection.cpp DarknetDetection.h
    Trackers.cpp Trackers.h)

add_library(mpfDarknetDetection SHARED ${DARKNET_COMPONENT_SOURCE_FILES})
target_link_libraries(mpfDarknetDetection mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
    ${OpenCV_LIBS})


set(DARKNET_STREAMING_COMPONENT_SOURCE_FILES
    DarknetStreamingDetection.cpp DarknetStreamingDetection.h
    Trackers.cpp Trackers.h)

add_library(mpfDarknetStreamingDetection SHARED ${DARKNET_STREAMING_COMPONENT_SOURCE_FILES})
target_link_libraries(mpfDarknetStreamingDetection mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
    ${OpenCV_LIBS})


configure_mpf_component(DarknetDetection TARGETS mpfDarknetDetection mpfDarknetStreamingDetection)

add_dependencies(mpfDarknetDetection darknet_wrapper)
if (DARKNET_BUILD_CUDA)
    add_dependencies(mpfDarknetDetection darknet_wrapper_cuda)
endif()

add_executable(sample_darknet_detector sample_darknet_detector.cpp)
target_link_libraries(sample_darknet_detector mpfDarknetDetection)

add_subdirectory(test)
