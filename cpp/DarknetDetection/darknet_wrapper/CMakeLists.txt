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
project(darknet-wrapper)

set(CMAKE_CXX_STANDARD 11)


find_package(OpenCV 4.5.0 EXACT REQUIRED PATHS /opt/opencv-4.5.0 COMPONENTS opencv_core opencv_highgui)

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(Qt4 REQUIRED)



set(DARKNET_WRAPPER_SOURCE_FILES
    DarknetImpl.cpp DarknetImpl.h
    ../include/DarknetInterface.h
    )

add_library(darknet_wrapper MODULE ${DARKNET_WRAPPER_SOURCE_FILES})
target_link_libraries(darknet_wrapper mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
    darknet_lib ${OpenCV_LIBS})


file(MAKE_DIRECTORY ${pluginLocation}/lib/)

add_custom_command(TARGET darknet_wrapper POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:darknet_wrapper> ${pluginLocation}/lib/
    )
copy_shared_libs(darknet_wrapper ${pluginLocation})



if (DARKNET_BUILD_CUDA)
    add_library(darknet_wrapper_cuda MODULE ${DARKNET_WRAPPER_SOURCE_FILES})
    target_link_libraries(darknet_wrapper_cuda mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils
        darknet_lib_cuda ${OpenCV_LIBS})
    target_compile_definitions(darknet_wrapper_cuda PRIVATE -DGPU)


    add_custom_command(TARGET darknet_wrapper_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:darknet_wrapper_cuda> ${pluginLocation}/lib/
        )

    copy_shared_libs(darknet_wrapper_cuda ${pluginLocation})
endif()

