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
project(KeywordTagging)
set(CMAKE_CXX_STANDARD 11)
include(../ComponentSetup.cmake)

set(PACKAGE_DIR ${CMAKE_CURRENT_BINARY_DIR}/plugin/${PROJECT_NAME})
message("Package in ${PACKAGE_DIR}")

find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package(OpenCV 4.5.5 EXACT REQUIRED PATHS /opt/opencv-4.5.5 COMPONENTS opencv_core)

find_package(Qt4 REQUIRED)

find_package(Boost 1.53.0 COMPONENTS regex locale filesystem)
include_directories(${Boost_INCLUDE_DIRS})

set(KEYWORD_TAGGING_SOURCES KeywordTagging.cpp KeywordTagging.h JSON.cpp JSONValue.cpp JSON.h JSONValue.h)


# Build library
add_library(mpfKeywordTagging SHARED ${KEYWORD_TAGGING_SOURCES})


target_link_libraries(mpfKeywordTagging mpfComponentInterface mpfDetectionComponentApi mpfComponentUtils ${Boost_LIBRARIES} ${LOG4CXX_LIBRARIES})
configure_mpf_component(KeywordTagging TARGETS mpfKeywordTagging EXTRA_LIB_PATHS "${extra_lib_dirs}")

add_subdirectory(test)

# Build executable
add_executable(sample_keyword_tagger sample_keyword_tagger.cpp)
target_link_libraries(sample_keyword_tagger mpfKeywordTagging)
