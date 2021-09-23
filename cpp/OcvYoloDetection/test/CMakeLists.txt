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
project(ocv-yolo-detection-tests)

set(CMAKE_CXX_STANDARD 11)
find_package(GTest)
if (${GTEST_FOUND})
    enable_testing()
    find_package(Qt4 REQUIRED)
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)

    if ($ENV{TRITON_SUPPORT})
        add_executable(OcvYoloDetectionTest
                test_ocv_local_yolo_detection.cpp
                test_ocv_triton_yolo_detection.cpp
                TestUtils.cpp TestUtils.h)
    else()
        add_executable(OcvYoloDetectionTest
                test_ocv_local_yolo_detection.cpp
                TestUtils.cpp TestUtils.h)
    endif()

    target_link_libraries(OcvYoloDetectionTest
            mpfOcvYoloDetection
            mpfComponentTestUtils
            GTest::GTest
            GTest::Main
            Qt4::QtCore)

    add_test(NAME OcvYoloDetectionTest COMMAND OcvYoloDetectionTest)

    # allow for 'make check'
    add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --verbose)

    # Create directory for test output files
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/test_output/)

    # Create link to test data so that test binary can access it.
    file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/data ${CMAKE_CURRENT_BINARY_DIR}/data SYMBOLIC)

    file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/../plugin-files
                     ${CMAKE_CURRENT_BINARY_DIR}/OcvYoloDetection SYMBOLIC)
endif()
