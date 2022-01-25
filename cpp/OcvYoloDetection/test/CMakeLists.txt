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
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)

    add_executable(OcvLocalYoloDetectionTest
            test_ocv_local_yolo_detection.cpp
            TestUtils.cpp TestUtils.h)

    target_link_libraries(OcvLocalYoloDetectionTest
            mpfOcvYoloDetection
            GTest::GTest
            GTest::Main)

    add_test(NAME OcvLocalYoloDetectionTest COMMAND OcvLocalYoloDetectionTest)

    if ($ENV{TRITON_SUPPORT})
        add_executable(OcvTritonYoloDetectionTest
                test_ocv_triton_yolo_detection.cpp
                TestUtils.cpp TestUtils.h)

        target_link_libraries(OcvTritonYoloDetectionTest
                mpfOcvYoloDetection
                GTest::GTest
                GTest::Main)
    endif()

    # Create directory for test output files
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test/test_output/)

    # Install test images, videos, and other files
    file(COPY data/ DESTINATION data)
    file(COPY log4cxx.properties DESTINATION .)

endif()
