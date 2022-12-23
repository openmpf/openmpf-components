#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
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
project(tesseractocr-detection-tests)

set(CMAKE_CXX_STANDARD 17)

find_package(GTest)
if (${GTEST_FOUND})

    enable_testing()
    find_package(GTest REQUIRED)
    find_package(mpfComponentTestUtils REQUIRED)

    include_directories(..)
    add_executable(TesseractOCRTextDetectionTest test_tesseract_ocr_detection.cpp)
    target_link_libraries(TesseractOCRTextDetectionTest mpfTesseractOCRTextDetection mpfComponentTestUtils GTest::gtest_main)

    add_test(NAME TesseractOCRTextDetectionTest COMMAND TesseractOCRTextDetectionTest)

    # Install test images and videos.
    file(COPY data/ DESTINATION data)
    file(COPY config/ DESTINATION config)
endif()
