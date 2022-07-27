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
project(TesseractOCRTextDetection)
ADD_DEFINITIONS("-std=c++11")
include(../ComponentSetup.cmake)


unset(OpenCV_CONFIG_PATH CACHE)
find_package(OpenCV 4.5.5 EXACT REQUIRED PATHS /opt/opencv-4.5.5
    COMPONENTS opencv_highgui opencv_objdetect opencv_features2d opencv_ml opencv_flann opencv_video)


find_package(mpfComponentInterface REQUIRED)
find_package(mpfDetectionComponentApi REQUIRED)
find_package(mpfComponentUtils REQUIRED)
find_package( Leptonica 1.75.0 REQUIRED )
find_package( Tesseract 4.1.1 REQUIRED )
find_package(ImageMagick REQUIRED COMPONENTS Magick++)

find_package(Boost 1.53.0 COMPONENTS regex locale filesystem)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${ImageMagick_INCLUDE_DIRS})
include_directories(${Tesseract_INCLUDE_DIRS})
include_directories(${Leptonica_INCLUDE_DIRS})

set(OCR_SOURCE_FILES
        TesseractOCRTextDetection.cpp TesseractOCRTextDetection.h)

add_definitions( -DMAGICKCORE_QUANTUM_DEPTH=16 )
add_definitions( -DMAGICKCORE_HDRI_ENABLE=0 )

# Build library
add_library(mpfTesseractOCRTextDetection SHARED ${OCR_SOURCE_FILES})


target_link_libraries(mpfTesseractOCRTextDetection mpfComponentInterface mpfDetectionComponentApi
    mpfComponentUtils ${OpenCV_LIBS} ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES}
    ${LOG4CXX_LIBRARIES} ${Boost_LIBRARIES} ${ImageMagick_LIBRARIES})

configure_mpf_component(TesseractOCRTextDetection TARGETS  mpfTesseractOCRTextDetection)


add_subdirectory(test)
add_subdirectory(model_updater_tesseract_src)

# Build executable
add_executable(sample_tesseract_ocr_detector sample_tesseract_ocr_detector.cpp)
target_link_libraries(sample_tesseract_ocr_detector mpfTesseractOCRTextDetection)

# Build executable
add_executable              (tessdata_model_updater tessdata_model_updater.cpp)
target_link_libraries       (tessdata_model_updater libtesseract ${Tesseract_LIBRARIES} ${Leptonica_LIBRARIES} ${Boost_LIBRARIES})
