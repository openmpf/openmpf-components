#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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
project(openmpf-components)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(OcvFaceDetection)
add_subdirectory(OcvDnnDetection)
add_subdirectory(SceneChangeDetection)
add_subdirectory(KeywordTagging)

# The following components require additional dependencies. Refer to their Dockerfiles for
# installation instructions.
# add_subdirectory(OalprLicensePlateTextDetection)
# add_subdirectory(OcvYoloDetection)
# add_subdirectory(TesseractOCRTextDetection)





