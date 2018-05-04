#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2018 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2018 The MITRE Corporation                                      #
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

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(OcvFaceDetection)
add_subdirectory(DlibFaceDetection)
add_subdirectory(OalprLicensePlateTextDetection)
add_subdirectory(CaffeDetection)

### Set the following variable when building components that use the Nvidia CUDA
### runtime and the nvcc compiler.

### By default, we build GPU components for maximum portability across GPU
### architectures. This means that we create PTX code for the minimum compute
### capability only. For the darknet component, we have not seen a performance
### benefit to creating to ELF code in addition to PTX code. However, for other
### components, this may not be the case. Component developers may override the
### value of the ARCH_BUILD_FLAGS variable given here in their component-specific
### CMakeLists.txt file.

set(ARCH_BUILD_FLAGS "-gencode arch=compute_30,code=compute_30")

add_subdirectory(DarknetDetection)





