# syntax=docker/dockerfile:1.2

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

ARG BUILD_REGISTRY
ARG BUILD_TAG=latest
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN --mount=target=. build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./OcvFaceDetectionTest; fi



FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV LD_LIBRARY_PATH $PLUGINS_DIR/OcvFaceDetection/lib

COPY --from=build_component $BUILD_DIR/plugin/OcvFaceDetection $PLUGINS_DIR/OcvFaceDetection

COPY --from=build_component $BUILD_DIR/libmpfOcvFaceDetection.so \
                            $PLUGINS_DIR/OcvFaceDetection/lib/

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF OpenCV Face Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
