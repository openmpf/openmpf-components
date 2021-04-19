# syntax=docker/dockerfile:1.2

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

ARG BUILD_REGISTRY
ARG BUILD_TAG=latest

FROM centos:7 as download_weights
# Download weights in separate stage so they don't need to be re-downloaded when base images change.
RUN mkdir /weights && curl --location https://pjreddie.com/media/files/yolov3.weights > /weights/yolov3.weights


####################################################
# Build Component
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=tmpfs,target=/var/cache/yum \
    yum install --nogpgcheck -y cuda-curand-dev-10-2;\
    yum clean all;

RUN --mount=target=. build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./DarknetDetectionTest; fi


####################################################
FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as darknet_component

COPY --from=download_weights /weights $PLUGINS_DIR/DarknetDetection/models

ENV LD_LIBRARY_PATH $PLUGINS_DIR/DarknetDetection/lib

COPY --from=build_component $BUILD_DIR/plugin/DarknetDetection $PLUGINS_DIR/DarknetDetection

COPY --from=build_component $BUILD_DIR/libmpfDarknetDetection.so \
                            $PLUGINS_DIR/DarknetDetection/lib/


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Darknet Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"


####################################################
FROM darknet_component as sample_executable

COPY --from=build_component $BUILD_DIR/sample_darknet_detector /opt/mpf/

COPY --from=build_component $BUILD_DIR/test/data /opt/mpf/test-data

RUN ln -s /opt/mpf/plugins /opt/mpf/plugin

ENV LD_LIBRARY_PATH $PLUGINS_DIR/DarknetDetection/lib

ENTRYPOINT ["/opt/mpf/sample_darknet_detector"]


####################################################
# Set final image when no --target specified.
FROM darknet_component
