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
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./OcvDnnDetectionTest; fi


FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as ocv_dnn_component

COPY --from=build_component /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 /usr/lib/x86_64-linux-gnu/

COPY --from=build_component /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 /usr/lib/x86_64-linux-gnu/

COPY --from=build_component $BUILD_DIR/plugin/OcvDnnDetection $PLUGINS_DIR/OcvDnnDetection

COPY --from=build_component $BUILD_DIR/libmpfOcvDnnDetection.so $PLUGINS_DIR/OcvDnnDetection/lib/

ENV LD_LIBRARY_PATH $PLUGINS_DIR/OcvDnnDetection/lib

LABEL org.label-schema.license="Mixed" \
      org.label-schema.name="OpenMPF OpenCV Deep Neural Network (DNN) Classification" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"


FROM ocv_dnn_component as sample_executable

COPY --from=build_component $BUILD_DIR/sample_ocv_dnn_classifier /opt/mpf/

COPY --from=build_component $BUILD_DIR/test/data /opt/mpf/test-data

RUN ln -s /opt/mpf/plugins /opt/mpf/plugin

ENV LD_LIBRARY_PATH $PLUGINS_DIR/OcvDnnDetection/lib

ENTRYPOINT ["/opt/mpf/sample_ocv_dnn_classifier"]


# Set final image when no --target specified.
FROM ocv_dnn_component
