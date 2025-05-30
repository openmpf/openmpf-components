# syntax=docker/dockerfile:1.4

#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
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

# To enable GPU resources, update
# below line to BUILD_TYPE=gpu
ARG BUILD_TYPE=cpu

FROM ${BUILD_REGISTRY}openmpf_python_component_build:${BUILD_TAG} AS build

RUN pip install langcodes

ARG BUILD_TYPE
RUN <<eot
    cd /home/mpf/openmpf-projects/openmpf-python-component-sdk/detection/nlp_text_splitter
    if [[ ${BUILD_TYPE,,} = gpu ]]; then
        ./install.sh --gpu
    else
        ./install.sh
    fi
eot

ARG RUN_TESTS=false
RUN --mount=target=.,readwrite <<eot
    install-component.sh;
    if [ "${RUN_TESTS,,}" == true ]; then
        python tests/test_acs_translation.py;
    fi
eot


FROM ${BUILD_REGISTRY}openmpf_python_executor:${BUILD_TAG} AS cpu_component

COPY --from=build $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV

COPY --from=build /opt/wtp /opt/wtp

COPY --from=build $PLUGINS_DIR/AzureTranslation $PLUGINS_DIR/AzureTranslation


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Azure Translation" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"


FROM cpu_component AS gpu_component

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM ${BUILD_TYPE}_component
