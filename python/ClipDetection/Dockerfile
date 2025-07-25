# syntax=docker/dockerfile:experimental

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

ARG MODELS_REGISTRY=openmpf/
ARG BUILD_REGISTRY
ARG BUILD_TAG=latest

FROM ${MODELS_REGISTRY}openmpf_clip_detection_models:8.0.0 AS models

FROM ${BUILD_REGISTRY}openmpf_python_component_build:${BUILD_TAG} AS build

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget

RUN pip3 install --upgrade pip

RUN --mount=type=tmpfs,target=/tmp \
    mkdir /tmp/CLIP; \
    wget -O- 'https://github.com/openai/CLIP/tarball/master' \
        | tar --extract --gzip --directory /tmp/CLIP; \
    cd /tmp/CLIP/*; \
    pip3 install . 'torchvision==0.14.1' 'tritonclient[grpc]==2.40'

ARG RUN_TESTS=false

RUN --mount=target=.,readwrite \
    --mount=from=models,source=/models,target=/models \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_clip.py; fi


FROM ${BUILD_REGISTRY}openmpf_python_executor:${BUILD_TAG}

COPY --from=models /models/ViT-B-32.pt /models/ViT-B-32.pt
COPY --from=models /models/ViT-L-14.pt /models/ViT-L-14.pt

COPY --from=build $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV

COPY --from=build $PLUGINS_DIR/ClipDetection $PLUGINS_DIR/ClipDetection

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF CLIP Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
