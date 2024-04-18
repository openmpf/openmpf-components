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

ARG RUN_TESTS=false
# To enable GPU resources, update
# below line to BUILD_TYPE=gpu
ARG BUILD_TYPE=cpu

FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG} as download_python_packages


RUN pip install --no-cache-dir langcodes

RUN apt-get update && \
    apt-get install -y git git-lfs && \
    git lfs install && \
    rm -rf /var/lib/apt/lists/*

# Install WtP and spaCy
RUN pip install --upgrade pip && \
    pip install "spacy>=3.7.4" && \
    pip install "wtpsplit>=1.3.0"

# Modify to add downloads for other models of interest.
RUN mkdir /wtp_models && cd /wtp_models && \
    git clone https://huggingface.co/benjamin/wtp-bert-mini && \
    python3 -m spacy download xx_sent_ud_sm

########################################################################
FROM download_python_packages as cpu_component_build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

########################################################################
FROM download_python_packages as gpu_component_build

# Environment variables required by nvidia runtime.
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip install torch

########################################################################

FROM ${BUILD_TYPE}_component_build as component_final

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_acs_translation.py; fi

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Azure Translation" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
