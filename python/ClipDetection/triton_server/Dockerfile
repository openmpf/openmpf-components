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

FROM ${MODELS_REGISTRY}openmpf_clip_detection_triton_models:9.0.0 as models

FROM nvcr.io/nvidia/tritonserver:22.04-py3 as openmpf_triton_server

COPY --from=models /models/vit_b_32.pt /models/vit_b_32/1/vit_b_32.pt
COPY models/vit_b_32.config.pbtxt /models/vit_b_32/config.pbtxt

COPY --from=models /models/vit_l_14.pt /models/vit_l_14/1/vit_l_14.pt
COPY models/vit_l_14.config.pbtxt /models/vit_l_14/config.pbtxt

RUN apt-get update; \
    apt-get -y upgrade; \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade setuptools wheel

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF CLIP Detection Triton Server" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"