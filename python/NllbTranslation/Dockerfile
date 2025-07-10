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

ARG BUILD_TYPE
WORKDIR /home/mpf/openmpf-projects/openmpf-python-component-sdk/detection/nlp_text_splitter
RUN if [[ "${BUILD_TYPE}" == "gpu" ]]; \
    then ./install.sh --gpu; \
    else ./install.sh; fi;

FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG} AS cpu_component

COPY --from=build $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV

COPY --from=build /opt/wtp /opt/wtp

# download nllb model
ARG MODEL_NAME=facebook/nllb-200-3.3B
ARG MODEL_REVISION=1a07f7d195896b2114afcb79b7b57ab512e7b43e
RUN --mount=type=tmpfs,target=/tmp/models \
    mkdir -p "/models/facebook"; \
    python -c \
    "import os; from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    modelrev = os.environ['MODEL_REVISION']; \
    modelname = os.environ['MODEL_NAME']; \
    modelpath = r'/models/' + modelname; \
    tokenizer = AutoTokenizer.from_pretrained(modelname, revision=modelrev); \
    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, revision=modelrev); \
    tokenizer.save_pretrained(modelpath); \
    model.save_pretrained(modelpath)"

ARG RUN_TESTS=false
RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_nllb_translation.py; fi


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Transformer Tagging" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"

FROM cpu_component AS gpu_component

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM ${BUILD_TYPE}_component