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

# To use only CPU resources, update
# below line to BUILD_TYPE=cpu
ARG BUILD_TYPE=gpu

# Use a base image we did not build so that we do not have re-download the model every time
# openmpf_python_component_build changes.
FROM python:3.12-alpine AS download_model

RUN pip install --no-cache-dir 'huggingface_hub[cli]'

ARG MODEL_NAME=facebook/nllb-200-3.3B
ARG MODEL_REVISION=1a07f7d195896b2114afcb79b7b57ab512e7b43e

RUN hf download "$MODEL_NAME" --local-dir "/models/$MODEL_NAME" --revision "$MODEL_REVISION" \
    && rm -rf /models/$MODEL_NAME/.[^.]*


FROM ${BUILD_REGISTRY}openmpf_python_component_build:${BUILD_TAG} AS build

ARG BUILD_TYPE
RUN <<eot
    cd /home/mpf/openmpf-projects/openmpf-python-component-sdk/detection/nlp_text_splitter
    if [[ "${BUILD_TYPE}" == "gpu" ]]; then
        ./install.sh --gpu
    else
        ./install.sh
    fi
eot

RUN pip install --no-cache-dir 'accelerate>=0.26.0' 'numpy<2'

ARG RUN_TESTS=false
RUN --mount=target=.,readwrite \
    --mount=from=download_model,src=/models,target=/models \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_nllb_translation.py; fi



FROM ${BUILD_REGISTRY}openmpf_python_executor:${BUILD_TAG} AS cpu_component

COPY --from=download_model /models /models
COPY --from=build /opt/wtp /opt/wtp

COPY --from=build $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV
COPY --from=build $PLUGINS_DIR/NllbTranslation $PLUGINS_DIR/NllbTranslation

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF No Language Left Behind Translation" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"

FROM cpu_component AS gpu_component

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM ${BUILD_TYPE}_component
