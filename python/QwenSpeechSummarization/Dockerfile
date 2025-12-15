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
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

ARG RUN_TESTS=true
RUN set -x; DEPS="transformers>=4.51.0 accelerate pydantic openai jinja2"; \
    if [ "${RUN_TESTS,,}" == true ]; then DEPS="$DEPS pytest"; fi; \
    pip3 install --no-cache-dir $DEPS

RUN --mount=target=.,readwrite \
    install-component.sh; \
    # make sure the tokenizer is available offline
    /opt/mpf/plugin-venv/bin/python3 -c 'from qwen_speech_summarization_component.qwen_speech_summarization_component import QwenSpeechSummaryComponent; QwenSpeechSummaryComponent()'; \
    if [ "${RUN_TESTS,,}" == true ]; then pytest qwen_speech_summarization_component; fi

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Qwen Speech Summarization" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
