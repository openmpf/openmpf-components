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

RUN pip3 install --no-cache-dir 'argostranslate>=1.8.0'

RUN argospm update

# To download a language model use argospm install translate_<INPUT_ISO_LANG>_en
# Refer to README for list of supported languages
RUN argospm install translate-de_en && \
    argospm install translate-fr_en && \
    argospm install translate-ru_en && \
    argospm install translate-es_en

ARG RUN_TESTS=false

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_argos_translation.py; fi

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Argos Translation" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"


