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

FROM openmpf/openmpf_argos_translation_models:1.9.1-to-eng AS argos_models

FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

COPY ./setup_models/argos_model_installer.py /model_setup/argos_model_installer.py

RUN pip3 install --no-cache-dir 'argostranslate==1.9.1'

# To include a language model add *<ISO_LANG>_en* to the cp command.
# Refer to README for list of supported languages
RUN --mount=from=argos_models,source=/models,target=/all-models \
    cd /all-models; \
    mkdir /models; \
    cp -- *de_en* *fr_en* *ru_en* *zh_en* *zt_en* *es_en* *ar_en* *ko_en* *fa_en* /models; \
    python3 /model_setup/argos_model_installer.py /models; \
    rm -r /model_setup /models;


RUN argospm update

# Another option for downloading language models is to add lines below for languages of interest:
# RUN argospm install translate-<INPUT_ISO_LANG>_en
# RUN argospm install translate-es_en

# Please note that argospm may download models that require a newer version of argostranslate.

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
