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

FROM openmpf/openmpf_argos_translation_models:1.9.1-to-eng AS argos_models



FROM ${BUILD_REGISTRY}openmpf_python_component_build:${BUILD_TAG} AS build

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update && apt-get install -y --no-install-recommends cmake pkg-config make


RUN pip3 install --no-cache-dir 'argostranslate==1.9.1'

RUN --mount=target=.,readwrite install-component.sh



FROM ${BUILD_REGISTRY}openmpf_python_executor:${BUILD_TAG} AS setup_models

COPY --from=build $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV

# To include a language model add *<ISO_LANG>_en* to the cp command.
# Refer to README for list of supported languages
RUN --mount=from=argos_models,source=/models,target=/all-models \
    --mount=source=setup_models/argos_model_installer.py,target=/opt/argos_model_installer.py \
    cd /all-models; \
    mkdir /models; \
    cp -- *de_en* *fr_en* *ru_en* *zh_en* *zt_en* *es_en* *ar_en* *ko_en* *fa_en* /models; \
    python3 /opt/argos_model_installer.py /models; \
    rm -r /models;

RUN argospm update

# Another option for downloading language models is to add lines below for languages of interest:
# RUN argospm install translate-<INPUT_ISO_LANG>_en
# RUN argospm install translate-es_en

# Please note that argospm may download models that require a newer version of argostranslate.


FROM build AS run_tests

ARG RUN_TESTS=false
RUN --mount=target=.,readwrite \
    --mount=from=setup_models,source=/root/.local/share/argos-translate,target=/root/.local/share/argos-translate \
<<eot
    if [ "${RUN_TESTS,,}" == true ]; then
        python tests/test_argos_translation.py
    fi
eot


FROM setup_models

COPY --from=run_tests $PLUGINS_DIR/ArgosTranslation $PLUGINS_DIR/ArgosTranslation


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Argos Translation" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
