#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
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
FROM ${BUILD_REGISTRY}openmpf_python_component_build:${BUILD_TAG} as build_component

# Manually install packages before COPY so they don't need to be re-installed every time the source code changes.
RUN pip3 install --no-cache-dir 'azure-storage-blob>=12.3' 'python-dateutil'

COPY . .

RUN install-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then python tests/test_acs_speech.py; fi



FROM ${BUILD_REGISTRY}openmpf_python_executor:${BUILD_TAG}

COPY --from=build_component $COMPONENT_VIRTUALENV $COMPONENT_VIRTUALENV

COPY --from=build_component $PLUGINS_DIR/AzureSpeechDetection $PLUGINS_DIR/AzureSpeechDetection


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Azure Speech Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
