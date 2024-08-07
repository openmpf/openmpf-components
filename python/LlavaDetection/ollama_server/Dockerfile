# syntax=docker/dockerfile:experimental

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

FROM ollama/ollama

RUN apt-get update && \
    apt-get install --no-install-recommends -y python3=3.10.6-1~22.04 && \
    apt-get install --no-install-recommends -y python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV HF_HUB_ENABLE_HF_TRANSFER=true

COPY ./run-ollama.sh /tmp/run-ollama.sh
WORKDIR /tmp

RUN chmod +x run-ollama.sh \
    && ./run-ollama.sh
EXPOSE 11434

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF LLaVA Detection Triton Server" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"