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

ARG BUILD_REGISTRY
ARG BUILD_TAG=latest
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget \
    # For Google Gemini 
    # After installing the following /usr/bin will have:
    # python3 -> python3.8
    # python3.8
    # python3.9
    python3.9 python3.9-venv libpython3.9

# Create separate venv for Python 3.9 subprocess
RUN mkdir -p /gemini-subprocess/venv; \
    python3.9 -m venv /gemini-subprocess/venv; \
    /gemini-subprocess/venv/bin/pip3 install google-genai pillow numpy

COPY gemini-process-image.py gemini_component/resource_tracker_monkeypatch.py /gemini-subprocess

RUN pip3 install --upgrade pip

RUN pip3 install tenacity opencv-python
        
ARG RUN_TESTS=false

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_gemini.py; fi

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Gemini Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"