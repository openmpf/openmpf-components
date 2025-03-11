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
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    apt-get upgrade -y; \
    apt-get install --no-install-recommends -y wget ca-certificates gnupg2; \
    wget --ca-directory=/etc/ssl/certs -O- 'https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xba6932366a755776' \
        | apt-key add -; \
    echo 'deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu focal main' \
        > /etc/apt/sources.list.d/deadsnakes.list; \
    apt-get update; \
    apt-get install --no-install-recommends -y \
        python3.10 python3.10-venv libpython3.10; \
    python3.10 -m ensurepip --upgrade --default-pip

RUN --mount=type=bind,source=subprocess/requirements.txt,target=/tmp/requirements.txt \
    mkdir -p /llama/venv; \
    python3.10 -m venv /llama/venv; \
    /llama/venv/bin/pip install --cert /etc/ssl/certs/ca-certificates.crt -r /tmp/requirements.txt

RUN . /llama/venv/bin/activate; \
    pip install "huggingface_hub[hf_transfer]"; \
    pip install hf_transfer; \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "DAMO-NLP-SG/VideoLLaMA3-7B"

COPY subprocess/summarize_video.py /llama/summarize_video.py

ARG RUN_TESTS=true

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_llama_video_summarization.py; fi

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF LLaMA Video Summarization" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
