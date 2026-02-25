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
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG} AS base

# https://huggingface.co/DAMO-NLP-SG/VideoLLaMA3-7B/tree/fd2ad62f4cec8814750a908774c199681dbb89fd
ENV MODEL_REVISION=fd2ad62f4cec8814750a908774c199681dbb89fd

RUN pip install huggingface_hub; \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt hf download "DAMO-NLP-SG/VideoLLaMA3-7B" --revision ${MODEL_REVISION}

RUN pip install --no-cache-dir \
        "jsonschema" \
        "torch==2.6.0" \
        "torchvision==0.21.0" \
        "transformers==4.46.3" \
        "decord" \
        "ffmpeg-python" \
        "imageio" \
        "opencv-python" \
        "accelerate==1.0.1"

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


# Optional build target for running unit tests with a GPU
FROM base AS gpu_test

COPY tests/test_llama_video_summarization.py /tests/test_llama_video_summarization.py
COPY tests/data /tests/data
ENV USE_MOCKS=False
ENTRYPOINT ["python", "/tests/test_llama_video_summarization.py", "TestComponent.test_multiple_videos"]


# By default Docker builds the final stage
FROM base AS final
