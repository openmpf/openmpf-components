<<<<<<< HEAD
# syntax=docker/dockerfile:experimental

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

ARG MODELS_REGISTRY=openmpf/
ARG BUILD_REGISTRY
ARG BUILD_TAG=latest
FROM ${MODELS_REGISTRY}openmpf_clip_detection_models:9.0.0-feature as models
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

COPY --from=models /models/ViT-B-32.pt /models/ViT-B-32.pt
COPY --from=models /models/ViT-L-14.pt /models/ViT-L-14.pt
COPY --from=models /models/model.pth.tar-50 /models/prompt_learner/model.pth.tar-50

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget

RUN pip3 install --upgrade pip

RUN pip3 install ftfy regex tqdm

RUN pip3 install torch==1.7.1 torchvision==0.8.2 --extra-index-url https://download.pytorch.org/whl/cu102

RUN --mount=type=tmpfs,target=/tmp \
        mkdir /tmp/CLIP; \
        wget -O- 'https://github.com/openai/CLIP/tarball/master' \
            | tar --extract --gzip --directory /tmp/CLIP; \
        cd /tmp/CLIP/*; \
        pip3 install .

COPY ./Dassl.pytorch/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

RUN --mount=target=.,readwrite \
    cd ./Dassl.pytorch && pip3 install .
        
COPY ./CoOp /opt/coop_src/CoOp
        
ARG RUN_TESTS=false

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_clip.py; fi

ENV PYTHONPATH="$PYTHONPATH:/opt/coop_src"

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF CLIP Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
=======
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

ARG MODELS_REGISTRY=openmpf/
ARG BUILD_REGISTRY
ARG BUILD_TAG=latest
FROM ${MODELS_REGISTRY}openmpf_clip_detection_models:8.0.0 as models
FROM ${BUILD_REGISTRY}openmpf_python_executor_ssb:${BUILD_TAG}

COPY --from=models /models/ViT-B-32.pt /models/ViT-B-32.pt
COPY --from=models /models/ViT-L-14.pt /models/ViT-L-14.pt

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y wget

RUN pip3 install --upgrade pip

RUN pip3 install ftfy regex tqdm 'setuptools<70'

RUN --mount=type=tmpfs,target=/tmp \
        mkdir /tmp/CLIP; \
        wget -O- 'https://github.com/openai/CLIP/tarball/master' \
            | tar --extract --gzip --directory /tmp/CLIP; \
        cd /tmp/CLIP/*; \
        pip3 install . 'torchvision==0.14.1'

ARG RUN_TESTS=false

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_clip.py; fi

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF CLIP Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
>>>>>>> origin/develop
