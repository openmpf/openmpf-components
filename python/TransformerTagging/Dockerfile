# syntax=docker/dockerfile:1.2

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

ARG RUN_TESTS=false

# needed to build sentence transformer wheel file
RUN pip install --no-cache-dir \
    'wheel'

RUN pip install --no-cache-dir \
    'nltk' \
    'sentence_transformers' \
    'pandas'

# copy over punkt files
RUN python -c \
    "import nltk; nltk.download('punkt')"

# download sentence transformer model
RUN --mount=type=tmpfs,target=/tmp/models \
    mkdir -p "/models/all-mpnet-base-v2"; \
    python -c \
    "import os; \
    os.environ['REQUESTS_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'; \
    from sentence_transformers import SentenceTransformer; \
    model = SentenceTransformer('all-mpnet-base-v2', cache_folder='/tmp/models'); \
    model.save('/models/all-mpnet-base-v2')" 

RUN --mount=target=.,readwrite \
    install-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then python tests/test_transformer_tagging.py; fi


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Transformer Tagging" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
