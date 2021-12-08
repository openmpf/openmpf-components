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

ARG MODELS_REGISTRY=openmpf/

FROM ${MODELS_REGISTRY}openmpf_ocv_yolo_detection_triton_models:6.5 as models


FROM nvcr.io/nvidia/tritonserver:21.06-py3 as openmpf_triton_server

# TODO:
# - Only load 416 on startup.
# - Start long-running expiration process.
# - Enable dynamic batching

# Weights will later be used to generate TensorRT engine files in the entrypoint.
COPY --from=models /models/yolov4.wts /model-gen/yolo/yolov4.wts

RUN apt-get update; \
    apt-get install -y cmake; \
    rm -rf /var/lib/apt/lists/*

# Generate binaries that will later be used to generate TensorRT engine files in the entrypoint.
# Also, generate TensorRT plugin libraries.
RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp; \
    git clone https://github.com/openmpf/yolov4-triton-tensorrt.git; \
    cd yolov4-triton-tensorrt; \
    mkdir -p /plugins; \
    # Support 416x416 images
    git checkout openmpf/dim-416; \
    mkdir build-416; \
    cd build-416; \
    cmake ..; \
    make; \
    mv main /model-gen/yolo/gen-engine-416; \
    mv libyolo416layerplugin.so /plugins/libyolo416layerplugin.so; \
    # Support 608x608 images \
    cd ..; \
    git checkout openmpf/dim-608; \
    mkdir build.608; \
    cd build.608; \
    cmake ..; \
    make; \
    mv main /model-gen/yolo/gen-engine-608; \
    mv libyolo608layerplugin.so /plugins/libyolo608layerplugin.so

COPY docker-entrypoint.sh /opt/tritonserver

ENTRYPOINT ["/opt/tritonserver/docker-entrypoint.sh"]

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Triton Server." \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
