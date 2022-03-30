# syntax=docker/dockerfile:1.2

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

# Download weights in separate stage so they don't need to be re-downloaded when base images change.
FROM ubuntu:20.04 as download_dependencies

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    apt-get install --no-install-recommends -y wget ca-certificates;

RUN mkdir /weights \
    && cd /weights \
    && wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


RUN mkdir --parents /apps/source/dlib-sources; \
    wget -O- 'http://dlib.net/files/dlib-19.20.tar.bz2' \
        | tar --extract --bzip2 --directory /apps/source/dlib-sources


RUN mkdir /opt/triton \
    && wget -O- https://github.com/triton-inference-server/server/releases/download/v2.11.0/v2.11.0_ubuntu2004.clients.tar.gz \
        | tar --extract --gzip --directory /opt/triton lib include



FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

ENV TRITON_SUPPORT=ON

ARG RUN_TESTS=false
ARG RUN_GPU_TESTS=false
ARG TRITON_SERVER=ocv-yolo-detection-server:8001

RUN --mount=target=. \
    --mount=from=download_dependencies,source=/apps/source/dlib-sources,target=/apps/source/dlib-sources \
    --mount=from=download_dependencies,source=/weights/yolov4.weights,target=/home/mpf/component_build/plugin/OcvYoloDetection/models/yolov4.weights \
    --mount=from=download_dependencies,source=/opt/triton/lib,target=/opt/triton/lib \
    --mount=from=download_dependencies,source=/opt/triton/include,target=/opt/triton/include \
    build-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then \
        cd $BUILD_DIR/test; \
        ./OcvLocalYoloDetectionTest; \
    fi; \
    if [ "${RUN_GPU_TESTS,,}" == true ]; then \
        cd $BUILD_DIR/test; \
        ./OcvTritonYoloDetectionTest $TRITON_SERVER; \
    fi


FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

COPY --from=download_dependencies /weights $PLUGINS_DIR/OcvYoloDetection/models

COPY --from=build_component /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 /usr/lib/x86_64-linux-gnu/

COPY --from=build_component /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 /usr/lib/x86_64-linux-gnu/

COPY --from=build_component $BUILD_DIR/plugin/OcvYoloDetection $PLUGINS_DIR/OcvYoloDetection

COPY --from=build_component $BUILD_DIR/libmpfOcvYoloDetection.so $PLUGINS_DIR/OcvYoloDetection/lib/

ENV LD_LIBRARY_PATH $PLUGINS_DIR/OcvYoloDetection/lib

LABEL org.label-schema.license="Mixed" \
      org.label-schema.name="OpenMPF OpenCV Yolo Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
