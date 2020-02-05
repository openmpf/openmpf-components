#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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

FROM centos:7 as download_weights
# Download weights in separate stage so they don't need to be re-downloaded when base images change.
RUN mkdir /weights && curl --location https://pjreddie.com/media/files/yolov3.weights > /weights/yolov3.weights


####################################################
# Build Component
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64 \
    && VERSION=9-0-9.0.176-1 \
    && yum install -y --nogpgcheck cuda-minimal-build-$VERSION cuda-cublas-dev-$VERSION cuda-curand-dev-$VERSION \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && ln -s cuda-9.0 /usr/local/cuda \
    && echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/cuda.conf

COPY . .

RUN build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./DarknetDetectionTest; fi


####################################################
FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG}

RUN yum-config-manager --add-repo http://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64 \
    && yum install -y --nogpgcheck cuda-cudart-9-0-9.0.176-1 \
    && yum clean all \
    && rm -rf /var/cache/yum/* \
    && ln -s cuda-9.0 /usr/local/cuda \
    && echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/cuda.conf

# Environment variables required by nvidia runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

COPY --from=download_weights /weights $INSTALL_DIR/models

ENV COMPONENT_LOG_NAME darknet-detection.log

COPY --from=build_component $BUILD_DIR/plugin/DarknetDetection $PLUGINS_DIR/DarknetDetection

COPY --from=build_component $BUILD_DIR/libmpfDarknetDetection.so \
                            $PLUGINS_DIR/DarknetDetection/lib/


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Darknet Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"

