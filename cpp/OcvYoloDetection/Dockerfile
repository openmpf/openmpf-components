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


FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_setup

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    apt-get install --no-install-recommends -y \
        git libssl-dev zlib1g-dev rapidjson-dev python3.8 rsync;


RUN wget -O- 'https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.tar.gz' \
        | tar --extract --gzip --directory=/usr --strip-components=1; \
    ln -sf /usr/bin/cmake /usr/bin/cmake3;


# Triton Client build
ARG TRITON_REPO_TAG=r21.06
ENV TRITON_VERSION=2.11.0
RUN --mount=type=tmpfs,target=/tmp \
    mkdir -p /tmp/triton; \
    cd /tmp/triton; \
    git clone -b ${TRITON_REPO_TAG}  https://github.com/triton-inference-server/client.git; \
    # build the client library
    mkdir -p /tmp/triton/client/build; \
    cd /tmp/triton/client/build; \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-std=c++11 -Wfatal-errors" \
          -DCMAKE_CXX_STANDARD=11 \
          -DCMAKE_CXX_STANDARD_REQUIRED=ON \
          -DCMAKE_INSTALL_PREFIX:PATH=/opt/triton/ \
          -DTRITON_VERSION="$TRITON_VERSION" \
          -DTRITON_COMMON_REPO_TAG="$TRITON_REPO_TAG" \
          -DTRITON_THIRD_PARTY_REPO_TAG="$TRITON_REPO_TAG" \
          -DTRITON_ENABLE_GPU=ON \
          -DTRITON_ENABLE_CC_GRPC=ON \
          -DTRITON_ENABLE_CC_HTTP=OFF \
          -DTRITON_ENABLE_PERF_ANALYZER=OFF \
          -DTRITON_ENABLE_PYTHON_HTTP=OFF \
          -DTRITON_ENABLE_PYTHON_GRPC=OFF \
          -DTRITON_ENABLE_EXAMPLES=OFF \
          -DTRITON_ENABLE_TESTS=OFF \
          ..; \
    make -j "$(nproc)"; \
    rsync -aL /tmp/triton/client/build/third-party/c-ares/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/c-ares/lib/ /opt/triton/lib/; \
    rsync -aL /tmp/triton/client/build/third-party/c-ares/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/client/build/third-party/curl/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/curl/lib/ /opt/triton/lib/; \
    rsync -aL /tmp/triton/client/build/third-party/curl/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/lib/ /opt/triton/lib/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/share/ /opt/triton/share/; \
    rsync -aL /tmp/triton/client/build/third-party/protobuf/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/protobuf/lib/ /opt/triton/lib/; \
    rsync -aL /tmp/triton/client/build/third-party/protobuf/bin/ /opt/triton/bin/; \
    echo '/opt/triton/lib' > /etc/ld.so.conf.d/triton.conf; \
    ldconfig;

ENV CMAKE_PREFIX_PATH="/opt/triton/lib/cmake:${CMAKE_PREFIX_PATH}"


FROM build_setup as build_component

ENV TRITON_SUPPORT=ON

ARG RUN_TESTS=false
ARG RUN_GPU_TESTS=false
ARG TRITON_SERVER=ocv-yolo-detection-server:8001

RUN --mount=target=. \
    --mount=from=download_dependencies,source=/apps/source/dlib-sources,target=/apps/source/dlib-sources \
    --mount=from=download_dependencies,source=/weights/yolov4.weights,target=/home/mpf/component_build/plugin/OcvYoloDetection/models/yolov4.weights \
    build-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then \
        cd $BUILD_DIR/test; \
        ./OcvLocalYoloDetectionTest; \
    fi; \
    if [ "${RUN_GPU_TESTS,,}" == true ]; then \
        cd $BUILD_DIR/test; \
        ./OcvTritonYoloDetectionTest $TRITON_SERVER; \
    fi

########################################################################
# CLion
# Refer to: https://blog.jetbrains.com/clion/2020/01/using-docker-with-clion/
# Based on: https://github.com/JetBrains/clion-remote/blob/master/Dockerfile.centos7-cpp-env
#
# CLion toolchain settings:
# - CMake: /usr/bin/cmake3
# - Make: /usr/bin/gmake
# - C Compiler: /usr/local/bin/gcc
# - C++ Compiler: /usr/local/bin/g++
# - Debugger: /usr/bin/gdb
#
# CLion CMake settings:
# - CMake options: -DCMAKE_PREFIX_PATH=/opt/triton/lib64/cmake;/home/mpf/mpf-sdk-install/lib/cmake
# - Environment: TRITON_SUPPORT=ON
#
# CLion deployment settings (SFTP Remote Development):
# - [Connection] SSH configuration: root@localhost:2222
# - [Connection] Root path: /home/mpf/component_src
# - [Mappings] Deployment path: .
#
# Build command:
#   COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker build . --progress=plain --target=clion \
#   -t openmpf_ocv_yolo_detection:triton-clion
#
# Run command:
#   docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env \
#   openmpf_ocv_yolo_detection:triton-clion
########################################################################
FROM build_setup as clion

COPY --from=download_library_source /apps/source/dlib-sources /apps/source/dlib-sources

RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=tmpfs,target=/var/cache/yum \
    yum install --assumeyes openssh-server gdb; \
    yum clean all

RUN ssh-keygen -A

RUN mkdir /root/.ssh; \
  ( \
    echo 'LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64'; \
  ) > /root/.ssh/environment

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'PermitUserEnvironment yes'; \
    echo 'Subsystem sftp /usr/libexec/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion

# Set root user password for SSH access.
RUN echo "openmpf123" | passwd root --stdin

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]

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
