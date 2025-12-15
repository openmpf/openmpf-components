# syntax=docker/dockerfile:1.20

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
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} AS build_component

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    apt-get install --no-install-recommends -y \
        git zlib1g-dev libssl-dev libcurl4-openssl-dev curl unzip

RUN ln --symbolic /usr/bin/python3 /usr/bin/python

RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp \
    && git clone -b v1.24.3 --depth 1 --recurse-submodules https://github.com/grpc/grpc \
    && mkdir -p /tmp/grpc/third_party/protobuf/cmake/build \
    && cd /tmp/grpc/third_party/protobuf/cmake/build \
    && cmake3 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release .. \
    && make -j $(nproc) install \
    && mkdir -p /tmp/grpc/third_party/cares/cares/cmake/build \
    && cd /tmp/grpc/third_party/cares/cares/cmake/build \
    && cmake3 -DCMAKE_BUILD_TYPE=Release ../.. \
    && make -j $(nproc) install \
    # build and install gRPC
    && mkdir -p /tmp/grpc/cmake/build \
    && cd /tmp/grpc/cmake/build \
    && cmake3 -DgRPC_INSTALL=ON \
              -DgRPC_BUILD_TESTS=OFF \
              -DgRPC_BENCHMARK_PROVIDER=none \
              -DgRPC_PROTOBUF_PROVIDER=package \
              -DgRPC_ZLIB_PROVIDER=package \
              -DgRPC_CARES_PROVIDER=package \
              -DgRPC_SSL_PROVIDER=package \
              -DCMAKE_BUILD_TYPE=Release ../.. \
    && make -j $(nproc) install \
    && ldconfig

# Install AWS SDK for C++ so we can use S3 storage.
RUN --mount=type=tmpfs,target=/tmp \
    mkdir /tmp/aws-sdk-cpp \
    && cd /tmp/aws-sdk-cpp \
    && wget -O- 'https://github.com/aws/aws-sdk-cpp/archive/1.9.379.tar.gz' \
        | tar --extract --gzip \
    && cd aws-sdk-cpp-1.9.379 \
    && ./prefetch_crt_dependency.sh \
    && mkdir build \
    && cd build \
    && cmake3 -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_ONLY="s3" \
             -DENABLE_TESTING=OFF \
             -DCMAKE_C_FLAGS="-Wno-unused-variable -Wno-unused-parameter -Wno-error=deprecated-declarations" \
             -DCMAKE_CXX_FLAGS="-Wno-unused-variable -Wno-unused-parameter -Wno-error=deprecated-declarations" .. \
    && make --jobs "$(nproc)" install;


# Install TensorRT-Inference-Server client libs.
RUN --mount=type=tmpfs,target=/tmp \
    cd /tmp/ \
    && git clone -b v1.7.0 --depth 1 https://github.com/NVIDIA/tensorrt-inference-server.git tensorrt-inference-server \
    # Disable building TRTIS python client.
    && sed -i '/add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/d' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt \
    # Change from: find_package(CURL CONFIG REQUIRED)
    && sed -i '33s/.*/find_package(CURL REQUIRED)/' /tmp/tensorrt-inference-server/src/clients/c++/library/CMakeLists.txt \
    && mkdir -p /tmp/tensorrt-inference-server/build/trtis-clients/build \
    && cd /tmp/tensorrt-inference-server/build/trtis-clients/build \
    && cmake3 .. \
    && make -j $(nproc) install \
    && ldconfig;

COPY . .

RUN build-component.sh


FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} AS install_component

ENV LD_LIBRARY_PATH=$PLUGINS_DIR/TrtisDetection/lib

COPY --from=build_component $BUILD_DIR/plugin/TrtisDetection \
                            $PLUGINS_DIR/TrtisDetection
COPY --from=build_component $BUILD_DIR/libmpfTrtisDetection.so \
                            $PLUGINS_DIR/TrtisDetection/lib/


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRTIS Client Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
