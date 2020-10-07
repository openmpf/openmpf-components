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
ARG MODELS_REGISTRY=openmpf/

FROM ${MODELS_REGISTRY}openmpf_trtis_detection_models:5.0 as models


FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN yum -y install git openssl-devel curl-devel \
    && yum clean all \
    && rm -rf /var/cache/yum/*

# Install cares, protobuf, zlib, gRPC
WORKDIR /tmp

RUN git clone -b v1.20.0 https://github.com/grpc/grpc  && cd grpc && git submodule update --init --recursive \
&& mkdir -p /tmp/grpc/third_party/protobuf/cmake/build \
&& cd /tmp/grpc/third_party/protobuf/cmake/build \
&& cmake3 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release .. \
&& make -j`nproc` \
&& make install \
&& cd /tmp/ \
&& rm -rf /tmp/grpc/third_party/protobuf \
&& mkdir -p /tmp/grpc/third_party/cares/cares/cmake/build \
&& cd /tmp/grpc/third_party/cares/cares/cmake/build \
&& cmake3 -DCMAKE_BUILD_TYPE=Release ../.. \
&& make -j`nproc` \
&& make install \
&& cd /tmp/ \
&& rm -rf /tmp/grpc/third_party/cares/cares \
&& rm -rf /tmp/grpc/third_party/zlib \
&& rm -rf /tmp/grpc/third_party/protobuf \
# build and install gRPC
&& mkdir -p /tmp/grpc/cmake/build \
&& cd /tmp/grpc/cmake/build \
&& cmake3 -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_TESTS=OFF \
          -DgRPC_PROTOBUF_PROVIDER=package \
          -DgRPC_ZLIB_PROVIDER=package \
          -DgRPC_CARES_PROVIDER=package \
          -DgRPC_SSL_PROVIDER=package \
          -DCMAKE_BUILD_TYPE=Release ../.. \
&& make -j`nproc` \
&& make install \
&& cd /tmp/ \
&& rm -rf /tmp/grpc \
&& ldconfig

# Install AWS SDK for C++ so we can use S3 storage.
# Do this before installing newer curl.
RUN mkdir /tmp/aws-sdk-cpp \
    && curl --location 'https://github.com/aws/aws-sdk-cpp/archive/1.8.48.tar.gz' \
        | tar --extract --gzip \
    && cd aws-sdk-cpp-1.8.48 \
    && mkdir build \
    && cd build \
    && cmake3 -DCMAKE_BUILD_TYPE=Release \
             -DBUILD_ONLY="s3" \
             -DCURL_DIR=/usr/lib64/ \
             -DCMAKE_C_FLAGS="-Wno-unused-variable -Wno-unused-parameter" \
             -DCMAKE_CXX_FLAGS="-Wno-unused-variable -Wno-unused-parameter" .. \
    && make --jobs "$(nproc)" install \
    && rm --recursive /tmp/aws-sdk-cpp

# Install curl
# Newer version needed for TRTIS.
RUN cd /tmp/ \
&& git clone -b curl-7_67_0 https://github.com/curl/curl.git \
&& cd curl \
&& mkdir build && cd build && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_SHARED_LIBS=ON .. && make install \
&& cp /usr/local/lib64/libcurl.so /usr/local/lib/libcurl.so \
&& ldconfig

# Install TensorRT-Inference-Server client libs.
RUN cd /tmp/ \
&& git clone -b v1.7.0 https://github.com/NVIDIA/tensorrt-inference-server.git tensorrt-inference-server \
# Disable building TRTIS python client.
&& sed -i '/add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/d' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt \
# Set version number.
&& sed -i 's/project (trtis-clients)/project(trtis-clients VERSION "0.0.0")/g' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt \
&& mkdir -p /tmp/tensorrt-inference-server/build/trtis-clients/build \
&& cd /tmp/tensorrt-inference-server/build/trtis-clients/build \
&& cmake3 -DCMAKE_INSTALL_PREFIX=/root/trtis -DCMAKE_PREFIX_PATH=/usr/lib64/ -DCURL_DIR=/usr/local/ .. \
&& make -j`nproc` \
&& make install \
&& cp /tmp/tensorrt-inference-server/src/core/constants.h /root/trtis/include/constants.h \
&& chmod 644 /root/trtis/include/constants.h \
&& echo '/root/trtis/lib/' > /etc/ld.so.conf.d/trtis.conf \
&& echo '/usr/local/lib'   > /etc/ld.so.conf.d/locallib.conf \
&& ldconfig \
# Fix bad header include references to 'src'.
&& find /root/trtis/include/ -type f -exec sed -i 's/#include "src\/clients\/c++\//#include "/g' {} \; \
&& find /root/trtis/include/ -type f -exec sed -i 's/#include "src\/core\//#include "/g' {} \; \
&& mkdir -p /root/trtis/cmake/ \
&& echo 'add_library(request SHARED IMPORTED) # or STATIC instead of SHARED'                  > /root/trtis/cmake/request-config.cmake \
&& echo 'set_target_properties(request PROPERTIES'                                           >> /root/trtis/cmake/request-config.cmake \
&& echo '                              IMPORTED_LOCATION "/root/trtis/lib/librequest.so"'    >> /root/trtis/cmake/request-config.cmake \
&& echo '                              INTERFACE_INCLUDE_DIRECTORIES "/root/trtis/include")' >> /root/trtis/cmake/request-config.cmake

ENV request_DIR=/root/trtis

WORKDIR /home/mpf/component_src

COPY --from=models /models/model.graphdef \
        $BUILD_DIR/plugin/TrtisDetection/models/ip_irv2_coco/1/model.graphdef

COPY . .

RUN build-component.sh

FROM nvcr.io/nvidia/tensorrtserver:19.04-py3 as openmpf_trtis_test_server

COPY --from=build_component /usr/lib64/libssl3.so /usr/lib64/libnspr4.so /usr/lib64/libnss3.so \
    /usr/lib64/libgtest*.so.0.0.0 /home/mpf/mpf-sdk-install/lib/ /usr/local/lib/
COPY --from=build_component /opt/opencv-3.4.7/lib64 /apps/install/opencv3.4.7/lib64/
COPY --from=build_component /home/mpf/component_build /Testing_MPF_FILES/
COPY --from=build_component /home/mpf/component_src/test/run_docker_tests.sh /Testing_MPF_FILES/test

ARG RUN_TESTS=false

RUN if [ "${RUN_TESTS,,}" == true ]; \
    then cd /Testing_MPF_FILES/test/ \
    && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Testing_MPF_FILES/plugin/TrtisDetection/lib/:/apps/install/opencv3.4.7/lib64/ \
    && ./run_docker_tests.sh \
; fi

FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV COMPONENT_LOG_NAME trtis-detection.log

# Must copy from openmpf_trtis_test_server so that Docker BuildKit does not skip test stage.
COPY --from=openmpf_trtis_test_server /Testing_MPF_FILES/plugin/TrtisDetection \
                                      $PLUGINS_DIR/TrtisDetection
COPY --from=openmpf_trtis_test_server /Testing_MPF_FILES/libmpfTrtisDetection.so \
                                      $PLUGINS_DIR/TrtisDetection/lib/

RUN cp $PLUGINS_DIR/TrtisDetection/lib/librequest.so /usr/local/lib/ && ldconfig

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRTIS Client Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
