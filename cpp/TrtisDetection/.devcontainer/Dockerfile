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


FROM openmpf_cpp_component_build:latest

RUN yum -y install vim  openssl-devel wget git gdb which mlocate curl-devel;

RUN echo '/usr/local/lib64' > /etc/ld.so.conf.d/usr_local_lib64.conf; \
    ldconfig;

# Install cares, protobuf, zlib, gRPC
WORKDIR /tmp

RUN git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc  && cd grpc && git submodule update --init --recursive \
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
&& make install

RUN cd /tmp/ \
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

#Install AWS SDK for C++ so we can use S3 storage
# needs to happen before adding newer curl
RUN cd /tmp/ \
&& git clone https://github.com/aws/aws-sdk-cpp.git aws-sdk-cpp \
&& mkdir -p /tmp/aws-sdk-cpp/build \
&& cd /tmp/aws-sdk-cpp/build \
&& cmake3 -D CMAKE_BUILD_TYPE=Release \
         -D BUILD_ONLY="s3" \
         -DCURL_DIR=/usr/lib64/ \
         -DCMAKE_C_FLAGS="-Wno-unused-variable -Wno-unused-parameter" \
         -DCMAKE_CXX_FLAGS="-Wno-unused-variable -Wno-unused-parameter" .. \
&& make -j`nproc` \
&& make install \
&& cd /tmp/ \
&& rm -fr /tmp/aws-sdk-cpp

# Install curl
# Updated version needed for trtis.
RUN cd /tmp/ \
&& git clone -b curl-7_67_0 https://github.com/curl/curl.git \
&& cd curl \
&& mkdir build && cd build \
&& cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_SHARED_LIBS=ON .. \
&& make install \
&& cp /usr/local/lib64/libcurl.so /usr/local/lib/libcurl.so \
&& ldconfig

# Install TensorRT-Inference-Server client libs.
RUN cd /tmp/ \
&& git clone -b v1.7.0 https://github.com/NVIDIA/tensorrt-inference-server.git tensorrt-inference-server

# Disable building TRTIS python client.
RUN sed -i '/add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/d' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt
# Set version number.
RUN sed -i 's/project (trtis-clients)/project(trtis-clients VERSION "0.0.0")/g' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt

RUN mkdir -p /tmp/tensorrt-inference-server/build/trtis-clients/build \
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
&& echo '                              INTERFACE_INCLUDE_DIRECTORIES "/root/trtis/include")' >> /root/trtis/cmake/request-config.cmake \
# Prepare libraries and model file for potential testing.
&& mkdir -p /home/mpf/transfer/lib \
&& cp /usr/lib64/libssl3.so  /home/mpf/transfer/lib \
&& cp /usr/lib64/libnspr4.so /home/mpf/transfer/lib \
&& cp /usr/lib64/libnss3.so  /home/mpf/transfer/lib \
&& cp /usr/lib64/libgtest*.so.0.0.0  /home/mpf/transfer/lib \
&& cp -r /home/mpf/mpf-sdk-install/lib/* /home/mpf/transfer/lib

ENV request_DIR=/root/trtis

ENV COMPONENT_LOG_NAME trtis-detection.log

RUN rm -fr /usr/bin/cmake \
&& ln -s /usr/bin/cmake3 /usr/bin/cmake;

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRTIS Client Detection Dev" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
