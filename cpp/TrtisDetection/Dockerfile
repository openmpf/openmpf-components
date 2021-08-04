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
ARG MODELS_REGISTRY=openmpf/

FROM ${MODELS_REGISTRY}openmpf_trtis_detection_models:6.2 as models


FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=tmpfs,target=/var/cache/yum \
    yum update --assumeyes; \
    yum --assumeyes --nogpgcheck install \
        which file git \
        cuda-cufft-dev-10-2 cuda-npp-dev-10-2 \
        libtool openssl-devel rapidjson-devel \
        # gcc4.9.4 dependancies
        gmp-devel mpfr-devel libmpc-devel; \
    # add libb64 needed by Triton
    rpm --nosignature -i http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm; \
    rpm --nosignature -i http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm; \
    yum clean all; \
    rm --recursive /var/cache/yum/*;

# install gcc 4.9.4
RUN --mount=type=tmpfs,target=/tmp \
   cd /tmp; \
   curl https://ftp.gnu.org/gnu/gcc/gcc-4.9.4/gcc-4.9.4.tar.gz | tar --extract --gzip; \
   mkdir -p /tmp/gcc-4.9.4/build; \
   cd /tmp/gcc-4.9.4/build; \
   ../configure --enable-languages=c,c++ --disable-multilib; \
   make -j$(nproc) && make install; \
   echo '/usr/local/lib64'   > /etc/ld.so.conf.d/locallib64.conf; \
   ldconfig; \
   rm --recursive /tmp/gcc-4.9.4;

ENV CC=/usr/local/bin/gcc
ENV CXX=/usr/local/bin/g++


# install updated cmake 3.18, configure existing opencv
RUN curl --location 'https://github.com/Kitware/CMake/releases/download/v3.18.6/cmake-3.18.6-Linux-x86_64.tar.gz' \
        | tar --extract --gzip --directory=/usr --strip-components=1; \
    ln -sf /usr/bin/cmake /usr/bin/cmake3; \
    echo '/opt/opencv-4.5.0/lib64' > /etc/ld.so.conf.d/opencv.conf; \
    ldconfig; \
    ln --symbolic '/opt/opencv-4.5.0/include/opencv4/opencv2' /usr/local/include/opencv2;

WORKDIR /tmp

# Install newer version of curl needed for AWS S3
RUN --mount=type=tmpfs,target=/tmp \
   cd /tmp; \
   git clone -b curl-7_76_1 https://github.com/curl/curl.git; \
   cd /tmp/curl; \
   mkdir build && cd build && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_SHARED_LIBS=ON .. && make install; \
   ln -s /usr/local/lib64/libcurl.so /usr/local/lib/libcurl.so; \
   mv /usr/local/lib64/libcurl.so /lib64/libcurl.so.7.76.1; \
   ln -s /lib64/libcurl.so.7.76.1 /usr/local/lib64/libcurl.so; \
   echo '/usr/local/lib64'   > /etc/ld.so.conf.d/locallib64.conf; \
   ldconfig; \
   rm --recursive /tmp/curl;

# Install AWS SDK for C++ so we can use S3 storage.
RUN --mount=type=tmpfs,target=/tmp \
    mkdir /tmp/aws-sdk-cpp; \
    cd /tmp/aws-sdk-cpp; \
    curl --location 'https://github.com/aws/aws-sdk-cpp/archive/1.8.179.tar.gz' \
        | tar --extract --gzip; \
    cd aws-sdk-cpp-1.8.179; \
    mkdir build; \
    cd build; \
    cmake3 -DCMAKE_BUILD_TYPE=Release \
           -DBUILD_ONLY="s3" \
           -DBUILD_SHARED_LIBS=ON \
           -DENABLE_TESTING=OFF \
           -DCURL_DIR=/usr/local/lib64/ ..; \
    make --jobs "$(nproc)" install; \
    rm --recursive /tmp/aws-sdk-cpp;

# Triton Client build
ARG TRITON_REPO_TAG=r21.04
RUN --mount=type=tmpfs,target=/tmp \
    mkdir -p /tmp/triton; \
    cd /tmp/triton; \
    git clone -b ${TRITON_REPO_TAG}  https://github.com/triton-inference-server/server.git; \
    # build the client library and examples
    mkdir -p /tmp/triton/server/builddir; \
    cd /tmp/triton/server/builddir; \
    # remove python client source so it doesn't try to build it
    sed -i '/add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/d' /tmp/triton/server/build/client/CMakeLists.txt; \
    # fix c++14 idiom for clearing queue
    sed -i 's/data_buffers_ = {};/data_buffers_ = std::queue<std::pair<uint8_t *, size_t>>();/g' /tmp/triton/server/src/clients/c++/library/http_client.cc; \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX:PATH=/opt/triton/ \
          -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_REPO_TAG} \
          -DTRITON_CORE_REPO_TAG:STRING=${TRITON_REPO_TAG} \
          -DTRITON_CLIENT_SKIP_EXAMPLES=ON \
          -DTRITON_ENABLE_PYTHON=OFF \
          -DTRITON_ENABLE_METRICS=OFF \
          -DTRITON_ENABLE_METRICS_GPU=OFF \
          -DTRITON_ENABLE_LOGGING=OFF \
          -DTRITON_ENABLE_STATS=OFF \
          -DTRITON_ENABLE_GPU=OFF \
          -DTRITON_ENABLE_GRPC=ON \
          -DTRITON_ENABLE_HTTP=ON \
          -DTRITON_CURL_WITHOUT_CONFIG=ON \
          ../build; \
    # simlink curl's lib64 to lib so we can find curl.a
    mkdir -p /tmp/triton/server/builddir/curl/install; \
    ln -s /tmp/triton/server/builddir/curl/install/lib64 /tmp/triton/server/builddir/curl/install/lib; \
    make -j "$(nproc)" client; \
    # move missing libraries to install in /opt/triton
    rsync -aL /tmp/triton/server/builddir/curl/install/ /opt/triton/; \
    rsync -aL /tmp/triton/server/builddir/protobuf/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/server/builddir/protobuf/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/server/builddir/protobuf/lib64/ /opt/triton/lib64/; \
    rsync -aL /tmp/triton/server/builddir/grpc/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/server/builddir/grpc/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/server/builddir/grpc/lib/ /opt/triton/lib/; \
    rsync -aL /tmp/triton/server/builddir/grpc/lib64/ /opt/triton/lib64; \
    rsync -aL /tmp/triton/server/builddir/grpc/share/ /opt/triton/share/; \
    rsync -aL /tmp/triton/server/builddir/c-ares/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/server/builddir/c-ares/include/ /opt/triton/include/c-ares/; \
    rsync -aL /tmp/triton/server/builddir/c-ares/lib/ /opt/triton/lib; \
    echo '/opt/triton/lib' > /etc/ld.so.conf.d/triton.conf; \
    ldconfig; \
    rm --recursive /tmp/triton;

# set Triton version correspoding to r21.04
ENV TRITON_VERSION=2.9.0
ENV request_DIR=/root/trtis

WORKDIR /home/mpf/component_src

COPY --from=models /models/model.graphdef \
        $BUILD_DIR/plugin/TrtisDetection/models/ip_irv2_coco/1/model.graphdef

COPY . .

RUN build-component.sh

FROM nvcr.io/nvidia/tritonserver:21.04-py3 as openmpf_trtis_test_server

COPY --from=build_component /usr/lib64/libssl3.so /usr/lib64/libnspr4.so /usr/lib64/libnss3.so \
    /usr/lib64/libgtest*.so.0.0.0 /home/mpf/mpf-sdk-install/lib/ /usr/local/lib/
COPY --from=build_component /opt/opencv-4.5.0/lib64 /apps/install/opencv4.5.0/lib64/
COPY --from=build_component /home/mpf/component_build /Testing_MPF_FILES/
COPY --from=build_component /home/mpf/component_src/test/run_docker_tests.sh /Testing_MPF_FILES/test

ARG RUN_TESTS=false

RUN if [ "${RUN_TESTS,,}" == true ]; \
    then cd /Testing_MPF_FILES/test/ \
    && LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Testing_MPF_FILES/plugin/TrtisDetection/lib/:/apps/install/opencv4.5.0/lib64/ \
    && ./run_docker_tests.sh \
; fi

FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV LD_LIBRARY_PATH $PLUGINS_DIR/TrtisDetection/lib

# Must copy from openmpf_trtis_test_server so that Docker BuildKit does not skip test stage.
COPY --from=openmpf_trtis_test_server /Testing_MPF_FILES/plugin/TrtisDetection \
                                      $PLUGINS_DIR/TrtisDetection
COPY --from=openmpf_trtis_test_server /Testing_MPF_FILES/libmpfTrtisDetection.so \
                                      $PLUGINS_DIR/TrtisDetection/lib/

RUN cp $PLUGINS_DIR/TrtisDetection/lib/librequest.so /usr/local/lib/ && ldconfig

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRITON Client Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
