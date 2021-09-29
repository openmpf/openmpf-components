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

FROM centos:7 as download_weights
# Download weights in separate stage so they don't need to be re-downloaded when base images change.
RUN mkdir /weights \
    && cd /weights \
    && curl --location --remote-name --fail \
        https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights \
    && curl --location --remote-name --fail \
        https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights


# Download library source code so it doesn't need to be re-downloaded when base images change
FROM centos:7 as download_library_source

RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=tmpfs,target=/var/cache/yum \
    yum install --assumeyes bzip2; \
    yum clean all

RUN mkdir --parents /apps/source/dlib-sources; \
    curl --location 'http://dlib.net/files/dlib-19.20.tar.bz2' \
        | tar --extract --bzip2 --directory /apps/source/dlib-sources

FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=tmpfs,target=/var/cache/yum \
     yum install --assumeyes --nogpgcheck \
        git gdb which mlocate bzip2 \
        cuda-cufft-dev-10-2 cuda-npp-dev-10-2 \
        libtool openssl-devel rapidjson-devel \
        # gcc4.9.4 dependancies
        gmp-devel mpfr-devel libmpc-devel; \
    # add libb64 needed by Triton
    rpm --nosignature -i http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-libs-1.2.1-2.1.el7.art.x86_64.rpm; \
    rpm --nosignature -i http://www6.atomicorp.com/channels/atomic/centos/7/x86_64/RPMS/libb64-devel-1.2.1-2.1.el7.art.x86_64.rpm; \
    # some debug messages support for system calls (e.g. nmap)
    rpm --nosignature -i http://download.sipxcom.org/pub/epel/6/x86_64/libexplain-0.52.D002-1.el6.x86_64.rpm; \
    rpm --nosignature -i http://download.sipxcom.org/pub/epel/6/x86_64/libexplain-devel-0.52.D002-1.el6.x86_64.rpm; \
    yum clean all;

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
ENV CXXFLAGS=-std=c++11
ENV LIBRARY_PATH=/usr/local/lib:/usr/local/lib64
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib64

# install updated cmake 3.18, configure existing opencv
RUN curl --location 'https://github.com/Kitware/CMake/releases/download/v3.21.1/cmake-3.21.1-linux-x86_64.tar.gz' \
        | tar --extract --gzip --directory=/usr --strip-components=1; \
    ln -sf /usr/bin/cmake /usr/bin/cmake3; \
    echo '/opt/opencv-4.5.0/lib64' > /etc/ld.so.conf.d/opencv.conf; \
    ldconfig; \
    ln --symbolic '/opt/opencv-4.5.0/include/opencv4/opencv2' /usr/local/include/opencv2;

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
          # GPU only needed if using cuda_shm
          -DTRITON_ENABLE_GPU=ON \
          -DTRITON_ENABLE_CC_GRPC=ON \
          -DTRITON_ENABLE_CC_HTTP=OFF \
          -DTRITON_ENABLE_PERF_ANALYZER=OFF \
          -DTRITON_ENABLE_PYTHON_HTTP=OFF \
          -DTRITON_ENABLE_PYTHON_GRPC=OFF \
          -DTRITON_ENABLE_EXAMPLES=OFF \
          -DTRITON_ENABLE_TESTS=OFF \
          ..; \
    Protobuf_DIR=/tmp/triton/client/build/third-party/protobuf \
    make -j "$(nproc)"; \
    # fix paths for 3rd party libraries installs
    rm -rf /opt/triton/lib64/lib64; \
    rsync -aL /tmp/triton/client/build/third-party/c-ares/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/c-ares/lib/ /opt/triton/lib64/; \
    rsync -aL /tmp/triton/client/build/third-party/c-ares/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/client/build/third-party/curl/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/curl/lib64/ /opt/triton/lib64/; \
    rsync -aL /tmp/triton/client/build/third-party/curl/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/lib/ /opt/triton/lib64/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/lib64/ /opt/triton/lib64/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/bin/ /opt/triton/bin/; \
    rsync -aL /tmp/triton/client/build/third-party/grpc/share/ /opt/triton/share/; \
    rsync -aL /tmp/triton/client/build/third-party/protobuf/include/ /opt/triton/include/; \
    rsync -aL /tmp/triton/client/build/third-party/protobuf/lib64/ /opt/triton/lib64/; \
    rsync -aL /tmp/triton/client/build/third-party/protobuf/bin/ /opt/triton/bin/; \
    ln -s /opt/triton/lib64 /opt/triton/lib; \
    echo '/opt/triton/lib64' > /etc/ld.so.conf.d/triton.conf; \
    ldconfig; \
    rm --recursive /tmp/triton;

ENV CMAKE_PREFIX_PATH="/opt/triton/lib64/cmake:${CMAKE_PREFIX_PATH}"

ENV TRITON_SUPPORT=ON


ARG RUN_TESTS=false

# DEBUG
COPY --from=download_library_source /apps/source/dlib-sources \
                                    /apps/source/dlib-sources

# DEBUG
COPY . .

# DEBUG
RUN build-component.sh; \
    if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test; ./OcvYoloDetectionTest; fi

# ORIG
#RUN --mount=target=. \
#    --mount=from=download_library_source,source=/apps/source/dlib-sources,target=/apps/source/dlib-sources \
#    build-component.sh; \
#    if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test; ./OcvYoloDetectionTest; fi


#FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component
#
#COPY --from=download_weights /weights $PLUGINS_DIR/OcvYoloDetection/models
#
#ENV LD_LIBRARY_PATH $PLUGINS_DIR/OcvYoloDetection/lib
#
#COPY --from=build_component $BUILD_DIR/plugin/OcvYoloDetection $PLUGINS_DIR/OcvYoloDetection
#
#COPY --from=build_component $BUILD_DIR/libmpfOcvYoloDetection.so \
#                            $PLUGINS_DIR/OcvYoloDetection/lib/
#
#LABEL org.label-schema.license="Mixed" \
#      org.label-schema.name="OpenMPF OpenCV Yolo Detection" \
#      org.label-schema.schema-version="1.0" \
#      org.label-schema.url="https://openmpf.github.io" \
#      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
#      org.label-schema.vendor="MITRE"
