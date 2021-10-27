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

FROM ${MODELS_REGISTRY}openmpf_trtis_detection_models:6.3 as models


FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_setup

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

# Install newer version of curl needed for AWS S3
RUN --mount=type=tmpfs,target=/tmp \
   cd /tmp; \
   git clone -b curl-7_76_1 https://github.com/curl/curl.git; \
   cd /tmp/curl; \
   mkdir build && cd build && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DBUILD_SHARED_LIBS=ON .. && make install; \
   echo '/usr/local/lib64' > /etc/ld.so.conf.d/locallib64.conf; \
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

ENV CMAKE_PREFIX_PATH="/opt/triton/lib64/cmake:${CMAKE_PREFIX_PATH}"


FROM build_setup as build_component

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
#
# CLion deployment settings (SFTP Remote Development):
# - [Connection] SSH configuration: root@localhost:2222
# - [Connection] Root path: /home/mpf/component_src
# - [Mappings] Deployment path: .
#
# Build command:
#   COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker build . --progress=plain --target=clion \
#   -t openmpf_trtis_detection:triton-clion
#
# Run command:
#   docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env \
#   openmpf_trtis_detection:triton-clion
########################################################################
FROM build_setup as clion

RUN --mount=type=tmpfs,target=/tmp \
    --mount=type=tmpfs,target=/var/cache/yum \
    yum install --assumeyes openssh-server; \
    yum clean all

RUN ssh-keygen -A

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/libexec/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion

# Set root user password for SSH access.
RUN echo "openmpf123" | passwd root --stdin

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]
########################################################################
# CLion
########################################################################


FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV LD_LIBRARY_PATH $PLUGINS_DIR/TrtisDetection/lib

# DEBUG
# Must copy from openmpf_trtis_test_server so that Docker BuildKit does not skip test stage.
# COPY --from=openmpf_trtis_test_server /Testing_MPF_FILES/plugin/TrtisDetection \
#                                      $PLUGINS_DIR/TrtisDetection
# COPY --from=openmpf_trtis_test_server /Testing_MPF_FILES/libmpfTrtisDetection.so \
#                                      $PLUGINS_DIR/TrtisDetection/lib/

COPY --from=build_component $BUILD_DIR/plugin/TrtisDetection \
                            $PLUGINS_DIR/TrtisDetection
COPY --from=build_component $BUILD_DIR/libmpfTrtisDetection.so \
                            $PLUGINS_DIR/TrtisDetection/lib/

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRITON Client Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
