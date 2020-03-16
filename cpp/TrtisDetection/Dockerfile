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

############################################################################
## Building Container
############################################################################

FROM openmpf_build as openmpf-dev

RUN yum -y install vim mlocate

# Install cares, protobuf, zlib, gRPC
# COPY fsroot/ /
WORKDIR /root

RUN git config --global --unset http.proxy

RUN git clone -b v1.20.0 https://github.com/grpc/grpc  && cd grpc && git submodule update --init --recursive \
&& mkdir -p /root/grpc/third_party/protobuf/cmake/build \
&& cd /root/grpc/third_party/protobuf/cmake/build \
&& cmake3 -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release .. \
&& make -j`nproc` \
&& make install \
&& cd /root/ \
&& rm -rf /root/grpc/third_party/protobuf \
&& mkdir -p /root/grpc/third_party/cares/cares/cmake/build \
&& cd /root/grpc/third_party/cares/cares/cmake/build \
&& cmake3 -DCMAKE_BUILD_TYPE=Release ../.. \
&& make -j`nproc` \
&& make install \
&& cd /root/ \
&& rm -rf /root/grpc/third_party/cares/cares \
# zlib already installed
&& rm -rf /root/grpc/third_party/zlib \
# protobuf already installed
&& rm -rf /root/grpc/third_party/protobuf \
# build and install gRPC
&& mkdir -p /root/grpc/cmake/build \
&& cd /root/grpc/cmake/build \
&& cmake3 -DgRPC_INSTALL=ON \
          -DgRPC_BUILD_TESTS=OFF \
          -DgRPC_PROTOBUF_PROVIDER=package \
          -DgRPC_ZLIB_PROVIDER=package \
          -DgRPC_CARES_PROVIDER=package \
          -DgRPC_SSL_PROVIDER=package \
          -DCMAKE_BUILD_TYPE=Release ../.. \
&& make -j`nproc` \
&& make install \
&& cd /root/ \
&& rm -rf /root/grpc \
&& ldconfig

# Install curl.
RUN cd /root/ \
&& git clone -b curl-7_67_0 https://github.com/curl/curl.git \
&& cd curl \
&& mkdir build && cd build && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. && make install && \
ldconfig


# Install TensorRT-Inference-Server client libs.
RUN cd /root/ \
&& git clone -b v1.7.0 https://github.com/NVIDIA/tensorrt-inference-server.git tensorrt-inference-server 2> /dev/null || (cd tensorrt-inference-server; git pull)

# Fix lower case protobuf package name.
RUN sed -i 's/find_package(Protobuf CONFIG REQUIRED)/find_package(protobuf CONFIG REQUIRED)/g' /root/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt \
&& sed -i 's/add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/#add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/g' /root/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt

# Set version number.
RUN sed -i 's/project (trtis-clients)/project(trtis-clients VERSION "0.0.0")/g' /root/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt
# RUN echo "$(cat /root/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt)"


RUN mkdir -p /root/tensorrt-inference-server/build/trtis-clients/build && cd /root/tensorrt-inference-server/build/trtis-clients/build \
&& cmake3 -DCMAKE_INSTALL_PREFIX=/root/trtis .. \
&& make -j`nproc` \
&& make install \
&& cp /root/tensorrt-inference-server/src/core/constants.h /root/trtis/include/constants.h \
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
&& cd /root

# && rm -fr /root/tensorrt-inference-server

ENV request_DIR=/root/trtis

RUN cd /root/ \
&& git clone -b develop https://github.com/openmpf/openmpf-cpp-component-sdk.git 2> /dev/null || (cd openmpf-cpp-component-sdk; git pull) \
&& mkdir -p /root/openmpf-cpp-component-sdk/build \
&& cd /root/openmpf-cpp-component-sdk/build \
&& cmake3 .. \
&& make -j`nproc` \
&& make install \
&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/mpf-sdk-install/lib
RUN updatedb

RUN cd /root/ \
&& git clone -b develop https://github.com/openmpf/openmpf-components.git 2> /dev/null \
&& mkdir openmpf-components/cpp/TrtisDetection

COPY . /root/openmpf-components/cpp/TrtisDetection
RUN cd /root/openmpf-components/cpp/TrtisDetection && mkdir build && cd build && cmake3 .. && make install

############################################################################
## Building Test Container
############################################################################

FROM nvcr.io/nvidia/tensorrtserver:19.04-py3 as test_server

ADD ./plugin-files/models /models/

COPY --from=openmpf-dev /usr/lib64/ /MPF_lib64/
COPY --from=openmpf-dev /usr/local/lib /MPF_lib/
COPY --from=openmpf-dev /apps/install/opencv3.4.7/lib64/ /apps/install/opencv3.4.7/lib64/
COPY --from=openmpf-dev /root/openmpf-components/cpp/TrtisDetection/build /Testing_MPF_FILES/
COPY --from=openmpf-dev /root/tensorrt-inference-server/build/trtis-clients/build/src/clients/c++/library/librequest.so /usr/local/lib/
COPY --from=openmpf-dev /root/mpf-sdk-install/lib /usr/local/lib

RUN  cd /opt/tensorrtserver/ && ./nvidia_entrypoint.sh && cd /Testing_MPF_FILES/test/ && trtserver --model-store=/models \
& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Testing_MPF_FILES/plugin/TrtisDetection/lib/ && cp /Testing_MPF_FILES/*.so /usr/local/lib/ \
&& export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/install/opencv3.4.7/lib64/ \
&& cp /MPF_lib64/libssl3.so /usr/local/lib \
&& cp /MPF_lib64/libnspr4.so /usr/local/lib \
&& cp /MPF_lib64/libnss3.so /usr/local/lib \
&& ldconfig \
&& cp /Testing_MPF_FILES/*.so /opt/tensorrtserver/ \
&& cd /Testing_MPF_FILES/test/ && ldconfig && sleep 30 && ./TrtisDetectionTest > /MPF_test_status.txt


FROM openmpf-dev
COPY --from=test_server  /MPF_test_status.txt  /MPF_test_status.txt
RUN echo "$(cat /MPF_test_status.txt)" && rm /MPF_test_status.txt


