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
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN yum -y install vim mlocate git openssl-devel wget \
&& mkdir -p /home/mpf/component_build/plugin/TrtisDetection/models/ip_irv2_coco/1 \
&& wget --load-cookies /tmp/cookies.txt \
"https://docs.google.com/uc?export=download&confirm=$(\
wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
'https://docs.google.com/uc?export=download&id=1EVYe2RHs7g2A8E2KLjVUXp-TV_ndjdtn' -O- \
| sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EVYe2RHs7g2A8E2KLjVUXp-TV_ndjdtn" \
-O /home/mpf/component_build/plugin/TrtisDetection/models/ip_irv2_coco/1/model.graphdef \
&& rm -rf /tmp/cookies.txt

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

# Install curl.
RUN cd /tmp/ \
&& git clone -b curl-7_67_0 https://github.com/curl/curl.git \
&& cd curl \
&& mkdir build && cd build && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. && make install && \
ldconfig


# Install TensorRT-Inference-Server client libs.
RUN cd /tmp/ \
&& git clone -b v1.7.0 https://github.com/NVIDIA/tensorrt-inference-server.git tensorrt-inference-server 2> /dev/null || (cd tensorrt-inference-server && git pull)

# Fix lower case protobuf package name.
RUN sed -i 's/find_package(Protobuf CONFIG REQUIRED)/find_package(protobuf CONFIG REQUIRED)/g' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt \
&& sed -i 's/add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/#add_subdirectory(\.\.\/\.\.\/src\/clients\/python src\/clients\/python)/g' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt

# Set version number.
RUN sed -i 's/project (trtis-clients)/project(trtis-clients VERSION "0.0.0")/g' /tmp/tensorrt-inference-server/build/trtis-clients/CMakeLists.txt


RUN mkdir -p /tmp/tensorrt-inference-server/build/trtis-clients/build && cd /tmp/tensorrt-inference-server/build/trtis-clients/build \
&& cmake3 -DCMAKE_INSTALL_PREFIX=/root/trtis .. \
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

COPY . .

RUN build-component.sh

FROM nvcr.io/nvidia/tensorrtserver:19.04-py3 as openmpf_trtis_test_server

ADD ./plugin-files/models /models/

COPY --from=build_component /usr/lib64/libssl3.so /usr/local/lib
COPY --from=build_component /usr/lib64/libnspr4.so /usr/local/lib
COPY --from=build_component /usr/lib64/libnss3.so /usr/local/lib
COPY --from=build_component /usr/lib64/libgtest*.so.0.0.0  /usr/local/lib/
COPY --from=build_component /opt/opencv-3.4.7/lib64 /apps/install/opencv3.4.7/lib64/
COPY --from=build_component /home/mpf/component_build /Testing_MPF_FILES/
COPY --from=build_component /home/mpf/component_build/plugin/TrtisDetection/lib/librequest.so /usr/local/lib/
COPY --from=build_component /home/mpf/mpf-sdk-install/lib /usr/local/lib

ARG RUN_TESTS=false

RUN if [ "${RUN_TESTS,,}" == true ]; \
then cd /opt/tensorrtserver/ && ./nvidia_entrypoint.sh \
&& cp -a /Testing_MPF_FILES/*.so* /usr/local/lib \
&& ln -s /usr/local/lib/libgtest.so.0.0.0 libgtest.so \
&& ln -s /usr/local/lib/libgtest.so.0.0.0 libgtest.so.0 \
&& ln -s /usr/local/lib/libgtest_main.so.0.0.0 libgtest_main.so \
&& ln -s /usr/local/lib/libgtest_main.so.0.0.0 libgtest_main.so.0 \
&& cp /Testing_MPF_FILES/*.so /opt/tensorrtserver/ \
&& ldconfig \
&& trtserver --model-store=/models 2> /Testing_MPF_FILES/server_logs.txt \
# Running test code alongside active server process.
& echo "Starting server and loading model files."\
&& cd /Testing_MPF_FILES/test/ \
&& LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Testing_MPF_FILES/plugin/TrtisDetection/lib/:/apps/install/opencv3.4.7/lib64/ \
# Wait for server to finish loading coco model files.
&& while : ;do \
       [[ -f "/Testing_MPF_FILES/server_logs.txt" ]] \
       && grep -q "Successfully loaded servable version" "/Testing_MPF_FILES/server_logs.txt" \
       && echo "Server ready for testing. Beginning detection tests." \
       && break ;\
       sleep 5 ;\
    done \
&& ./TrtisDetectionTest > /MPF_test_status.txt \
; fi

FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV COMPONENT_LOG_NAME trtis-detection.log

COPY --from=build_component $BUILD_DIR/plugin/TrtisDetection \
                            $PLUGINS_DIR/TrtisDetection

COPY --from=build_component $BUILD_DIR/libmpfTrtisDetection.so \
                            $PLUGINS_DIR/TrtisDetection/lib/



RUN cp $PLUGINS_DIR/TrtisDetection/lib/librequest.so /usr/local/lib/ && ldconfig

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF TRTIS Client Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
