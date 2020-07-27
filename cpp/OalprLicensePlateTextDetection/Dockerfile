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

RUN yum install --assumeyes libjpeg-devel libpng-devel libtiff-devel zlib; \
    yum clean all; \
    rm --recursive /var/cache/yum/*

RUN mkdir /tmp/leptonica; \
    cd /tmp/leptonica; \
    curl --location 'https://github.com/DanBloomberg/leptonica/archive/1.75.0.tar.gz' \
        | tar --extract --gzip; \
    mkdir leptonica-1.75.0/build; \
    cd leptonica-1.75.0/build; \
    cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..; \
    make install --jobs "$(nproc)"; \
    rm --recursive /tmp/leptonica;

RUN mkdir /tmp/tesseract; \
    cd /tmp/tesseract; \
    curl --location 'https://github.com/tesseract-ocr/tesseract/archive/4.0.0.tar.gz' \
        | tar --extract --gzip; \
    mkdir tesseract-4.0.0/build; \
    cd tesseract-4.0.0/build; \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..; \
    make install --jobs "$(nproc)"; \
    rm --recursive /tmp/tesseract;

RUN mkdir /tmp/oalpr; \
    cd /tmp/oalpr; \
    curl --location 'https://github.com/openalpr/openalpr/archive/6ead2fb8f72ac1929a2e04912c5795af0bf37761.tar.gz' \
        | tar --extract --gzip; \
    cd openalpr-6ead2fb8f72ac1929a2e04912c5795af0bf37761/src; \
    mkdir build; \
    cd build; \
    cmake3 -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DWITH_DAEMON=OFF \
           -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DLeptonica_LIB=/usr/local/lib/libleptonica.so ..; \
    make install -j "$(nproc)"; \
    rm --recursive /tmp/oalpr

COPY . .

RUN build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./OalprLicensePlateTextDetectionTest; fi



FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

ENV LD_LIBRARY_PATH $PLUGINS_DIR/OalprLicensePlateTextDetection/lib

COPY --from=build_component $BUILD_DIR/plugin/OalprLicensePlateTextDetection \
                            $PLUGINS_DIR/OalprLicensePlateTextDetection

COPY --from=build_component $BUILD_DIR/libmpfOALPRLicensePlateTextDetection.so \
                            $PLUGINS_DIR/OalprLicensePlateTextDetection/lib/


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF OALPR License Plate Text Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
