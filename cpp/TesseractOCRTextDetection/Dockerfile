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


RUN yum install -y http://rpms.remirepo.net/enterprise/remi-release-7.rpm \
    && yum --enablerepo=remi install -y ImageMagick ImageMagick-c++ \
         ImageMagick-c++-devel ImageMagick-devel ImageMagick-libs \
         libjpeg libpng libtiff zlib ghostscript \
    && yum clean all \
    && rm -rf /var/cache/yum/*


RUN mkdir -p /tmp/leptonica \
    && cd /tmp/leptonica \
    && curl -SL 'https://github.com/DanBloomberg/leptonica/archive/1.75.0.tar.gz' \
        | tar --extract --gzip \
    && mkdir -p leptonica-1.75.0/build \
    && cd leptonica-1.75.0/build \
    && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. \
    && make install -j $(nproc) \
    && rm -rf /tmp/leptonica


RUN mkdir -p /tmp/tesseract \
    && cd /tmp/tesseract \
    && curl -SL 'https://github.com/tesseract-ocr/tesseract/archive/4.0.0.tar.gz' \
        | tar --extract --gzip \
    && mkdir -p tesseract-4.0.0/build \
    && cd tesseract-4.0.0/build \
    && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. \
    && make install -j $(nproc) \
    && rm -rf /tmp/tesseract

COPY . .

RUN build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./TesseractOCRTextDetectionTest; fi



FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} as install_component

RUN yum install -y http://rpms.remirepo.net/enterprise/remi-release-7.rpm \
    && yum --enablerepo=remi install -y ImageMagick \
    && yum clean all \
    && rm -rf /var/cache/yum/*

ENV LD_LIBRARY_PATH $PLUGINS_DIR/TesseractOCRTextDetection/lib

COPY --from=build_component $BUILD_DIR/plugin/TesseractOCRTextDetection $PLUGINS_DIR/TesseractOCRTextDetection

COPY --from=build_component $BUILD_DIR/libmpfTesseractOCRTextDetection.so \
                            $PLUGINS_DIR/TesseractOCRTextDetection/lib/

COPY --from=build_component $BUILD_DIR/tessdata_model_updater \
                            /opt/mpf/tessdata_model_updater

ENV LD_LIBRARY_PATH $PLUGINS_DIR/TesseractOCRTextDetection/lib

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Tesseract OCR Text Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
