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
        pkg-config libmagick++-dev libboost-filesystem-dev libpango1.0-dev \
        libboost-regex-dev libboost-locale-dev ghostscript; \
    ln --symbolic /usr/include/x86_64-linux-gnu/ImageMagick-6/magick/magick-baseconfig.h \
                  /usr/include/ImageMagick-6/magick/magick-baseconfig.h


RUN --mount=type=tmpfs,target=/tmp \
    mkdir -p /tmp/leptonica \
    && cd /tmp/leptonica \
    && wget -O- 'https://github.com/DanBloomberg/leptonica/archive/1.75.0.tar.gz' \
        | tar --extract --gzip \
    && mkdir -p leptonica-1.75.0/build \
    && cd leptonica-1.75.0/build \
    && cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local .. \
    && make install -j $(nproc) \
    && ldconfig


RUN --mount=type=tmpfs,target=/tmp \
    mkdir -p /tmp/tesseract \
    && cd /tmp/tesseract \
    && wget -O- 'https://github.com/tesseract-ocr/tesseract/archive/5.0.0.tar.gz' \
        | tar --extract --gzip \
    && mkdir -p tesseract-5.0.0/build \
    && cd tesseract-5.0.0/build \
    && PKG_CONFIG_PATH=/usr/local/lib/pkgconfig cmake3 \
        -DCMAKE_INSTALL_PREFIX:PATH=/usr/local \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        .. \
    && make install -j $(nproc) \
    && cp /tmp/tesseract/tesseract-5.0.0/src/ccutil/unicharset.h /usr/local/include/tesseract/ \
    && cp /tmp/tesseract/tesseract-5.0.0/src/ccutil/errcode.h /usr/local/include/tesseract/ \
    && cp /tmp/tesseract/tesseract-5.0.0/src/ccutil/helpers.h /usr/local/include/tesseract/ \
    && cp /tmp/tesseract/tesseract-5.0.0/src/ccutil/serialis.h /usr/local/include/tesseract/ \
    && cp /tmp/tesseract/tesseract-5.0.0/src/ccutil/unicharmap.h /usr/local/include/tesseract/ \
    && cp /tmp/tesseract/tesseract-5.0.0/src/arch/intsimdmatrix.h /usr/local/include/tesseract/ \
    && ldconfig


RUN --mount=target=. build-component.sh

ARG RUN_TESTS=false
RUN if [ "${RUN_TESTS,,}" == true ]; then cd $BUILD_DIR/test && ./TesseractOCRTextDetectionTest; fi



FROM ${BUILD_REGISTRY}openmpf_cpp_executor:${BUILD_TAG} AS install_component

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    apt-get install --no-install-recommends -y ghostscript libmagickcore-6.q16-6

ENV LD_LIBRARY_PATH=$PLUGINS_DIR/TesseractOCRTextDetection/lib

COPY --from=build_component $BUILD_DIR/plugin/TesseractOCRTextDetection $PLUGINS_DIR/TesseractOCRTextDetection

COPY --from=build_component $BUILD_DIR/libmpfTesseractOCRTextDetection.so \
                            $PLUGINS_DIR/TesseractOCRTextDetection/lib/

# COPY --from=build_component $BUILD_DIR/tessdata_model_updater \
#                             /opt/mpf/tessdata_model_updater

ENV LD_LIBRARY_PATH=$PLUGINS_DIR/TesseractOCRTextDetection/lib

LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Tesseract OCR Text Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
