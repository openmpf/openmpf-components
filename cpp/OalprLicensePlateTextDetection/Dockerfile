# syntax=docker/dockerfile:1.2

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
FROM ${BUILD_REGISTRY}openmpf_cpp_component_build:${BUILD_TAG} as build_component

RUN --mount=type=tmpfs,target=/var/cache/apt \
    --mount=type=tmpfs,target=/var/lib/apt/lists  \
    --mount=type=tmpfs,target=/tmp \
    apt-get update; \
    apt-get install --no-install-recommends -y pkg-config;

RUN --mount=type=tmpfs,target=/tmp \
    mkdir /tmp/leptonica; \
    cd /tmp/leptonica; \
    wget -O- 'https://github.com/DanBloomberg/leptonica/archive/1.75.0.tar.gz' \
        | tar --extract --gzip; \
    mkdir leptonica-1.75.0/build; \
    cd leptonica-1.75.0/build; \
    cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..; \
    make install --jobs "$(nproc)";

RUN --mount=type=tmpfs,target=/tmp \
    mkdir /tmp/tesseract; \
    cd /tmp/tesseract; \
    wget -O- 'https://github.com/tesseract-ocr/tesseract/archive/4.0.0.tar.gz' \
        | tar --extract --gzip; \
    mkdir tesseract-4.0.0/build; \
    cd tesseract-4.0.0/build; \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig cmake3 -DCMAKE_INSTALL_PREFIX:PATH=/usr/local ..; \
    make install --jobs "$(nproc)";

RUN --mount=type=tmpfs,target=/tmp \
    mkdir /tmp/oalpr; \
    cd /tmp/oalpr; \
    wget -O- 'https://github.com/openalpr/openalpr/archive/386895386781f04ef056e93dd70d9ec36f72d350.tar.gz' \
        | tar --extract --gzip; \
    cd openalpr-386895386781f04ef056e93dd70d9ec36f72d350/src; \
    # The following sed command will need to be modified when upgrading OALPR.
    # Line 101 of the OALPR CMakeLists.txt contains: FIND_PACKAGE( OpenCV REQUIRED )
    # That pulls in a bunch of OpenCV libraries that OALPR doesn't need.
    sed -i '101s/.*/  FIND_PACKAGE( OpenCV REQUIRED COMPONENTS opencv_core opencv_imgcodecs opencv_highgui opencv_objdetect opencv_video)/' \
            CMakeLists.txt; \
    mkdir build; \
    cd build; \
    cmake3 -DCMAKE_CXX_STANDARD=11 -DCMAKE_CXX_STANDARD_REQUIRED=ON -DWITH_DAEMON=OFF -DWITH_GPU_DETECTOR=false \
           -DCMAKE_INSTALL_PREFIX:PATH=/usr/local -DLeptonica_LIB=/usr/local/lib/libleptonica.so ..; \
    make install -j "$(nproc)"; \
    ldconfig

ARG RUN_TESTS
RUN --mount=target=. build-component.sh;

RUN if [ "${RUN_TESTS,,}" == true ]; then \
        cd $BUILD_DIR/test; \
        ./OalprLicensePlateTextDetectionTest; \
    fi


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
