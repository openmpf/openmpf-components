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
FROM ${BUILD_REGISTRY}openmpf_build:${BUILD_TAG} as openmpf_build


FROM ${BUILD_REGISTRY}openmpf_java_component_build:${BUILD_TAG} as build_component

RUN --mount=from=openmpf_build,source=/home/mpf/openmpf-projects/openmpf-build-tools/mpf-maven-deps.tar.gz,target=/root/mpf-maven-deps.tar.gz \
    tar --extract --file /root/mpf-maven-deps.tar.gz --directory /root/.m2/repository/

COPY pom.xml pom.xml

# Download dependencies before copying in the rest of the source code so that maven doesn't have to re-download
# dependencies every time the source code changes.
RUN mvn org.apache.maven.plugins:maven-dependency-plugin:3.1.1:go-offline;

COPY . .

ARG RUN_TESTS=false

RUN if [ "${RUN_TESTS,,}" == true ]; then \
        mvn package -Dmpf.assembly.format=dir; \
    else \
        mvn package -Dmpf.assembly.format=dir -Dmaven.test.skip=true; \
    fi



FROM ${BUILD_REGISTRY}openmpf_java_executor:${BUILD_TAG} as install_component


COPY --from=build_component /home/mpf/component_src/target/plugin-packages/SphinxSpeechDetection/SphinxSpeechDetection \
                            $PLUGINS_DIR/SphinxSpeechDetection


LABEL org.label-schema.license="Apache 2.0" \
      org.label-schema.name="OpenMPF Sphinx Speech Detection" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"
