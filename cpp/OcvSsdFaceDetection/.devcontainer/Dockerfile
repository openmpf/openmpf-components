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
from openmpf_cpp_component_build:gpu

RUN yum install --assumeyes \
               git gdb which mlocate \
               jasper-devel openexr-devel libwebp-devel \
               bzip2 libtiff-devel libdc1394-devel tbb-devel libv4l-devel \
               gstreamer-plugins-base-devel ffmpeg-devel \
               openblas openblas-devel lapack lapack-devel \
               gcc-gfortran gcc-c++

RUN echo '/usr/local/lib64' > /etc/ld.so.conf.d/usr_local_lib64.conf; \
    ldconfig;

RUN mkdir -p /tmp/dlib; \
    cd /tmp/dlib; \
    [[ -e dlib-19.20 ]] || curl --location 'http://dlib.net/files/dlib-19.20.tar.bz2' | tar --extract --bzip2; \
    cd dlib-19.20; \
    mkdir build; \
    cd build; \
    cmake3 -DBUILD_SHARED_LIBS:BOOL="1" ..; \
    cmake3 --build . --config Release; \
    make --jobs "$(nproc)" install; \
    ldconfig;

RUN ln -s /usr/bin/cmake3 /usr/bin/cmake;

LABEL org.label-schema.license="Mixed" \
      org.label-schema.name="OpenMPF OpenCV SSD Face Detection Dev GPU" \
      org.label-schema.schema-version="1.0" \
      org.label-schema.url="https://openmpf.github.io" \
      org.label-schema.vcs-url="https://github.com/openmpf/openmpf-components" \
      org.label-schema.vendor="MITRE"