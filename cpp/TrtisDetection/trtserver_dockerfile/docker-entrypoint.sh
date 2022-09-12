#!/usr/bin/env bash

#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed don an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

# TRTIS 1.1.0 (nvcr.io/nvidia/tensorrtserver:19.04-py3) has a bug that occurs
# when creating the container using "docker-compose up", pressing ctrl-C to
# stop the deployment (without running "docker-compose down"), and then
# running "docker-compose up" again:
#
#  trtis-detection-server_1     | WARNING: The NVIDIA Driver was not detected.  GPU functionality will not be available.
#  trtis-detection-server_1     |    Use 'nvidia-docker run' to start this container; see
#  trtis-detection-server_1     |    https://github.com/NVIDIA/nvidia-docker/wiki/nvidia-docker .
#  trtis-detection-server_1     | ln: failed to create symbolic link '/opt/tensorrtserver/lib/libcuda.so.1': File exists
#  openmpf_trtis-detection-server_1 exited with code 1
#
# This only happens when running in CPU mode ("NVIDIA_VISIBLE_DEVICES=").
# To address this, we ensure that the symlinks created by the previous run of
# the "nvidia_entrypoint.sh" script are removed before running it again.

if [[ "$(find /usr -name libcuda.so.1 | grep -v "compat") " == " " || "$(ls /dev/nvidiactl 2>/dev/null) " == " " ]]; then
  rm -f /opt/tensorrtserver/lib/libcuda.so.1
  rm -f /opt/tensorrtserver/lib/libnvidia-ml.so.1
  rm -f /opt/tensorrtserver/lib/libnvidia-fatbinaryloader.so.${CUDA_DRIVER_VERSION}
fi

exec /opt/tensorrtserver/nvidia_entrypoint.sh "$@"
