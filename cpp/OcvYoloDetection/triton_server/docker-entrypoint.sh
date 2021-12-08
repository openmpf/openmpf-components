#!/usr/bin/env bash

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
# distributed under the License is distributed don an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

cd /model-gen/yolo

./gen-engine-416
LD_PRELOAD="/plugins/libyolo416layerplugin.so"

./gen-engine-608
LD_PRELOAD="$LD_PRELOAD:/plugins/libyolo608layerplugin.so"

exec env LD_PRELOAD="$LD_PRELOAD" \
    tritonserver \
    --model-repository=/models \
    --strict-model-config=false \
    --model-control-mode=explicit \
    --load-model=yolo-416 \  # GPU
    --load-model=yolo-608 \  # GPU
    # --load-model=ip_irv2_coco \  # CPU
    --log-verbose=1 \  # (optional)
    --grpc-infer-allocation-pool-size=16
