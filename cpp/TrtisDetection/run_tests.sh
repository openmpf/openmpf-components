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
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

set -o errexit -o pipefail -o xtrace

cleanup() {
    if [ "$server_cid" ]; then
        docker stop "$server_cid" ||:
    fi
    if [ "$trt_test_net" ]; then
        docker network rm "$trt_test_net"
    fi
}

trap cleanup EXIT

script_dir=$(dirname "$0")
image_id=$(docker build "$script_dir" --target build_component --quiet \
            --build-arg BUILD_REGISTRY --build-arg BUILD_TAG)

trt_test_net="trt_test_net_$RANDOM"
docker network create "$trt_test_net"

server_cid=$(docker run --rm --detach --network "$trt_test_net" --network-alias trtserver \
                "${BUILD_REGISTRY}openmpf_trtis_detection_server:${BUILD_TAG:-latest}" \
                trtserver --model-store=/models)

until docker logs "$server_cid" |& grep -q 'Successfully loaded servable version'; do
    sleep 1
done

docker run --rm --network "$trt_test_net" --env TRTIS_SERVER=trtserver:8001 \
    --workdir /home/mpf/component_build/test --entrypoint ./TrtisDetectionTest "$image_id"

echo Tests passed
