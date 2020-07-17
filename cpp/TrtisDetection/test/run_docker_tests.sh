#!/usr/bin/env bash

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

set -o errexit -o pipefail

# Copy over library and model files.
cd /opt/tensorrtserver/ && ./nvidia_entrypoint.sh
cp -a /Testing_MPF_FILES/*.so* /usr/local/lib
ln -s /usr/local/lib/libgtest.so.0.0.0 libgtest.so
ln -s /usr/local/lib/libgtest.so.0.0.0 libgtest.so.0
ln -s /usr/local/lib/libgtest_main.so.0.0.0 libgtest_main.so
ln -s /usr/local/lib/libgtest_main.so.0.0.0 libgtest_main.so.0
cp /Testing_MPF_FILES/*.so /opt/tensorrtserver/
cp /Testing_MPF_FILES/plugin/TrtisDetection/lib/librequest.so /usr/local/lib/
cp -r /Testing_MPF_FILES/plugin/TrtisDetection/models /
rm /Testing_MPF_FILES/plugin/TrtisDetection/models/ip_irv2_coco/1/model.graphdef
ldconfig

# Start trtserver as a background process.
trtserver --model-store=/models 2> /Testing_MPF_FILES/server_logs.txt &

# Running test code alongside active server process.
echo "Starting server and loading model files."
cd /Testing_MPF_FILES/test/ \

# Wait for server to finish loading coco model files.
 for i in {1..10}
    do
    [[ -f "/Testing_MPF_FILES/server_logs.txt" ]] \
    && grep -q "Successfully loaded servable version" "/Testing_MPF_FILES/server_logs.txt" \
    && echo "Server ready for testing. Beginning detection tests." \
    && break

    sleep 5

    if [ $i == 10 ]
    then
        echo "Time limit reached, model load unsuccessful. Exiting tests with failure."
        exit 1
    fi
    done

# Run TRTIS tests.
./TrtisDetectionTest

