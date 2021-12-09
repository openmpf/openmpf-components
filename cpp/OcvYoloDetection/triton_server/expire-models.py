#!/usr/bin/env python3

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

import os
import requests
import sys
import threading

from datetime import datetime
from http import HTTPStatus

HOST = 'localhost'
CHECK_PERIOD_SECONDS = int(os.getenv('EXPIRE_CHECK_PERIOD_SECONDS', 60 * 10))  # 10 minute default
ELAPSED_TIME_SECONDS = int(os.getenv('EXPIRE_ELAPSED_TIME_SECONDS', 60 * 30))  # 30 minute default

NEVER_UNLOAD_MODELS = os.getenv('EXPIRE_NEVER_UNLOAD_MODELS', '').split(',')
NEVER_UNLOAD_MODELS = list(filter(None, NEVER_UNLOAD_MODELS))  # remove empty elements

def print_std_out(text):
    print(text)
    sys.stdout.flush()


def print_std_err(text):
    print(text, file=sys.stderr)
    sys.stderr.flush()


def check_and_expire():
    current_time = datetime.utcnow()
    print_std_out('The time is ' + str(current_time) + '. Checking for expired models.')

    stats_url = 'http://' + HOST + ':8000/v2/models/stats'

    try:
        resp = requests.get(url=stats_url)
    except requests.exceptions.RequestException as e:
        print_std_err('Failed to GET from ' + stats_url + ' due to ' + str(e)
                      + '. Triton may not be running (yet).')
        return

    if resp.status_code != HTTPStatus.OK:
        print_std_err('Failed to GET from ' + stats_url + '. Response code: ' + str(resp.status_code))
        return

    for model_json in resp.json()['model_stats']:

        model_name = model_json['name']
        if model_name in NEVER_UNLOAD_MODELS:
            continue

        last_inference = model_json['last_inference']

        if last_inference != 0:  # don't expire explicitly loaded models
            timestamp_secs = int(last_inference) / 1000  # convert from milliseconds to seconds
            last_inference_time = datetime.fromtimestamp(timestamp_secs)

            time_delta = current_time - last_inference_time
            # print(time_delta)  # DEBUG

            if time_delta.total_seconds() > ELAPSED_TIME_SECONDS:
                print_std_out(model_name + ' was last used at ' + str(last_inference_time) + '. Unloading now.')

                upload_url = 'http://' + HOST + ':8000/v2/repository/models/' + model_name + '/unload'

                try:
                    resp = requests.post(url=upload_url)
                except requests.exceptions.RequestException as e:
                    print_std_err('Failed to POST to ' + upload_url + ' due to ' + str(e))
                    continue

                if resp.status_code != HTTPStatus.OK:
                    print_std_err('Failed to POST to ' + upload_url + '. Response code: ' + str(resp.status_code))
                    continue

                print_std_out('Successfully unloaded ' + model_name)


def check_and_expire_loop():
    check_and_expire()
    threading.Timer(CHECK_PERIOD_SECONDS, check_and_expire_loop).start()


if CHECK_PERIOD_SECONDS <= 0:
    print_std_out('Disabled periodic check for expired models.')
    exit(0)

print_std_out('Will check for expired models every ' + str(CHECK_PERIOD_SECONDS) + ' seconds.'
              + ' Models will be unloaded if not used for ' + str(ELAPSED_TIME_SECONDS) + ' seconds.')

if NEVER_UNLOAD_MODELS:
    print_std_out('The following models will never be unloaded: ' + str(NEVER_UNLOAD_MODELS))

check_and_expire_loop()
