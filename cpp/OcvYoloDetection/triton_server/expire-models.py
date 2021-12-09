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

import requests
import sys
import threading

from datetime import datetime
from http import HTTPStatus

HOST = 'localhost'  # DEBUG: localhost
CHECK_PERIOD_SECONDS = 5  # DEBUG: 60 x 10  # 10 minutes
EXPIRE_TIME_SECONDS = 10  # DEBUG: 60 * 60  # 1 hour


def check_and_expire():
    current_time = datetime.utcnow()
    print('Checking for expired models at ' + str(current_time))

    stats_url = 'http://' + HOST + ':8000/v2/models/stats'

    try:
        resp = requests.get(url=stats_url)
    except requests.exceptions.RequestException as e:
        print('Failed to GET from ' + stats_url + ' due to ' + str(e)
              + '. Triton may not be running (yet).', file=sys.stderr)
        return

    if resp.status_code != HTTPStatus.OK:
        print('Failed to GET from ' + stats_url + '. Response code: ' + str(resp.status_code), file=sys.stderr)
        return

    for model_json in resp.json()['model_stats']:

        model_name = model_json['name']
        last_inference = model_json['last_inference']

        if last_inference != 0:  # don't expire explicitly loaded models
            timestamp_secs = int(last_inference) / 1000  # convert from milliseconds to seconds
            last_inference_time = datetime.fromtimestamp(timestamp_secs)

            time_delta = current_time - last_inference_time
            # print(time_delta)  # DEBUG

            if time_delta.total_seconds() > EXPIRE_TIME_SECONDS:
                print(model_name + ' was last used at ' + str(last_inference_time) + '. Unloading now.')  # DEBUG

                upload_url = 'http://' + HOST + ':8000/v2/repository/models/' + model_name + '/unload'

                try:
                    resp = requests.post(url=upload_url)
                except requests.exceptions.RequestException as e:
                    print('Failed to POST to ' + upload_url + ' due to ' + str(e), file=sys.stderr)
                    continue

                if resp.status_code != HTTPStatus.OK:
                    print('Failed to POST to ' + upload_url + '. Response code: ' + str(resp.status_code),
                          file=sys.stderr)
                    continue

                print('Successfully unloaded ' + model_name)


def check_and_expire_loop():
    check_and_expire()
    threading.Timer(CHECK_PERIOD_SECONDS, check_and_expire_loop).start()


check_and_expire_loop()
