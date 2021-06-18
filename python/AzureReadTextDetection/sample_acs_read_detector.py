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
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

from __future__ import division, print_function

import sys

import mpf_component_api as mpf

from acs_read_detection_component import AcsReadDetectionComponent


def main():
    # Debugging option, set to true to view individual lines.
    check_lines = False
    if len(sys.argv) != 4:
        sys.exit('Error: Invalid number of arguments. \nUsage: {} <acs_url> <acs_subscription_key> <img_path>'
                 .format(sys.argv[0]))
    _, acs_url, acs_subscription_key, img_path = sys.argv

    merge_lines = "True"
    if check_lines:
        merge_lines = "False"
    if img_path.lower().endswith('.pdf'):
        job = mpf.GenericJob('Sample Job', img_path,
                             dict(ACS_URL=acs_url, ACS_SUBSCRIPTION_KEY=acs_subscription_key, MERGE_LINES=merge_lines), {},
                             None)
        detections = list(AcsReadDetectionComponent().get_detections_from_generic(job))
    elif img_path.lower().endswith('.avi'):
        job = mpf.VideoJob('Sample Job', img_path, 0, -1,
                             dict(ACS_URL=acs_url, ACS_SUBSCRIPTION_KEY=acs_subscription_key, MERGE_LINES=merge_lines), {},
                             None)
        detections = list(AcsReadDetectionComponent().get_detections_from_video(job))
    else:
        job = mpf.ImageJob('Sample Job', img_path, dict(ACS_URL=acs_url, ACS_SUBSCRIPTION_KEY=acs_subscription_key, MERGE_LINES=merge_lines), {},
                           None)
        detections = list(AcsReadDetectionComponent().get_detections_from_image(job))

    print('Found', len(detections), 'detections.')
    if len(detections) == 0:
        return

    print()
    print('=' * 80)
    print()
    for idx, detection in enumerate(detections):
        print('Detection', idx + 1, 'details:')
        print(detection)
        print()
        print('=' * 80)
        print()


if __name__ == '__main__':
    main()
