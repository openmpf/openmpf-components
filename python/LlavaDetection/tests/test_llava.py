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

import sys
import os
import logging

# Add clip_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llava_component.llava_component import LlavaComponent

import unittest
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)

class TestLlava(unittest.TestCase):
    def test_image_file(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = list(component.get_detections_from_image(job))[0]
        
        self.assertTrue("CLOTHING" in result.detection_properties and "ACTIVITY" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["CLOTHING"]) > 0 and len(result.detection_properties["ACTIVITY"]) > 0)
    
    def test_custom_config(self):
        ff_loc = mpf.ImageLocation(0, 0, 900, 1600, -1, dict(CLASSIFICATION="DOG"))
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                CONFIG_JSON_PATH=self._get_test_file('custom_config.json')
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = list(component.get_detections_from_image(job))[0]
        
        self.assertTrue("DESCRIPTION" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["DESCRIPTION"]) > 0)

    def test_video_file(self):
        ff_loc = mpf.ImageLocation(0, 0, 900, 1600, -1, dict(CLASSIFICATION="DOG"))
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                CONFIG_JSON_PATH=self._get_test_file('custom_config.json')
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = list(component.get_detections_from_image(job))[0]

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)