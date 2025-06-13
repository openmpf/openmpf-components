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
import numpy as np

# Add gemini_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gemini_component.gemini_component import GeminiComponent

import unittest
import unittest.mock
from unittest.mock import MagicMock, Mock
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)
USE_MOCKS = True

# Replace with your own API key
GEMINI_API_KEY = ''

# Replace with your own desired model name
MODEL_NAME = "gemma-3-27-it"

class TestGemini(unittest.TestCase):
    def run_patched_job(self, component, job, side_effect_function):
        if isinstance(job, mpf.ImageJob):
            detection_func = component.get_detections_from_image
        elif isinstance(job, mpf.VideoJob):
            detection_func = component.get_detections_from_video
        else:
            raise Exception("Must be image or video job.")
    
        if not USE_MOCKS:
            return detection_func(job)

        with unittest.mock.patch("gemini_component.gemini_component.GeminiComponent._get_gemini_response", side_effect=side_effect_function):
            results = list(detection_func(job))
            return results

    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME
            ),
            media_properties={},
            feed_forward_location=None
        )
        component = GeminiComponent()

        expected_response = "The scene appears to be a banquet hall or conference room in a hotel or convention center."

        def side_effect_function(*args, **kwargs):
            return expected_response

        result = self.run_patched_job(component, job, side_effect_function)[0]
        
        self.assertTrue("LOCATION" in result.detection_properties)
        self.assertEqual(result.detection_properties["LOCATION"], expected_response)
    
    def test_get_frames(self):
        component = GeminiComponent()
        self.assertEqual(component._get_frames_to_process([], 1), [])
        self.assertEqual(component._get_frames_to_process([1], 2), [1])
        self.assertEqual(component._get_frames_to_process([503], 2), [503])
        self.assertEqual(component._get_frames_to_process([503, 1_000], 5_000), [503])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5], 1), [0,1,2,3,4,5])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5], 2), [0,2,4, 5])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5], 3), [0,3,5])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5,900], 3), [0,3,5,900])
        self.assertEqual(component._get_frames_to_process([4,900,902,905,906,907,908,909,910,911,912,913], 5), [4,900,905,910,913])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918], 6), [910,916])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919], 6), [910,916,919])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919,920], 6), [910,916,920])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919,920,921], 6), [910,916,921])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919,920,921,922], 6), [910,916,922])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,5_000,5_001,10_000], 6), [910,916,5_000,10_000])

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)

if __name__ == '__main__':
    unittest.main(verbosity=2)