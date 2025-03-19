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
GEMINI_API_KEY = 'your api key here'

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
                GEMINI_API_KEY=GEMINI_API_KEY
            ),
            media_properties={},
            feed_forward_location=None
        )
        component = GeminiComponent()

        def side_effect_function(prompt_dict, data_uri, detection_properties, video_process_timer):
            detection_properties['CLOTHING'] = "The person in the image is wearing a dark suit with a matching tie. The shirt underneath appears to be light-colored, possibly white or off-white. He has glasses on his face and is smiling as he shakes hands with someone who isn't fully visible in the frame. His attire suggests a formal setting, possibly for business or an event that requires professional dress code."
            detection_properties['ACTIVITY'] = "The person in the image appears to be shaking someone's hand. They are wearing a suit and tie, which suggests they may be in a professional or formal setting. The context of the photo is not clear from this angle, but it looks like they could be at an event or gathering where such interactions are common."
            detection_properties['ANNOTATED BY GEMINI'] = True

        result = self.run_patched_job(component, job, side_effect_function)[0]
        
        self.assertTrue("CLOTHING" in result.detection_properties and "ACTIVITY" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["CLOTHING"]) > 0 and len(result.detection_properties["ACTIVITY"]) > 0)
    
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