#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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
from clip_component.clip_component import ClipComponent

import unittest
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)

class TestClip(unittest.TestCase):

    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('collie.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 10,
                NUMBER_OF_TEMPLATES = 80,
                CLASSIFICATION_LIST = 'imagenet',
                ENABLE_CROPPING='False', 
                ENABLE_TRITON='True',
                TRITON_SERVER='clip-detection-server:8001'
            ),
            media_properties={},
            feed_forward_location=None
        )      
        result = list(ClipComponent().get_detections_from_image(job))[0]
        self.assertTrue("collie" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]) or "border collie" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]))
        

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)
    
    @staticmethod
    def _output_to_list(output):
        return [elt.strip() for elt in output.split('; ')]


if __name__ == '__main__':
    unittest.main(verbosity=2)