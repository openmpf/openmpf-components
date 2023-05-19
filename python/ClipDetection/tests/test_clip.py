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

expected_feature = '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'
logging.basicConfig(level=logging.DEBUG)

class TestClip(unittest.TestCase):

    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 3,
                NUMBER_OF_TEMPLATES = 1,
                CLASSIFICATION_LIST = 'coco',
                ENABLE_CROPPING='False',
                INCLUDE_FEATURES = 'True'
            ),
            media_properties={},
            feed_forward_location=None
        )
        component = ClipComponent()
        result = list(component.get_detections_from_image(job))[0]
        
        self.assertEqual(job.job_properties["NUMBER_OF_CLASSIFICATIONS"], len(self._output_to_list(result.detection_properties["CLASSIFICATION LIST"])))
        self.assertTrue("dog" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]))
        self.assertEqual("dog", result.detection_properties["CLASSIFICATION"])
        self.assertTrue(result.detection_properties["FEATURE"] is not None)
    
    def test_image_file_custom(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('riot.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 4,
                CLASSIFICATION_PATH = self._get_test_file("violence_classes.csv"),
                TEMPLATE_PATH = self._get_test_file("violence_templates.txt")
            ),
            media_properties={},
            feed_forward_location=None
        )
        result = list(ClipComponent().get_detections_from_image(job))[0]
        self.assertEqual(job.job_properties["NUMBER_OF_CLASSIFICATIONS"], len(self._output_to_list(result.detection_properties["CLASSIFICATION LIST"])))
        self.assertTrue("violent scene" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]))
        self.assertEqual("violent scene", result.detection_properties["CLASSIFICATION"])

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)
    
    @staticmethod
    def _output_to_list(output):
        return [elt.strip() for elt in output.split('; ')]


if __name__ == '__main__':
    unittest.main(verbosity=2)