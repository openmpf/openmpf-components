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
            data_uri=self._get_test_file('sturgeon.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 3,
                NUMBER_OF_TEMPLATES = 80,
                CLASSIFICATION_LIST = 'imagenet',
                ENABLE_CROPPING='False'
            ),
            media_properties={},
            feed_forward_location=None
        )
        result = list(ClipComponent().get_detections_from_image(job))[0]
        self.assertEqual(job.job_properties["NUMBER_OF_CLASSIFICATIONS"], len(self._output_to_list(result.detection_properties["CLASSIFICATION LIST"])))
        self.assertTrue("sturgeon" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]))
        self.assertEqual("sturgeon", result.detection_properties["CLASSIFICATION"])
        
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
        self.assertTrue("violent" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]))
        self.assertEqual("violent", result.detection_properties["CLASSIFICATION"])

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)
    
    @staticmethod
    def _output_to_list(output):
        return [elt.strip() for elt in output.split('; ')]


if __name__ == '__main__':
    unittest.main(verbosity=2)