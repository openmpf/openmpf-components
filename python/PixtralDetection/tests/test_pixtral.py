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

# Add pixtral_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pixtral_component.pixtral_component import PixtralComponent

import unittest
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)

class TestPixtral(unittest.TestCase):
    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('bitcoin.jpeg'),
            job_properties=dict(),
            media_properties={
                'MIME_TYPE': 'image/jpeg'
            },
            feed_forward_location=None
        )
        component = PixtralComponent()

        result = component.get_detections_from_image(job)[0]
        self.assertTrue('ANNOTATED BY PIXTRAL' in result.detection_properties)

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)

if __name__ == '__main__':
    unittest.main(verbosity=2)