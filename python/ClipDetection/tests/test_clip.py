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
    def test_image_coop(self):
        job = mpf.ImageJob(
            job_name='test-image-coop',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                CLASSIFICATION_LIST='imagenet',
                NUMBER_OF_CLASSIFICATIONS = 3,
                TEMPLATE_TYPE = 'openai_1',
                ENABLE_CROPPING ='False',
                INCLUDE_FEATURES = 'True',
                MODEL_NAME="CoOp"
            ),
            media_properties={},
            feed_forward_location=None 
        )
        component = ClipComponent()
        result = list(component.get_detections_from_image(job))[0]
        # print(result.detection_properties["FEATURE"])
        self.assertTrue("toy terrier", result.detection_properties["CLASSIFICATION"])

    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 3,
                TEMPLATE_TYPE = 'openai_1',
                ENABLE_CROPPING ='False',
                INCLUDE_FEATURES = 'True',
                MODEL_NAME="ViT-B/32"
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
            job_name='test-image-custom',
            data_uri=self._get_test_file('riot.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 4,
                CLASSIFICATION_PATH = self._get_test_file("violence_classes.csv"),
                TEMPLATE_PATH = self._get_test_file("violence_templates.txt")
            ),
            media_properties={},
            feed_forward_location=None
        )
        component = ClipComponent()
        result = list(component.get_detections_from_image(job))[0]

        self.assertEqual(job.job_properties["NUMBER_OF_CLASSIFICATIONS"], len(self._output_to_list(result.detection_properties["CLASSIFICATION LIST"])))
        self.assertTrue("violent scene" in self._output_to_list(result.detection_properties["CLASSIFICATION LIST"]))
        self.assertEqual("violent scene", result.detection_properties["CLASSIFICATION"])
        
    def test_image_file_rollup(self):
        job = mpf.ImageJob(
            job_name='test-image-rollup',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                NUMBER_OF_CLASSIFICATIONS = 4,
                TEMPLATE_TYPE = 'openai_1',
                CLASSIFICATION_PATH = self._get_test_file("rollup.csv"),
                ENABLE_CROPPING = 'False'
            ),
            media_properties={},
            feed_forward_location=None
        )
        result = list(ClipComponent().get_detections_from_image(job))[0]
        self.assertEqual("indoor animal", result.detection_properties["CLASSIFICATION"])

    def test_video_file(self):
        job = mpf.VideoJob(
            job_name='test-video',
            data_uri=self._get_test_file('test_video.mp4'),
            start_frame=0,
            stop_frame=14,
            job_properties=dict(
                TEMPLATE_TYPE = 'openai_1',
                ENABLE_CROPPING = 'False',
                DETECTION_FRAME_BATCH_SIZE = 4
            ),
            media_properties={},
            feed_forward_track=None
        )
        component = ClipComponent()
        results = list(component.get_detections_from_video(job))

        self.assertEqual(results[0].detection_properties['CLASSIFICATION'], "dog")
        self.assertEqual(results[0].start_frame, 0)
        self.assertEqual(results[0].stop_frame, 4)

        self.assertEqual(results[1].detection_properties['CLASSIFICATION'], "orange")
        self.assertEqual(results[1].start_frame, 5)
        self.assertEqual(results[1].stop_frame, 9)

        self.assertEqual(results[2].detection_properties['CLASSIFICATION'], "dog")
        self.assertEqual(results[2].start_frame, 10)
        self.assertEqual(results[2].stop_frame, 14)

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)
    
    @staticmethod
    def _output_to_list(output):
        return [elt.strip() for elt in output.split('; ')]


if __name__ == '__main__':
    unittest.main(verbosity=2)