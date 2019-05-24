#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2019 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2019 The MITRE Corporation                                      #
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
import os

try:
    # Add east_component to path.
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from east_component.east_component import EastComponent
except ImportError as e:
    print('Failed to import east_component due to:', e)
    EastComponent = None

import unittest
import mpf_component_api as mpf


@unittest.skipUnless(EastComponent, 'EAST dependencies do not appear to be installed.')
class TestEast(unittest.TestCase):

    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('unstructured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                PADDING='0.0'
            ),
            media_properties={},
            feed_forward_location=None
        )
        results = list(EastComponent().get_detections_from_image(job))
        self.assertEqual(3, len(results))

    def test_merging(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-structured-no-padding',
            data_uri=self._get_test_file('structured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_MIN_OVERLAP='0.0001',
                PADDING='0.0'
            ),
            media_properties={},
            feed_forward_location=None
        )
        num_dets_nopad = len(list(comp.get_detections_from_image(job)))

        # With a higher overlap threshold, lots of text will remain unmerged
        job = mpf.ImageJob(
            job_name='test-structured-high-overlap',
            data_uri=self._get_test_file('structured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_MIN_OVERLAP='0.2',
                PADDING='0.0'
            ),
            media_properties={},
            feed_forward_location=None
        )
        num_dets_high_overlap = len(list(comp.get_detections_from_image(job)))

        # With plenty of padding, Most text of the same size should be grouped
        job = mpf.ImageJob(
            job_name='test-structured-high-padding',
            data_uri=self._get_test_file('structured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_MIN_OVERLAP='0.0001',
                PADDING='0.5'
            ),
            media_properties={},
            feed_forward_location=None
        )
        num_dets_pad = len(list(comp.get_detections_from_image(job)))
        self.assertTrue(
            num_dets_pad
            < num_dets_nopad
            < num_dets_high_overlap
        )

    def test_structured_text_detection(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-structured',
            data_uri=self._get_test_file('structured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
            ),
            media_properties={},
            feed_forward_location=None
        )
        res = next(iter(comp.get_detections_from_image(job)))
        self.assertTrue(res.detection_properties['TEXT_TYPE'] == 'STRUCTURED')

        # With a higher overlap threshold, lots of text will remain unmerged
        job = mpf.ImageJob(
            job_name='test-unstructured',
            data_uri=self._get_test_file('unstructured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
            ),
            media_properties={},
            feed_forward_location=None
        )
        res = next(iter(comp.get_detections_from_image(job)))
        self.assertTrue(res.detection_properties['TEXT_TYPE'] == 'UNSTRUCTURED')

    def test_video_file(self):
        job = mpf.VideoJob(
            job_name='test-video',
            data_uri=self._get_test_file('toronto-highway-shortened.mp4'),
            start_frame=0,
            stop_frame=-1,
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_MIN_OVERLAP='0.0001',
                MERGE_REGIONS='TRUE'
            ),
            media_properties={},
            feed_forward_track=None
        )

        results = list(EastComponent().get_detections_from_video(job))
        self.assertTrue(len(results) > 0)

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
