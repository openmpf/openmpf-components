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
from collections import Counter

# Add east_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from east_component.east_component import EastComponent

import unittest
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)

class TestEast(unittest.TestCase):

    def test_image_file(self):
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('unstructured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
            ),
            media_properties={},
            feed_forward_location=None
        )
        results = list(EastComponent().get_detections_from_image(job))
        self.assertEqual(3, len(results))

    def test_video_file(self):
        job = mpf.VideoJob(
            job_name='test-video',
            data_uri=self._get_test_file('toronto-highway-shortened.mp4'),
            start_frame=0,
            stop_frame=-1,
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
            ),
            media_properties={},
            feed_forward_track=None
        )

        results = EastComponent().get_detections_from_video(job)
        present_frames = set([t.start_frame for t in results])
        self.assertEqual(150, len(present_frames))

    # TODO: Move these track comparison methods to openmpf-python-component-sdk test utils.

    @staticmethod
    def _image_locations_equal(loc_a, loc_b):
        return loc_a.x_left_upper == loc_b.x_left_upper \
               and loc_a.y_left_upper == loc_b.y_left_upper \
               and loc_a.width == loc_b.width \
               and loc_a.height == loc_b.height \
               and loc_a.confidence == loc_b.confidence \
               and loc_a.detection_properties == loc_b.detection_properties

    @staticmethod
    def _frame_locations_equal(flocs_a, flocs_b):
        if set(flocs_a.keys()) != set(flocs_b.keys()):
            return False
        for k in flocs_a.keys():
            if not TestEast._image_locations_equal(flocs_a[k], flocs_b[k]):
                return False
        return True

    @staticmethod
    def _video_tracks_equal(track_a, track_b):
        return track_a.start_frame == track_b.start_frame \
            and track_a.stop_frame == track_b.stop_frame \
            and track_a.confidence == track_b.confidence \
            and TestEast._frame_locations_equal(track_a.frame_locations, track_b.frame_locations) \
            and track_a.detection_properties == track_b.detection_properties

    @staticmethod
    def _all_video_tracks_equal(tracks_a, tracks_b):
        if len(tracks_a) != len(tracks_b):
            return False
        for track_a, track_b in zip(tracks_a, tracks_b):
            if not TestEast._video_tracks_equal(track_a, track_b):
                return False
        return True

    def test_video_batch_size(self):
        job = mpf.VideoJob(
            job_name='test-batch-remainder',
            data_uri=self._get_test_file('toronto-highway-shortened.mp4'),
            start_frame=0,
            stop_frame=15,
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                BATCH_SIZE='4'
            ),
            media_properties={},
            feed_forward_track=None
        )

        results_r0 = list(EastComponent().get_detections_from_video(job))

        job = mpf.VideoJob(
            job_name='test-batch-remainder-1',
            data_uri=self._get_test_file('toronto-highway-shortened.mp4'),
            start_frame=0,
            stop_frame=15,
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                BATCH_SIZE='5'
            ),
            media_properties={},
            feed_forward_track=None
        )

        results_r1 = list(EastComponent().get_detections_from_video(job))
        self.assertTrue(self._all_video_tracks_equal(results_r0, results_r1))
        del results_r1

        job = mpf.VideoJob(
            job_name='test-batch-remainder-4',
            data_uri=self._get_test_file('toronto-highway-shortened.mp4'),
            start_frame=0,
            stop_frame=15,
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                BATCH_SIZE='6'
            ),
            media_properties={},
            feed_forward_track=None
        )

        results_r4 = list(EastComponent().get_detections_from_video(job))
        self.assertTrue(self._all_video_tracks_equal(results_r0, results_r4))
        del results_r4

    def test_baseline_thresholds(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-baseline-thresholds',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))

        # There should be 9 detections
        self.assertEqual(9, len(detections))

        # Check that all detections are structured text
        props = [d.detection_properties for d in detections]
        for p in props:
            self.assertEqual('STRUCTURED', p['TEXT_TYPE'])

        # Check that there are 6 horizontal detections and 3 at about -25 degs
        round5 = lambda x: int(5 * round(float(x) / 5.0)) % 360
        rot_counts = Counter([round5(p['ROTATION']) for p in props])
        self.assertEqual(2, len(rot_counts))
        self.assertEqual(6, rot_counts[0])
        self.assertEqual(3, rot_counts[335])

    def test_nms(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-nms',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MERGE_REGIONS='FALSE',
                MAX_SIDE_LENGTH='1280',
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))

        # Check that NMS produces many more (>5x) detections than merging
        self.assertGreater(len(detections), 45)

        # Check that most detections are small (>80% smaller than mean)
        areas = [d.width * d.height for d in detections]
        mean = sum(areas) / float(len(areas))
        smaller_than_mean = sum(a < mean for a in areas) / float(len(areas))
        self.assertGreater(smaller_than_mean, 0.8)

    def test_padding(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-low-x-padding',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                TEMPORARY_PADDING_X='0.0',
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))
        low_padding_x = len(detections)

        # Check that no x padding results in less merging
        self.assertGreater(low_padding_x, 9)

        job = mpf.ImageJob(
            job_name='test-low-y-padding',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                TEMPORARY_PADDING_Y='0.0',
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))
        low_padding_y = len(detections)

        # Check that no y padding results in even less merging
        self.assertGreater(low_padding_y, low_padding_x)

        job = mpf.ImageJob(
            job_name='test-low-padding',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                TEMPORARY_PADDING_X='0.0',
                TEMPORARY_PADDING_Y='0.0',
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))
        low_padding_area = sum(d.width * d.height for d in detections)
        low_padding = len(detections)

        # Check that no padding results in the least merging
        self.assertGreater(low_padding, low_padding_y)

        job = mpf.ImageJob(
            job_name='test-final-padding',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                TEMPORARY_PADDING_X='0.0',
                TEMPORARY_PADDING_Y='0.0',
                FINAL_PADDING='0.1'
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))
        high_padding_area = sum(d.width * d.height for d in detections)
        high_padding = len(detections)

        # Check that higher final padding results in larger total detection area
        self.assertGreater(high_padding_area, low_padding_area)

        # Check that final padding doesn't affect total number of detections
        self.assertEqual(high_padding, low_padding)

    def test_max_side_length(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-low-max-side-length',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='320',
            ),
            media_properties={},
            feed_forward_location=None
        )
        small_image = len(list(comp.get_detections_from_image(job)))

        # Check that low side length results in only large text detected
        self.assertEqual(4, small_image)

    def test_overlap_threshold(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-high-overlap-threshold',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_OVERLAP_THRESHOLD='0.1',
            ),
            media_properties={},
            feed_forward_location=None
        )
        high_threshold = len(list(comp.get_detections_from_image(job)))

        # A higher threshold should result in less merging (more detections)
        self.assertLess(9, high_threshold)

        job = mpf.ImageJob(
            job_name='test-negative-overlap-threshold',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_OVERLAP_THRESHOLD='-1.0',
            ),
            media_properties={},
            feed_forward_location=None
        )
        neg_threshold = len(list(comp.get_detections_from_image(job)))

        # The negative threshold should merge the two axis-aligned pieces of
        #  small text, and the two rotated pieces of small text.
        self.assertEqual(7, neg_threshold)

    def test_rotation_threshold(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-high-rotation-threshold',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_MAX_ROTATION_DIFFERENCE='90',
            ),
            media_properties={},
            feed_forward_location=None
        )
        high_threshold = len(list(comp.get_detections_from_image(job)))

        # The high threshold should should merge the three blocks of small text
        # in the middle of the image
        self.assertEqual(7, high_threshold)

    def test_text_height_threshold(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-high-text-height-threshold',
            data_uri=self._get_test_file('thresholds.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_MAX_TEXT_HEIGHT_DIFFERENCE='10',
                MERGE_OVERLAP_THRESHOLD='0.01'
            ),
            media_properties={},
            feed_forward_location=None
        )
        high_threshold = len(list(comp.get_detections_from_image(job)))

        # The high threshold should merge the four pieces of text in the top
        # left of the image, as well as the two rotated piees below that
        self.assertEqual(5, high_threshold)

    def test_text_type_threshold(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-standard-text-type-threshold-structured',
            data_uri=self._get_test_file('structured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MIN_STRUCTURED_TEXT_THRESHOLD='0.01',
            ),
            media_properties={},
            feed_forward_location=None
        )
        ttype = next(iter(
            comp.get_detections_from_image(job))
        ).detection_properties['TEXT_TYPE']
        self.assertEqual('STRUCTURED', ttype)

        job = mpf.ImageJob(
            job_name='test-standard-text-type-threshold-unstructured',
            data_uri=self._get_test_file('unstructured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MIN_STRUCTURED_TEXT_THRESHOLD='0.01',
            ),
            media_properties={},
            feed_forward_location=None
        )
        ttype = next(iter(
            comp.get_detections_from_image(job))
        ).detection_properties['TEXT_TYPE']
        self.assertEqual('UNSTRUCTURED', ttype)

        job = mpf.ImageJob(
            job_name='test-high-text-type-threshold-structured',
            data_uri=self._get_test_file('structured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MIN_STRUCTURED_TEXT_THRESHOLD='0.5',
            ),
            media_properties={},
            feed_forward_location=None
        )
        ttype = next(iter(
            comp.get_detections_from_image(job))
        ).detection_properties['TEXT_TYPE']

        # With a high structured text threshold, most images will be classified
        # as unstructured text
        self.assertEqual('UNSTRUCTURED', ttype)

        job = mpf.ImageJob(
            job_name='test-low-text-type-threshold-unstructured',
            data_uri=self._get_test_file('unstructured.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MIN_STRUCTURED_TEXT_THRESHOLD='0.0001',
            ),
            media_properties={},
            feed_forward_location=None
        )
        ttype = next(iter(
            comp.get_detections_from_image(job))
        ).detection_properties['TEXT_TYPE']

        # With a low structured text threshold, most images will be classified
        # as structured text
        self.assertEqual('STRUCTURED', ttype)

    def test_vertical_suppression(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-vsupp-off',
            data_uri=self._get_test_file('rotation.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_REGIONS='FALSE',
                ROTATE_AND_DETECT='TRUE',
                SUPPRESS_VERTICAL='FALSE'
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))

        # Confirm that vertical detections were not suppressed
        at_least_one_vertical = False
        for d in detections:
            if d.height > d.width:
                at_least_one_vertical = True
                break
        self.assertTrue(at_least_one_vertical)

        job = mpf.ImageJob(
            job_name='test-vsupp-on',
            data_uri=self._get_test_file('rotation.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                MERGE_REGIONS='FALSE',
                ROTATE_AND_DETECT='TRUE'
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = list(comp.get_detections_from_image(job))

        # Confirm that vertical detections were suppressed
        for d in detections:
            self.assertGreater(d.width, d.height)

        job = mpf.ImageJob(
            job_name='test-vsupp-off',
            data_uri=self._get_test_file('rotation.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                ROTATE_AND_DETECT='TRUE',
                SUPPRESS_VERTICAL='FALSE'
            ),
            media_properties={},
            feed_forward_location=None
        )
        vsupp_off = len(list(comp.get_detections_from_image(job)))

        job = mpf.ImageJob(
            job_name='test-vsupp-on',
            data_uri=self._get_test_file('rotation.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                ROTATE_AND_DETECT='TRUE'
            ),
            media_properties={},
            feed_forward_location=None
        )
        vsupp_on = len(list(comp.get_detections_from_image(job)))

        # Without vertical suppression, there are many low-quality detections
        self.assertGreater(vsupp_off, vsupp_on)

    def test_rotate_and_detect(self):
        comp = EastComponent()

        job = mpf.ImageJob(
            job_name='test-rotate-off',
            data_uri=self._get_test_file('rotation.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280'
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = comp.get_detections_from_image(job)

        # Filter out small single-word detections that failed to be merged
        detections = [d for d in detections if d.height > 200 or d.width > 200]

        # There should be only three merged detections, all horizontal
        self.assertEqual(3, len(detections))
        round5 = lambda x: int(5 * round(float(x) / 5.0)) % 180
        for d in detections:
            self.assertEqual(0, round5(d.detection_properties['ROTATION']))

        job = mpf.ImageJob(
            job_name='test-rotate-on',
            data_uri=self._get_test_file('rotation.jpg'),
            job_properties=dict(
                MAX_SIDE_LENGTH='1280',
                ROTATE_AND_DETECT='TRUE'
            ),
            media_properties={},
            feed_forward_location=None
        )
        detections = comp.get_detections_from_image(job)

        # Filter out small single-word detections that failed to be merged
        detections = [d for d in detections if d.height > 200 or d.width > 200]

        # Check that there are 3 horizontal and 3 vertical detections
        rots = Counter([round5(d.detection_properties['ROTATION']) for d in detections])
        self.assertEqual(2, len(rots))
        self.assertEqual(3, rots[0])
        self.assertEqual(3, rots[90])

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
