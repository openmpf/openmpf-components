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

import http.server
import logging
import os
import queue
import shutil
import sys
import threading
from typing import ClassVar
import unittest

import cv2
import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../acs_ocr_component'))
import acs_ocr_component
from acs_ocr_component import AcsOcrComponent, FrameEncoder


human_rights_image_text = '''Human Rights. Bulgarian and English.
UNIVERSAL DECLARATION OF HUMAN RIGHTS
Article 1
All human beings are born free and equal in dignity and
rights. They are endowed with reason and conscience
and should act towards one another in a spirit of
brotherhood.'''

logging.basicConfig(level=logging.DEBUG)

class TestAcs(unittest.TestCase):

    mock_server: ClassVar['MockServer']

    @classmethod
    def setUpClass(cls):
        cls.mock_server = MockServer()

    @classmethod
    def tearDownClass(cls):
        cls.mock_server.shutdown()

    @classmethod
    def set_results_path(cls, results_path):
        cls.mock_server.set_results_path(results_path)

    def tearDown(self):
        self.mock_server.drain_queue()


    def run_image_test(self, image_file_name, results_file_name, expected_detection_rect, expected_rotation):
        self.set_results_path(get_test_file(results_file_name))

        img_path = get_test_file(image_file_name)
        job = mpf.ImageJob('Test', img_path, get_test_properties(), {}, None)
        detections = list(AcsOcrComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))
        detection = detections[0]
        self.assertEqual(human_rights_image_text, detection.detection_properties['TEXT'])
        self.assertEqual(expected_detection_rect,
                         mpf_util.Rect.from_image_location(detection))
        self.assertAlmostEqual(expected_rotation, float(detection.detection_properties['ROTATION']))
        self.assertEqual('en', detection.detection_properties['TEXT_LANGUAGE'])



    def test_image_file(self):
        self.run_image_test('orientation/img/eng.png', 'orientation/img/eng-results.json',
                            mpf_util.Rect(113, 57, 990, 429), 0)


    def test_image_15deg_rotation(self):
        self.run_image_test('orientation/img/eng-15deg.png', 'orientation/img/eng-15deg-results.json',
                            mpf_util.Rect(70, 192, 991, 432), 14.8)


    def test_image_90deg_rotation(self):
        self.run_image_test('orientation/img/eng-90deg.png', 'orientation/img/eng-90deg-results.json',
                            mpf_util.Rect(57, 1080, 990, 429), 90)

    def test_image_100deg_rotation(self):
        self.run_image_test('orientation/img/eng-100deg.png', 'orientation/img/eng-100deg-results.json',
                            mpf_util.Rect(143, 1118, 991, 432), 99.9)

    def test_image_180deg_rotation(self):
        self.run_image_test('orientation/img/eng-180deg.png', 'orientation/img/eng-180deg-results.json',
                            mpf_util.Rect(1080, 585, 990, 429), 180)


    def test_image_200deg_rotation(self):
        self.run_image_test('orientation/img/eng-200deg.png', 'orientation/img/eng-200deg-results.json',
                            mpf_util.Rect(1138, 420, 992, 433), 199.7)

    def test_image_270deg_rotation(self):
        self.run_image_test('orientation/img/eng-270deg.png', 'orientation/img/eng-270deg-results.json',
                            mpf_util.Rect(585, 113, 990, 429), 270)

    def test_image_290deg_rotation(self):
        self.run_image_test('orientation/img/eng-290deg.png', 'orientation/img/eng-290deg-results.json',
                            mpf_util.Rect(409, 133, 990, 430), 290.1)


    def test_video(self):
        test_video_frame_count = 8
        for i in range(test_video_frame_count):
            self.set_results_path(get_test_file('orientation/video/results-frame{}.json'.format(i)))

        job = mpf.VideoJob('Test', get_test_file('orientation/video/test-vid.avi'), 0, -1, get_test_properties(), {},
                           None)
        tracks = list(AcsOcrComponent().get_detections_from_video(job))

        self.assertEqual(8, len(tracks))

        detections_indexed_by_frame = {}

        for track in tracks:
            self.assertEqual(1, len(track.frame_locations))
            detection = track.frame_locations[track.start_frame]
            self.assertFalse(track.start_frame in detections_indexed_by_frame)

            self.assertEqual(human_rights_image_text, track.detection_properties['TEXT'])
            self.assertEqual(human_rights_image_text, detection.detection_properties['TEXT'])

            self.assertEqual('en', detection.detection_properties['TEXT_LANGUAGE'])
            self.assertEqual('en', track.detection_properties['TEXT_LANGUAGE'])

            detections_indexed_by_frame[track.start_frame] = detection


        self.assertEqual(mpf_util.Rect(113, 57, 990, 429),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[0]))
        self.assertAlmostEqual(0, float(detections_indexed_by_frame[0].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(69, 189, 993, 435),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[1]))
        self.assertAlmostEqual(14.6, float(detections_indexed_by_frame[1].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(57, 1080, 990, 429),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[2]))
        self.assertAlmostEqual(90, float(detections_indexed_by_frame[2].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(140, 1119, 992, 434),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[3]))
        self.assertAlmostEqual(99.7, float(detections_indexed_by_frame[3].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(1080, 585, 990, 429),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[4]))
        self.assertAlmostEqual(180, float(detections_indexed_by_frame[4].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(1136, 417, 990, 431),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[5]))
        self.assertAlmostEqual(200, float(detections_indexed_by_frame[5].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(585, 113, 990, 429),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[6]))
        self.assertAlmostEqual(270, float(detections_indexed_by_frame[6].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(409, 132, 990, 431),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[7]))
        self.assertAlmostEqual(290.1, float(detections_indexed_by_frame[7].detection_properties['ROTATION']))



    def assert_new_size(self, initial_width, initial_height, expected_width, expected_height):
        frame = np.zeros((initial_height, initial_width, 3), dtype=np.uint8)
        encoded_frame, new_size = FrameEncoder().resize_and_encode(frame)

        self.assertEqual(expected_width, new_size.width)
        self.assertEqual(expected_height, new_size.height)

        self.assertEqual(initial_width // initial_height, new_size.width // new_size.height)
        self.assertTrue((new_size.width * new_size.height) <= FrameEncoder.MAX_PIXELS)

        self.assertTrue(new_size.width >= FrameEncoder.MIN_DIMENSION_LENGTH)
        self.assertTrue(new_size.height >= FrameEncoder.MIN_DIMENSION_LENGTH)

        self.assertTrue(new_size.width <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(new_size.height <= FrameEncoder.MAX_DIMENSION_LENGTH)

        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)


    def test_resize_frame_due_to_pixel_count(self):
        self.assert_new_size(1234, 421, 1234, 421)
        self.assert_new_size(5000, 1000, 4200, 840)
        self.assert_new_size(1000, 5000, 840, 4200)

        self.assert_new_size(4000, 5000, 2828, 3535)

        self.assert_new_size(4229, 75082, 236, 4200)
        self.assert_new_size(750, 13324, 236, 4200)

        self.assert_new_size(1000, 49, 1020, 50)


    def test_invalid_frame_size(self):
        frame = np.zeros((FrameEncoder.MIN_DIMENSION_LENGTH - 1, FrameEncoder.MAX_DIMENSION_LENGTH + 1, 3), np.uint8)
        with self.assertRaises(mpf.DetectionException) as cm:
            FrameEncoder().resize_and_encode(frame)
        self.assertEqual(mpf.DetectionError.BAD_FRAME_SIZE, cm.exception.error_code)


    def test_invalid_frame_size_too_big_after_resize(self):
        frame = np.zeros((10, 4200, 3), np.uint8)
        with self.assertRaises(mpf.DetectionException) as cm:
            FrameEncoder().resize_and_encode(frame)
        self.assertEqual(mpf.DetectionError.BAD_FRAME_SIZE, cm.exception.error_code)


    def test_invalid_frame_size_too_small_after_resize(self):
        frame = np.zeros((50, 4400, 3), np.uint8)
        with self.assertRaises(mpf.DetectionException) as cm:
            FrameEncoder().resize_and_encode(frame)
        self.assertEqual(mpf.DetectionError.BAD_FRAME_SIZE, cm.exception.error_code)


    def test_resize_due_to_compression(self):
        img = cv2.imread(get_test_file('downsampling/noise.png'))

        original_size = mpf_util.Size.from_frame(img)
        encoded_frame, new_size = FrameEncoder().resize_and_encode(img)
        self.assertNotEqual(original_size, new_size)

        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)
        self.assertEqual(original_size.width // original_size.height, new_size.width // new_size.height)
        self.assertTrue((new_size.width * new_size.height) <= FrameEncoder.MAX_PIXELS)
        self.assertTrue(new_size.width <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(new_size.height <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)



    def test_mapping_coordinates_after_downsampling(self):
        self.run_image_test('downsampling/eng-with-noise.png', 'downsampling/eng-with-noise-results.json',
                            mpf_util.Rect(112, 57, 990, 430), 0)

    def test_rotation_with_downsampling(self):
        self.run_image_test('downsampling/eng-with-noise-30deg.png',
                            'downsampling/eng-with-noise-30deg-results.json',
                            mpf_util.Rect(42, 340, 990, 428), 30)

    def test_left_orientation_with_downsampling(self):
        self.run_image_test('downsampling/eng-with-noise-90deg.png',
                            'downsampling/eng-with-noise-90deg-results.json',
                            mpf_util.Rect(55, 1686, 989, 430), 90)

    def test_left_orientation_with_angle_and_downsampling(self):
        self.run_image_test('downsampling/eng-with-noise-100deg.png',
                            'downsampling/eng-with-noise-100deg-results.json',
                            mpf_util.Rect(316, 1629, 990, 430), 100)


    def test_upsampling(self):
        self.set_results_path(get_test_file('upsampling/tiny-image-results.json'))
        job = mpf.ImageJob('Test', get_test_file('upsampling/tiny-image.png'), get_test_properties(), {}, None)
        detections = list(AcsOcrComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))
        detection = detections[0]
        self.assertEqual('It was the best of times, it was the worst of times', detection.detection_properties['TEXT'])
        self.assertEqual('en', detection.detection_properties['TEXT_LANGUAGE'])
        self.assertEqual(mpf_util.Rect(29, 7, 586, 24), mpf_util.Rect.from_image_location(detection))
        self.assertAlmostEqual(0, float(detection.detection_properties['ROTATION']))


    def test_multiple_regions(self):
        self.set_results_path(get_test_file('multiple-regions/region-test-results.json'))
        job = mpf.ImageJob('Test', get_test_file('multiple-regions/region-test.png'), get_test_properties(), {}, None)
        detections = list(AcsOcrComponent().get_detections_from_image(job))

        self.assertEqual(2, len(detections))

        detection1, detection2 = detections
        expected_text1 = '''\
It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness,'''

        self.assertEqual(expected_text1, detection1.detection_properties['TEXT'])
        self.assertEqual(mpf_util.Rect(50, 55, 219, 85), mpf_util.Rect.from_image_location(detection1))
        self.assertEqual('en', detection1.detection_properties['TEXT_LANGUAGE'])

        expected_text2 = '''\
it was the epoch of belief,
it was the epoch of incredulity,
it was the season
it was the season
of Darkness'''

        self.assertEqual(expected_text2, detection2.detection_properties['TEXT'])
        self.assertEqual(mpf_util.Rect(827, 204, 455, 323), mpf_util.Rect.from_image_location(detection2))
        self.assertEqual('en', detection2.detection_properties['TEXT_LANGUAGE'])



    def test_multiple_regions_with_merging(self):
        self.set_results_path(get_test_file('multiple-regions/region-test-results.json'))
        job = mpf.ImageJob('Test', get_test_file('multiple-regions/region-test.png'),
                           get_test_properties(MERGE_REGIONS='true'), {}, None)

        detections = list(AcsOcrComponent().get_detections_from_image(job))
        self.assertEqual(1, len(detections))
        detection = detections[0]

        expected_text = '''\
It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness,

it was the epoch of belief,
it was the epoch of incredulity,
it was the season
it was the season
of Darkness'''

        self.assertEqual(expected_text, detection.detection_properties['TEXT'])
        self.assertEqual(mpf_util.Rect(50, 55, 1232, 472), mpf_util.Rect.from_image_location(detection))
        self.assertEqual('en', detection.detection_properties['TEXT_LANGUAGE'])



    def test_multiple_regions_with_rotation(self):
        self.set_results_path(get_test_file('multiple-regions/region-test-10deg-results.json'))
        job = mpf.ImageJob('Test', get_test_file('multiple-regions/region-test-10deg.png'), get_test_properties(), {},
                           None)

        detections = list(AcsOcrComponent().get_detections_from_image(job))
        self.assertEqual(2, len(detections))
        detection1, detection2 = detections

        expected_text1 = '''\
It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness,'''

        self.assertEqual(expected_text1, detection1.detection_properties['TEXT'])
        self.assertEqual(mpf_util.Rect(57, 164, 220, 86), mpf_util.Rect.from_image_location(detection1))
        self.assertAlmostEqual(9.5, float(detection1.detection_properties['ROTATION']))
        self.assertEqual('en', detection1.detection_properties['TEXT_LANGUAGE'])

        expected_text2 = '''\
it was the epoch ofbelief,
it was the epoch of incredulity,
It was the season
was the season
of Darkness'''

        self.assertEqual(expected_text2, detection2.detection_properties['TEXT'])
        self.assertEqual(mpf_util.Rect(848, 174, 458, 323), mpf_util.Rect.from_image_location(detection2))
        self.assertAlmostEqual(9.5, float(detection2.detection_properties['ROTATION']))
        self.assertEqual('en', detection2.detection_properties['TEXT_LANGUAGE'])


    def test_multiple_regions_with_rotation_and_merging(self):
        self.set_results_path(get_test_file('multiple-regions/region-test-10deg-results.json'))
        job = mpf.ImageJob('Test', get_test_file('multiple-regions/region-test-10deg.png'),
                           get_test_properties(MERGE_REGIONS='true'), {}, None)

        detections = list(AcsOcrComponent().get_detections_from_image(job))
        self.assertEqual(1, len(detections))
        detection = detections[0]

        expected_text = '''\
It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness,

it was the epoch ofbelief,
it was the epoch of incredulity,
It was the season
was the season
of Darkness'''

        self.assertEqual(expected_text, detection.detection_properties['TEXT'])
        self.assertAlmostEqual(9.5, float(detection.detection_properties['ROTATION']))
        self.assertEqual(mpf_util.Rect(57, 164, 1236, 464), mpf_util.Rect.from_image_location(detection))
        self.assertEqual('en', detection.detection_properties['TEXT_LANGUAGE'])


    def test_chinese_text(self):
        self.set_results_path(get_test_file('foreign/chinese-two-cities-results.json'))
        job = mpf.ImageJob('Test', get_test_file('foreign/chinese-two-cities.png'), get_test_properties(), {}, None)

        detections = list(AcsOcrComponent().get_detections_from_image(job))
        self.assertEqual(1, len(detections))
        detection = detections[0]

        with open(get_test_file('foreign/chinese-two-cities-expected-text.txt')) as f:
            expected_text = f.read().strip()

        self.assertEqual(expected_text, detection.detection_properties['TEXT'])
        self.assertEqual('zh-Hans', detection.detection_properties['TEXT_LANGUAGE'])





    def test_no_results_img(self):
        self.set_results_path(get_test_file('no-results/black-results.json'))
        job = mpf.ImageJob('Test', get_test_file('no-results/black.png'), get_test_properties(), {}, None)
        detections = list(AcsOcrComponent().get_detections_from_image(job))
        self.assertEqual(0, len(detections))


    def test_no_results_video(self):
        for i in range(3):
            self.set_results_path(get_test_file('no-results/black-results.json'))

        job = mpf.VideoJob('Test', get_test_file('no-results/black.avi'), 0, 2, get_test_properties(), {}, None)
        tracks = list(AcsOcrComponent().get_detections_from_video(job))
        self.assertEqual(0, len(tracks))



    def test_get_acs_url_default_options(self):
        acs_url = 'http://localhost:10669/vision/v1.0/ocr'
        self.assertEqual('http://localhost:10669/vision/v1.0/ocr?detectOrientation=true',
                         acs_ocr_component.JobRunner.get_acs_url(dict(ACS_URL=acs_url)))


    def test_get_acs_url_preserves_existing_query_params(self):
        acs_url = 'http://localhost:10669/vision/v1.0/ocr?hello=world'
        properties = dict(ACS_URL=acs_url)
        url = acs_ocr_component.JobRunner.get_acs_url(properties)

        if '?d' in url:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?detectOrientation=true&hello=world', url)
        else:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?hello=world&detectOrientation=true', url)


    def test_get_acs_url_specific_language(self):
        acs_url = 'http://localhost:10669/vision/v1.0/ocr'
        properties = dict(ACS_URL=acs_url, LANGUAGE='en')

        url = acs_ocr_component.JobRunner.get_acs_url(properties)
        if '?d' in url:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?detectOrientation=true&language=en', url)
        else:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?language=en&detectOrientation=true', url)


    def test_get_acs_url_with_explicit_detect_orientation(self):
        acs_url = 'http://localhost:10669/vision/v1.0/ocr'
        properties = dict(ACS_URL=acs_url, LANGUAGE='en', DETECT_ORIENTATION='true')

        url = acs_ocr_component.JobRunner.get_acs_url(properties)
        if '?d' in url:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?detectOrientation=true&language=en', url)
        else:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?language=en&detectOrientation=true', url)


        properties = dict(ACS_URL=acs_url, LANGUAGE='en', DETECT_ORIENTATION='false')

        url = acs_ocr_component.JobRunner.get_acs_url(properties)
        if '?d' in url:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?detectOrientation=false&language=en', url)
        else:
            self.assertEqual('http://localhost:10669/vision/v1.0/ocr?language=en&detectOrientation=false', url)


def get_test_properties(**extra_properties):
    return {
        'ACS_URL': os.getenv('ACS_URL', 'http://localhost:10669/vision/v1.0/ocr'),
        'ACS_SUBSCRIPTION_KEY': os.getenv('ACS_SUBSCRIPTION_KEY', 'test_key'),
        **extra_properties
    }


def get_test_file(filename):
    return os.path.join(os.path.dirname(__file__), 'data', filename)



class MockServer(http.server.HTTPServer):
    def __init__(self):
        super(MockServer, self).__init__(('', 10669), MockRequestHandler)
        self._results_path_queue = queue.Queue()
        threading.Thread(target=self.serve_forever).start()

    def set_results_path(self, results_path):
        self._results_path_queue.put_nowait(results_path)

    def get_results_path(self):
        return self._results_path_queue.get()

    def drain_queue(self):
        try:
            while not self._results_path_queue.empty():
                self._results_path_queue.get_nowait()
        except queue.Empty:
            pass


class MockRequestHandler(http.server.BaseHTTPRequestHandler):
    server: MockServer

    def do_POST(self):
        results_path = self.server.get_results_path()

        if not self.headers['Ocp-Apim-Subscription-Key']:
            self.send_error(403, 'Expected the Ocp-Apim-Subscription-Key header to contain the subscription key.')
            return

        if self.headers['Content-Type'] != 'application/octet-stream':
            self.send_error(415, 'Expected content type to be application/octet-stream')
            return

        content_len = int(self.headers['Content-Length'])
        post_body = self.rfile.read(content_len)
        self._validate_frame(post_body)

        self.send_response(200)
        self.end_headers()
        with open(results_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)


    def _validate_frame(self, post_body):
        if len(post_body) > FrameEncoder.MAX_FILE_SIZE:
            msg = 'Posted file was too large'
            self.send_error(400, msg)
            raise Exception(msg)

        if not post_body.startswith(b'\x89PNG\x0d\x0a\x1a\x0a'):
            msg = 'Expected a png file, but magic bytes were wrong.'
            self.send_error(400, msg)
            raise Exception(msg)

        frame = cv2.imdecode(np.frombuffer(post_body, dtype=np.uint8), cv2.IMREAD_COLOR)
        size = mpf_util.Size.from_frame(frame)
        if size.width * size.height > FrameEncoder.MAX_PIXELS:
            msg = 'Image contained too many pixels.'
            self.send_error(400, msg)
            raise Exception(msg)

        if size.width < FrameEncoder.MIN_DIMENSION_LENGTH:
            msg = 'Image was not wide enough.'
            self.send_error(400, msg)
            raise Exception(msg)

        if size.height < FrameEncoder.MIN_DIMENSION_LENGTH:
            msg = 'Image was not tall enough.'
            self.send_error(400, msg)
            raise Exception(msg)

        if size.width > FrameEncoder.MAX_DIMENSION_LENGTH:
            msg = 'Image was too wide.'
            self.send_error(400, msg)
            raise Exception(msg)

        if size.height > FrameEncoder.MAX_DIMENSION_LENGTH:
            msg = 'Image was too tall.'
            self.send_error(400, msg)
            raise Exception(msg)




if __name__ == '__main__':
    unittest.main(verbosity=2)
