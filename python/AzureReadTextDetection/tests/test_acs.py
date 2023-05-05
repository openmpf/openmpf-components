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

import http.server
import logging
import os
import queue
import shutil
import sys
import threading
from typing import ClassVar
import unittest
from unittest import mock


import cv2
import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../acs_read_detection_component'))

import acs_read_detection_component
from acs_read_detection_component import AcsReadDetectionComponent, FrameEncoder


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

    def assert_contains_tag(self, detection, tag, msg=None):
        self.assertIn(tag, detection.detection_properties['TAGS'], msg)

    def assert_does_not_contain_tag(self, detection, tag, msg=None):
        self.assertNotIn(tag, detection.detection_properties['TAGS'], msg)

    def assert_new_size(self, initial_width, initial_height, expected_width, expected_height):
        frame = np.zeros((initial_height, initial_width, 3), dtype=np.uint8)
        encoded_frame, new_size = FrameEncoder().resize_and_encode(frame)

        self.assertEqual(expected_width, new_size.width)
        self.assertEqual(expected_height, new_size.height)

        self.assertEqual(initial_width // initial_height, new_size.width // new_size.height)

        self.assertTrue(new_size.width >= FrameEncoder.MIN_DIMENSION_LENGTH)
        self.assertTrue(new_size.height >= FrameEncoder.MIN_DIMENSION_LENGTH)

        self.assertTrue(new_size.width <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(new_size.height <= FrameEncoder.MAX_DIMENSION_LENGTH)

        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)

    def test_resize_frame_due_to_pixel_count(self):
        self.assert_new_size(1234, 421, 1234, 421)
        self.assert_new_size(12000, 600, 10000, 500)
        self.assert_new_size(600, 12000, 500, 10000)
        self.assert_new_size(750, 13324, 562, 10000)
        self.assert_new_size(1000, 40, 1250, 50)

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
        frame = np.zeros((10, 4400, 3), np.uint8)
        with self.assertRaises(mpf.DetectionException) as cm:
            FrameEncoder().resize_and_encode(frame)
        self.assertEqual(mpf.DetectionError.BAD_FRAME_SIZE, cm.exception.error_code)

    # Set the class max file size limit to a smaller value to force resizing.
    @mock.patch.object(FrameEncoder, 'MAX_FILE_SIZE', new_callable=lambda: 20 * 1024 * 1024)
    def test_resize_due_to_compression(self, _):
        img = cv2.imread(get_test_file('downsampling/noise.png'))

        original_size = mpf_util.Size.from_frame(img)
        encoded_frame, new_size = FrameEncoder().resize_and_encode(img)
        self.assertNotEqual(original_size, new_size)

        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)
        self.assertEqual(original_size.width // original_size.height, new_size.width // new_size.height)
        self.assertTrue(new_size.width <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(new_size.height <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)

    def test_pdf(self):
        self.set_results_path(get_test_file('regular/pdf-results.json'))
        job = mpf.GenericJob('Test', get_test_file('regular/form-recognizer-demo-sample1.pdf'),
                             get_test_properties(), {}, None)
        detections = list(AcsReadDetectionComponent().get_detections_from_generic(job))

        self.assertEqual(1, len(detections))

        line_detection = detections[0]
        self.assertAlmostEqual(1.0, line_detection.confidence)
        self.assertEqual('MERGED_LINES', line_detection.detection_properties['OUTPUT_TYPE'])
        self.assertEqual('1', line_detection.detection_properties['PAGE_NUM'])
        self.assertTrue('Phone: 432-555-0189' in line_detection.detection_properties['TEXT'])
        self.assertTrue('Email: contoso@example.com' in line_detection.detection_properties['TEXT'])
        self.assertTrue('Invoice Date:\n4/14/2019' in line_detection.detection_properties['TEXT'])


    def test_png(self):
        self.set_results_path(get_test_file('regular/png-results.json'))
        job = mpf.ImageJob('Test', get_test_file('regular/contoso-invoice.png'), get_test_properties(), {}, None)
        detections = list(AcsReadDetectionComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))

        line_detection = detections[0]
        self.assertAlmostEqual(0.9586029411764704, line_detection.confidence)
        self.assertEqual('MERGED_LINES', line_detection.detection_properties['OUTPUT_TYPE'])
        self.assertEqual('1', line_detection.detection_properties['PAGE_NUM'])
        self.assertTrue('Contoso, Ltd.' in line_detection.detection_properties['TEXT'])
        self.assertTrue('554 Magnolia Way\nWalnut, WY, 98432' in line_detection.detection_properties['TEXT'])
        self.assertTrue('Thank you for your business!' in line_detection.detection_properties['TEXT'])

    def test_chinese_text(self):
        self.set_results_path(get_test_file('foreign/chinese-two-cities-results.json'))
        job = mpf.ImageJob('Test', get_test_file('foreign/chinese-two-cities.png'), get_test_properties(), {}, None)

        detections = list(AcsReadDetectionComponent().get_detections_from_image(job))
        self.assertEqual(1, len(detections))
        detection = detections[0]

        with open(get_test_file('foreign/chinese-two-cities-expected-text.txt')) as f:
            expected_text = f.read().strip()

        self.assertEqual(expected_text, detection.detection_properties['TEXT'])


    def run_image_test(self, image_file_name, results_file_name, expected_detection_rect, expected_rotation):
        self.set_results_path(get_test_file(results_file_name))

        img_path = get_test_file(image_file_name)
        job = mpf.ImageJob('Test', img_path, get_test_properties(), {}, None)
        detections = list(AcsReadDetectionComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))
        detection = detections[0]
        self.assertEqual(human_rights_image_text, detection.detection_properties['TEXT'])
        self.assertEqual(expected_detection_rect,
                       mpf_util.Rect.from_image_location(detection))
        self.assertAlmostEqual(expected_rotation, float(detection.detection_properties['ROTATION']))


    def test_image_file(self):
        self.run_image_test('orientation/img/eng.png', 'orientation/img/eng-results.json',
                            mpf_util.Rect(109, 56, 995, 432), 0)


    def test_image_15deg_rotation(self):
        self.run_image_test('orientation/img/eng-15deg.png', 'orientation/img/eng-15deg-results.json',
                            mpf_util.Rect(66, 194, 998, 433), 15.219872160655177)


    def test_image_90deg_rotation(self):
        self.run_image_test('orientation/img/eng-90deg.png', 'orientation/img/eng-90deg-results.json',
                            mpf_util.Rect(55, 1081, 996, 434), 90.0)


    def test_image_100deg_rotation(self):
        self.run_image_test('orientation/img/eng-100deg.png', 'orientation/img/eng-100deg-results.json',
                            mpf_util.Rect(145, 1120, 994, 434), 100.31515702822162)


    def test_image_180deg_rotation(self):
        self.run_image_test('orientation/img/eng-180deg.png', 'orientation/img/eng-180deg-results.json',
                            mpf_util.Rect(1081, 584, 996, 433), 180.0)


    def test_image_200deg_rotation(self):
        self.run_image_test('orientation/img/eng-200deg.png', 'orientation/img/eng-200deg-results.json',
                            mpf_util.Rect(1139, 414, 997, 434), 200.05181234876986)


    def test_image_270deg_rotation(self):
        self.run_image_test('orientation/img/eng-270deg.png', 'orientation/img/eng-270deg-results.json',
                            mpf_util.Rect(585, 108, 997, 434), 270.0)


    def test_image_290deg_rotation(self):
        self.run_image_test('orientation/img/eng-290deg.png', 'orientation/img/eng-290deg-results.json',
                            mpf_util.Rect(407, 128, 1002, 433), 290.0075209483723)


    def test_video(self):
        test_video_frame_count = 8
        for i in range(test_video_frame_count):
            self.set_results_path(get_test_file('orientation/video/results-frame{}.json'.format(i)))

        job = mpf.VideoJob('Test', get_test_file('orientation/video/test-vid.avi'), 0, -1, get_test_properties(), {},
                           None)
        tracks = list(AcsReadDetectionComponent().get_detections_from_video(job))

        self.assertEqual(8, len(tracks))

        detections_indexed_by_frame = {}

        for track in tracks:
            self.assertEqual(1, len(track.frame_locations))
            detection = track.frame_locations[track.start_frame]
            self.assertFalse(track.start_frame in detections_indexed_by_frame)

            self.assertEqual(human_rights_image_text, track.detection_properties['TEXT'])
            self.assertEqual(human_rights_image_text, detection.detection_properties['TEXT'])

            detections_indexed_by_frame[track.start_frame] = detection

        self.assertEqual(mpf_util.Rect(1106, 49, 438, 998),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[0]))
        self.assertAlmostEqual(270.2616227456948, float(detections_indexed_by_frame[0].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(64, 193, 1000, 438),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[1]))
        self.assertAlmostEqual(14.938792576116121, float(detections_indexed_by_frame[1].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(54, 1085, 999, 437),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[2]))
        self.assertAlmostEqual(90.0, float(detections_indexed_by_frame[2].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(141, 1123, 1002, 438),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[3]))
        self.assertAlmostEqual(99.94173427712408, float(detections_indexed_by_frame[3].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(1087, 587, 1003, 440),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[4]))
        self.assertAlmostEqual(180.0571243873999, float(detections_indexed_by_frame[4].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(1141, 416, 999, 437),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[5]))
        self.assertAlmostEqual(199.93904791680956, float(detections_indexed_by_frame[5].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(590, 105, 1001, 441),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[6]))
        self.assertAlmostEqual(270.28619032471966, float(detections_indexed_by_frame[6].detection_properties['ROTATION']))

        self.assertEqual(mpf_util.Rect(410, 128, 997, 436),
                         mpf_util.Rect.from_image_location(detections_indexed_by_frame[7]))
        self.assertAlmostEqual(290.1057533495411, float(detections_indexed_by_frame[7].detection_properties['ROTATION']))

    def test_upsampling(self):
        self.set_results_path(get_test_file('upsampling/tiny-image-results.json'))
        job = mpf.ImageJob('Test', get_test_file('upsampling/tiny-image.png'), get_test_properties(), {}, None)
        detections = list(AcsReadDetectionComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))
        detection = detections[0]
        self.assertEqual('It was the best of times, it was the worst of times', detection.detection_properties['TEXT'])

    def test_no_results_img(self):
        self.set_results_path(get_test_file('no-results/black-results.json'))
        job = mpf.ImageJob('Test', get_test_file('no-results/black.png'), get_test_properties(), {}, None)
        detections = list(AcsReadDetectionComponent().get_detections_from_image(job))
        self.assertEqual(0, len(detections))

    def test_get_acs_url_default_options(self):
        acs_url = 'http://localhost:10669/vision/v3.1/read/analyze'
        self.assertEqual('http://localhost:10669/vision/v3.1/read/analyze',
                         acs_read_detection_component.JobRunner.get_acs_url(dict(ACS_URL=acs_url)))

    def test_get_acs_url_preserves_existing_query_params(self):
        acs_url = 'http://localhost:10669/vision/v3.1/read/analyze?hello=world'
        properties = dict(ACS_URL=acs_url, LANGUAGE='en')
        url = acs_read_detection_component.JobRunner.get_acs_url(properties)

        if '?l' in url:
            self.assertEqual('http://localhost:10669/vision/'
                             'v3.1/read/analyze?language=en&hello=world', url)
        else:
            self.assertEqual('http://localhost:10669/vision/'
                             'v3.1/read/analyze?hello=world&language=en', url)

    def test_get_acs_url_modified_parameter(self):
        acs_url = 'http://localhost:10669/vision/v3.1/read/analyze'
        properties = dict(ACS_URL=acs_url, LANGUAGE='en')

        url = acs_read_detection_component.JobRunner.get_acs_url(properties)
        self.assertEqual('http://localhost:10669/vision/'
                         'v3.1/read/analyze?language=en', url)


def get_test_properties(**extra_properties):
    return {
        'ACS_URL': os.getenv('ACS_URL', 'http://localhost:10669/vision/v3.1/read/analyze'),
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
        if not self.headers['Ocp-Apim-Subscription-Key']:
            self.send_error(403, 'Expected the Ocp-Apim-Subscription-Key header to contain the subscription key.')
            return

        if self.headers['Content-Type'] != 'application/octet-stream' and \
                self.headers['Content-Type'] != 'application/pdf':
            self.send_error(415, 'Expected content type to be application/octet-stream or application/pdf')
            return

        content_len = int(self.headers['Content-Length'])
        post_body = self.rfile.read(content_len)

        self._validate_body(post_body)

        self.send_response(200)
        self.send_header("Operation-Location", "http://localhost:10669/vision/v3.1/read/analyze")
        self.end_headers()

    def do_GET(self):
        results_path = self.server.get_results_path()
        self.send_response(200)
        self.end_headers()
        with open(results_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)

    def _validate_body(self, post_body):
        if len(post_body) > FrameEncoder.MAX_FILE_SIZE:
            msg = 'Posted file was too large'
            self.send_error(400, msg)
            raise Exception(msg)

        if self.headers['Content-Type'] == 'application/pdf':
            if post_body.startswith(b'%PDF'):
                return True
            else:
                msg = 'Expected a PDF file, but magic bytes were wrong.'
                self.send_error(400, msg)
                raise Exception(msg)

        if not post_body.startswith(b'\x89PNG\x0d\x0a\x1a\x0a'):
            msg = 'Expected a png file, but magic bytes were wrong.'
            self.send_error(400, msg)
            raise Exception(msg)

        frame = cv2.imdecode(np.frombuffer(post_body, dtype=np.uint8), cv2.IMREAD_COLOR)
        size = mpf_util.Size.from_frame(frame)

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
