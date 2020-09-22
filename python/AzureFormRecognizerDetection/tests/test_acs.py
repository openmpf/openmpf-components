#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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
import acs_form_recognizer_component
from acs_form_recognizer_component import AcsFormRecognizerComponent, FrameEncoder

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../acs_form_recognizer_component'))


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

    def test_resize_due_to_compression(self):
        img = cv2.imread(get_test_file('downsampling/noise.png'))

        original_size = mpf_util.Size.from_frame(img)
        encoded_frame, new_size = FrameEncoder().resize_and_encode(img)
        self.assertNotEqual(original_size, new_size)

        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)
        self.assertEqual(original_size.width // original_size.height, new_size.width // new_size.height)
        self.assertTrue(new_size.width <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(new_size.height <= FrameEncoder.MAX_DIMENSION_LENGTH)
        self.assertTrue(len(encoded_frame) <= FrameEncoder.MAX_FILE_SIZE)

    def test_pdf_form(self):
        self.set_results_path(get_test_file('regular-forms/pdf-results.json'))
        job = mpf.GenericJob('Test', get_test_file('regular-forms/form-recognizer-demo-sample1.pdf'),
                             get_test_properties(), {}, None)
        detections = list(AcsFormRecognizerComponent().get_detections_from_generic(job))

        self.assertEqual(3, len(detections))

        line_detection = detections[0]
        self.assertEqual('MERGED_LINES', line_detection.detection_properties['OUTPUT_TYPE'])
        self.assertTrue('Phone: 432-555-0189' in line_detection.detection_properties['TEXT'])
        self.assertTrue('Email: contoso@example.com' in line_detection.detection_properties['TEXT'])
        self.assertTrue('Invoice Date:\n4/14/2019' in line_detection.detection_properties['TEXT'])

        table_detection = detections[1]
        self.assertEqual('TABLE', table_detection.detection_properties['OUTPUT_TYPE'])
        self.assertTrue('Item #,Description,Qty,Unit Price,Discount,Price' in
                        table_detection.detection_properties['TABLE_CSV_OUTPUT'])
        self.assertTrue('Z4567,Invoice 3-456-2 Data 1,39,$ 5.00,$ -,$ 195.00' in
                        table_detection.detection_properties['TABLE_CSV_OUTPUT'])

    def test_png_form(self):
        self.set_results_path(get_test_file('regular-forms/png-results.json'))
        job = mpf.ImageJob('Test', get_test_file('regular-forms/contoso-invoice.png'), get_test_properties(), {}, None)
        detections = list(AcsFormRecognizerComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))

        line_detection = detections[0]
        self.assertEqual('MERGED_LINES', line_detection.detection_properties['OUTPUT_TYPE'])
        self.assertTrue('Contoso, Ltd.' in line_detection.detection_properties['TEXT'])
        self.assertTrue('554 Magnolia Way\nWalnut, WY, 98432' in line_detection.detection_properties['TEXT'])
        self.assertTrue('Thank you for your business!' in line_detection.detection_properties['TEXT'])

    def test_upsampling(self):
        self.set_results_path(get_test_file('upsampling/tiny-image-results.json'))
        job = mpf.ImageJob('Test', get_test_file('upsampling/tiny-image.png'), get_test_properties(), {}, None)
        detections = list(AcsFormRecognizerComponent().get_detections_from_image(job))

        self.assertEqual(1, len(detections))
        detection = detections[0]
        self.assertEqual('It was the best of times, it was the worst of times', detection.detection_properties['TEXT'])

    def test_no_results_img(self):
        self.set_results_path(get_test_file('no-results/black-results.json'))
        job = mpf.ImageJob('Test', get_test_file('no-results/black.png'), get_test_properties(), {}, None)
        detections = list(AcsFormRecognizerComponent().get_detections_from_image(job))
        self.assertEqual(0, len(detections))

    def test_get_acs_url_default_options(self):
        acs_url = 'http://localhost:10669/formrecognizer/v2.1-preview.1/Layout/analyze'
        self.assertEqual('http://localhost:10669/formrecognizer/v2.1-preview.1/Layout/analyze?includeTextDetails=true',
                         acs_form_recognizer_component.JobRunner.get_acs_url(dict(ACS_URL=acs_url)))

    def test_get_acs_url_preserves_existing_query_params(self):
        acs_url = 'http://localhost:10669/formrecognizer/v2.1-preview.1/Layout/analyze?hello=world'
        properties = dict(ACS_URL=acs_url)
        url = acs_form_recognizer_component.JobRunner.get_acs_url(properties)

        if '?d' in url:
            self.assertEqual('http://localhost:10669/formrecognizer/'
                             'v2.1-preview.1/Layout/analyze?includeTextDetails=true&hello=world', url)
        else:
            self.assertEqual('http://localhost:10669/formrecognizer/'
                             'v2.1-preview.1/Layout/analyze?hello=world&includeTextDetails=true', url)

    def test_get_acs_url_modified_parameter(self):
        acs_url = 'http://localhost:10669/formrecognizer/v2.1-preview.1/Layout/analyze'
        properties = dict(ACS_URL=acs_url, INCLUDE_TEXT_DETAILS='false')

        url = acs_form_recognizer_component.JobRunner.get_acs_url(properties)
        self.assertEqual('http://localhost:10669/formrecognizer/'
                         'v2.1-preview.1/Layout/analyze?includeTextDetails=false', url)


def get_test_properties(**extra_properties):
    return {
        'ACS_URL': os.getenv('ACS_URL', 'http://localhost:10669/formrecognizer/v2.1-preview.1/Layout/analyze'),
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

        if self.headers['Content-Type'] == 'application/octet-stream':
            self._validate_frame(post_body)

        self.send_response(200)
        self.send_header("Operation-Location", "http://localhost:10669/formrecognizer/v2.1-preview.1/Layout/analyze")
        self.end_headers()

    def do_GET(self):
        results_path = self.server.get_results_path()
        self.send_response(200)
        self.end_headers()
        with open(results_path, 'rb') as f:
            shutil.copyfileobj(f, self.wfile)

    def _validate_frame(self, post_body):
        if len(post_body) > FrameEncoder.MAX_FILE_SIZE:
            msg = 'Posted file was too large'
            self.send_error(400, msg)
            raise Exception(msg)

        if not post_body.startswith(b'\x89PNG\x0d\x0a\x1a\x0a') and not post_body.startswith(b'%PDF'):
            msg = 'Expected a png or pdf file, but magic bytes were wrong.'
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
