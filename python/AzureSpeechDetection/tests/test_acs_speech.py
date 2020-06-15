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
import json
import unittest

from threading import Thread
from typing import ClassVar
from http.server import HTTPServer as BaseHTTPServer, SimpleHTTPRequestHandler

from unittest.mock import patch, Mock

# Add pyspeech_component to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from acs_speech_component import AcsSpeechComponent

import mpf_component_api as mpf
import mpf_component_util as mpf_util


local_path = os.path.realpath(os.path.dirname(__file__))
base_local_path = os.path.join(local_path, 'test_data')
base_url_path = '/api/speechtotext/v2.0/'

test_port = 10669
origin = f'http://localhost:{test_port}'
transcription_url = f'{origin}{base_url_path}transcriptions'
blobs_url = f'{origin}{base_url_path}recordings'
outputs_url = f'{origin}{base_url_path}outputs'
def get_test_properties(**extra_properties):
    return dict(
        ACS_URL=transcription_url,
        ACS_ACCOUNT_NAME='acs_account_name',
        ACS_SUBSCRIPTION_KEY='acs_subscription_key',
        ACS_SPEECH_KEY='acs_speech_key',
        ACS_CONTAINER_NAME='acs_container_name',
        ACS_ENDPOINT_SUFFIX='acs.endpoint.suffix',
        **extra_properties
    )

class TestAcsSpeech(unittest.TestCase):
    mock_server: ClassVar['MockServer']

    @classmethod
    def setUpClass(cls):
        cls.mock_server = MockServer(
            base_local_path,
            base_url_path,
            ('', test_port)
        )

    @classmethod
    def tearDownClass(cls):
        cls.mock_server.shutdown()

    def tearDown(self):
        self.mock_server.jobs = set()

    @staticmethod
    def run_patched_jobs(comp, mode, *jobs):
        if mode == 'audio':
            detection_func = comp.get_detections_from_audio
        elif mode == 'video':
            detection_func = comp.get_detections_from_video
        else:
            raise Exception("Mode must be 'audio' or 'video'.")

        with patch.object(comp.processor.acs, 'delete_blob'),\
                patch.object(comp.processor.acs, 'get_blob_client'),\
                patch.object(
                    comp.processor.acs,
                    'generate_account_sas',
                    return_value="[sas_url]"):
            return list(map(detection_func, jobs))

    def test_audio_file(self):
        job = mpf.AudioJob(
            job_name="test_audio",
            data_uri=self._get_test_file('left.wav'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE="FALSE",
                LANGUAGE="en-US"
            ),
            media_properties={},
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        result, = self.run_patched_jobs(comp, 'audio', job)

        self.assertEqual(1, len(result))
        transcript = result[0].detection_properties['TRANSCRIPT']
        self.assertEqual("There's 3 left on the left side of the one closest to us.", transcript)


    def test_video_file(self):
        job = mpf.VideoJob(
            job_name='test_video',
            data_uri=self._get_test_file('left.avi'),
            start_frame=0,
            stop_frame=-1,
            job_properties=get_test_properties(
                DIARIZE="FALSE",
                LANGUAGE="en-US"
            ),
            media_properties=dict(
                FPS="24"
            ),
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        result, = self.run_patched_jobs(comp, 'video', job)

        self.assertEqual(1, len(result))
        transcript = result[0].detection_properties['TRANSCRIPT']
        self.assertEqual("There's 3 left on the left side the one closest to us.", transcript)

    def test_diarization(self):
        job_raw = mpf.AudioJob(
            job_name="test_diarization_disabled",
            data_uri=self._get_test_file('poverty.mp3'),
            start_time=770000,
            stop_time=830000,
            job_properties=get_test_properties(
                DIARIZE="FALSE",
                LANGUAGE="en-US"
            ),
            media_properties={},
            feed_forward_track=None
        )

        job_dia = mpf.AudioJob(
            job_name="test_diarization_enabled",
            data_uri=self._get_test_file('poverty.mp3'),
            start_time=770000,
            stop_time=830000,
            job_properties=get_test_properties(
                DIARIZE="TRUE",
                LANGUAGE="en-US"
            ),
            media_properties={},
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        results = self.run_patched_jobs(comp, 'audio', job_raw, job_dia)

        # There should be two speakers with diarization, one without
        len_raw, len_dia = [
            len(set([
                track.detection_properties['LONG_SPEAKER_ID']
                for track in result
            ]))
            for result in results
        ]
        self.assertEqual(1, len_raw)
        self.assertEqual(2, len_dia)

        # A nonzero start_time indicates to the component that this is a
        #  subjob, so all SPEAKER_IDs should be equal to 0
        ids_raw, ids_dia = [
            set([
                track.detection_properties['SPEAKER_ID']
                for track in result
            ])
            for result in results
        ]
        self.assertEqual(set(['0']), ids_raw)
        self.assertEqual(set(['0']), ids_dia)


    def test_language(self):
        job_en = mpf.AudioJob(
            job_name="test_bilingual_english",
            data_uri=self._get_test_file('bilingual.mp3'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE="TRUE",
                LANGUAGE="en-US"
            ),
            media_properties={},
            feed_forward_track=None
        )

        job_sp = mpf.AudioJob(
            job_name="test_bilingual_spanish",
            data_uri=self._get_test_file('bilingual.mp3'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE="TRUE",
                LANGUAGE="sp-MX"
            ),
            media_properties={},
            feed_forward_track=None
        )
        comp = AcsSpeechComponent()
        res_en, res_sp = self.run_patched_jobs(comp, 'audio', job_en, job_sp)

        self.assertEqual(15, len(res_en))
        self.assertEqual(22, len(res_sp))

        # Audio switches from Spanish to English at about 0:58.
        #  Speaker 1 says one sentence in English at about 0:58-1:00.
        #  Speaker 2 begins at 1:04 and speaks only English.
        lang_switch = 58000

        bad = [r.confidence for r in res_en if r.start_time < lang_switch]
        good = [r.confidence for r in res_en if r.start_time > lang_switch]
        self.assertGreater(min(good), max(bad))

        bad = [r.confidence for r in res_sp if r.start_time > lang_switch]
        good = [r.confidence for r in res_sp if r.start_time < lang_switch]
        self.assertGreater(min(good), max(bad))


    def test_missing_audio(self):
        job = mpf.AudioJob(
            job_name="test_missing_audio",
            data_uri=self._get_test_file('silence.mp3'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE="FALSE",
                LANGUAGE="en-US"
            ),
            media_properties=dict(
                DURATION='60020'
            ),
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        result, = self.run_patched_jobs(comp, 'audio', job)
        self.assertEqual(0, len(result))

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(
            os.path.dirname(__file__),
            'test_data',
            'recordings',
            filename
        )


class MockServer(BaseHTTPServer):
    def __init__(self, base_local_path, base_url_path, server_address):
        super(MockServer, self).__init__(server_address, MockRequestHandler)
        self._base_local_path = base_local_path
        self._base_url_path = base_url_path
        self.jobs = set()
        Thread(target=self.serve_forever).start()


class MockRequestHandler(SimpleHTTPRequestHandler):
    server: MockServer

    def translate_path(self, path):
        if path.startswith(self.server._base_url_path):
            path = path[len(self.server._base_url_path):]
            if path == '':
                path = '/'
        else:
            return None
        path = SimpleHTTPRequestHandler.translate_path(self, path)
        relpath = os.path.relpath(path, os.getcwd())
        fullpath = os.path.join(self.server._base_local_path, relpath)
        return fullpath

    def _validate_headers(self):
        if not self.headers.get('Ocp-Apim-Subscription-Key'):
            self.send_error(403, 'Forbidden')
            self.wfile.write("No permission to access this resource.".encode())
            return

        if self.headers.get('Content-Type') != 'application/json':
            self.send_error(400, 'BadRequest')
            self.wfile.write("The request has been incorrect.".encode())
            return

    def do_GET(self):
        if not (type(self.path) is str or type(self.path) is unicode):
            self.send_response(442)
            self.send_headers()
            self.wfile.close()
            return

        if not self.path.startswith(self.server._base_url_path):
            self.send_error(404, "NotFound")
            self.wfile.write("The resource you are looking for has been removed, had its name changed, or is temporarily unavailable.".encode())
            return
        tail = self.path[len(self.server._base_url_path):]

        if tail.startswith("transcriptions"):
            self._validate_headers()
            if self.path not in self.server.jobs:
                self.send_error(404, "NotFound")
                self.wfile.write("The specified entity cannot be found.".encode())
                return
        elif tail.startswith("outputs"):
            pass
        else:
            self.send_error(404, "NotFound")
            self.wfile.write("The resource you are looking for has been removed, had its name changed, or is temporarily unavailable.".encode())
            return

        if self.path.startswith(self.server._base_url_path):
            super(MockRequestHandler, self).do_GET()

    def do_DELETE(self):
        if not (type(self.path) is str or type(self.path) is unicode):
            self.send_response(442)
            self.send_headers()
            self.wfile.close()
            return

        if not self.path.startswith(self.server._base_url_path):
            self.send_error(404, "NotFound")
            self.wfile.write("The resource you are looking for has been removed, had its name changed, or is temporarily unavailable.".encode())
            return
        tail = self.path[len(self.server._base_url_path):]

        if tail.startswith("transcriptions"):
            self._validate_headers()
            self.send_response(204)
            if self.path in self.server.jobs:
                self.server.jobs.remove(self.path)
            self.end_headers()
        else:
            self.send_error(404)
            self.wfile.write("The resource you are looking for has been removed, had its name changed, or is temporarily unavailable.".encode())
            return


    def do_POST(self):
        self._validate_headers()

        content_len = int(self.headers['Content-Length'])
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body)

        if not data.get('recordingsUrl'):
            self.send_error(400, "InvalidPayload")
            self.wfile.write("Only absolute URIs containing a valid scheme, authority and path are allowed for transcription.recordingsUrl.".encode())
            return

        if not data.get('name'):
            self.send_error(400, "InvalidPayload")
            self.wfile.write("A non empty string is required for transcription.name.".encode())
            return

        if not data.get('locale'):
            self.send_error(400, "InvalidPayload")
            self.wfile.write("A non empty string is required for transcription.locale.".encode())
            return

        properties = data.get('properties')
        if not properties or properties.get('AddWordLevelTimestamps') != "True":
            raise Exception("Expected AddWordLevelTimestamps to be enabled")

        self.send_response(202)

        location = os.path.join(
            self.server._base_url_path,
            'transcriptions',
            data.get('name') + '.json'
        )
        self.server.jobs.add(location)
        self.send_header('Location', origin + location)
        self.end_headers()
        return


if __name__ == '__main__':
    unittest.main(verbosity=2)
