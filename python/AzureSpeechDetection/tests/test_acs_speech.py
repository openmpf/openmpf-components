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
import logging
import os
import json
import unittest
import threading
import urllib.parse
from typing import ClassVar
from http.server import HTTPServer, SimpleHTTPRequestHandler
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from acs_speech_component import AcsSpeechComponent

import mpf_component_api as mpf
import mpf_component_util as mpf_util


local_path = os.path.realpath(os.path.dirname(__file__))
base_local_path = os.path.join(local_path, 'test_data')
base_url_path = '/speechtotext/'
api_version = '2025-10-15'

test_port = 10669
origin = 'http://localhost:{}'.format(test_port)
url_prefix = origin + base_url_path.rstrip('/')
transcription_url = url_prefix
blobs_url = url_prefix + '/recordings'
outputs_url = url_prefix + '/outputs'
models_url = url_prefix + '/models'
container_url = 'https://account_name.blob.core.endpoint.suffix/container_name'
account_sas = '[sas_url]'


def get_test_properties(**extra_properties):
    return dict(
        ACS_URL=transcription_url,
        ACS_API_VERSION=api_version,
        ACS_SUBSCRIPTION_KEY='acs_subscription_key',
        ACS_BLOB_CONTAINER_URL=container_url,
        ACS_BLOB_SERVICE_KEY='acs_blob_service_key',
        **extra_properties
    )

logging.basicConfig(level=logging.DEBUG)


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
        self.mock_server.sas_enabled = False

    @staticmethod
    def run_patched_jobs(comp, mode, *jobs):
        if mode == 'audio':
            detection_func = comp.get_detections_from_audio
        elif mode == 'video':
            detection_func = comp.get_detections_from_video
        else:
            raise Exception('Mode must be "audio" or "video".')

        with patch.object(comp.processor.acs, 'delete_blob'), \
                patch.object(comp.processor.acs, 'get_blob_client'), \
                patch.object(
                    comp.processor.acs,
                    'generate_account_sas',
                    return_value=account_sas):
            return list(map(detection_func, jobs))

    def test_audio_file(self):
        self.mock_server.sas_enabled = True
        job = mpf.AudioJob(
            job_name='test_audio',
            data_uri=self._get_test_file('left.wav'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE='FALSE',
                LANGUAGE='EN-us',
                USE_SAS_AUTH='TRUE'
            ),
            media_properties={},
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        result, = self.run_patched_jobs(comp, 'audio', job)

        self.assertEqual(1, len(result))
        transcript = result[0].detection_properties['TRANSCRIPT']
        self.assertEqual(
            "There's 3 left on the left side, the one closest to us.",
            transcript
        )

    def test_video_file(self):
        job = mpf.VideoJob(
            job_name='test_video',
            data_uri=self._get_test_file('left.avi'),
            start_frame=0,
            stop_frame=-1,
            job_properties=get_test_properties(
                DIARIZE='FALSE',
                LANGUAGE='En-Us'
            ),
            media_properties=dict(
                FPS='24'
            ),
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        result, = self.run_patched_jobs(comp, 'video', job)

        self.assertEqual(1, len(result))
        transcript = result[0].detection_properties['TRANSCRIPT']
        self.assertEqual(
            "There's 3 left on the left side, the one closest to us.",
            transcript
        )

    def test_diarization(self):
        job_raw = mpf.AudioJob(
            job_name='test_diarization_disabled',
            data_uri=self._get_test_file('poverty.mp3'),
            start_time=770000,
            stop_time=830000,
            job_properties=get_test_properties(
                DIARIZE='FALSE',
                LANGUAGE='en-US'
            ),
            media_properties={},
            feed_forward_track=None
        )

        job_dia = mpf.AudioJob(
            job_name='test_diarization_enabled',
            data_uri=self._get_test_file('poverty.mp3'),
            start_time=770000,
            stop_time=830000,
            job_properties=get_test_properties(
                DIARIZE='TRUE',
                LANGUAGE='en-US'
            ),
            media_properties={},
            feed_forward_track=None
        )

        comp = AcsSpeechComponent()
        results = self.run_patched_jobs(comp, 'audio', job_raw, job_dia)

        # There should be two speakers with diarization, one without
        len_raw, len_dia = [
            len(set([
                track.detection_properties['SPEAKER_ID']
                for track in result
            ]))
            for result in results
        ]
        self.assertEqual(1, len_raw)
        self.assertEqual(2, len_dia)

    def test_language(self):
        job_en = mpf.AudioJob(
            job_name='test_bilingual_english',
            data_uri=self._get_test_file('bilingual.mp3'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE='TRUE',
                LANGUAGE='en-us'
            ),
            media_properties={},
            feed_forward_track=None
        )

        job_sp = mpf.AudioJob(
            job_name='test_bilingual_spanish',
            data_uri=self._get_test_file('bilingual.mp3'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE='TRUE',
                LANGUAGE='es-MX'
            ),
            media_properties={},
            feed_forward_track=None
        )
        comp = AcsSpeechComponent()
        res_en, res_sp = self.run_patched_jobs(comp, 'audio', job_en, job_sp)

        self.assertEqual(23, len(res_en))
        self.assertEqual(23, len(res_sp))

        # Audio switches from Spanish to English at about 0:58.
        #  Speaker 1 says one sentence in English at about 0:58-1:00.
        #  Speaker 2 begins at 1:04 and speaks only English.
        lang_switch = 58000

        mean = lambda lst: sum(lst) / len(lst)
        bad = [r.confidence for r in res_en if r.start_time < lang_switch]
        good = [r.confidence for r in res_en if r.start_time > lang_switch]
        self.assertGreater(mean(good), mean(bad))

        bad = [r.confidence for r in res_sp if r.start_time > lang_switch]
        good = [r.confidence for r in res_sp if r.start_time < lang_switch]
        self.assertGreater(mean(good), mean(bad))

    def test_missing_audio(self):
        job = mpf.AudioJob(
            job_name='test_missing_audio',
            data_uri=self._get_test_file('silence.mp3'),
            start_time=0,
            stop_time=-1,
            job_properties=get_test_properties(
                DIARIZE='FALSE',
                LANGUAGE='en-US'
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


class MockServer(HTTPServer):
    def __init__(self, base_local_path, base_url_path, server_address):
        super(MockServer, self).__init__(server_address, MockRequestHandler)
        self.base_local_path = base_local_path
        self.base_url_path = base_url_path
        self.jobs = set()
        self._sas_enabled = False
        self.sas_lock = threading.Lock()
        threading.Thread(target=self.serve_forever).start()

    @property
    def sas_enabled(self):
        with self.sas_lock:
            return self._sas_enabled

    @sas_enabled.setter
    def sas_enabled(self, value):
        with self.sas_lock:
            self._sas_enabled = value


class MockRequestHandler(SimpleHTTPRequestHandler):
    server: MockServer

    def translate_path(self, path):
        """Translate a /-separated PATH to the local filename syntax.
        Components that mean special things to the local file system
        (e.g. drive or directory names) are ignored.  (XXX They should
        probably be diagnosed.)
        """
        parsed = urllib.parse.urlparse(path)
        path_only = parsed.path

        if path_only.startswith(self.server.base_url_path):
            path_only = path_only[len(self.server.base_url_path):]
            if path_only == '':
                path_only = '/'
        else:
            return None

        path_only = super().translate_path(path_only)
        rel_path = os.path.relpath(path_only, os.getcwd())
        full_path = os.path.join(self.server.base_local_path, rel_path)
        return full_path

    def _validate_headers(self):
        if not self.headers.get('Ocp-Apim-Subscription-Key'):
            self.send_error(403, 'Forbidden')
            self.wfile.write('No permission to access this resource.'.encode())
            return False

        if self.headers.get('Content-Type') != 'application/json':
            self.send_error(400, 'BadRequest')
            self.wfile.write('The request has been incorrect.'.encode())
            return False
        return True

    def _get_tail_and_query(self):
        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith(self.server.base_url_path):
            return None, None
        tail = parsed.path[len(self.server.base_url_path):]
        query = urllib.parse.parse_qs(parsed.query)
        return tail, query

    def do_GET(self):
        if not isinstance(self.path, str):
            self.send_response(442)
            self.end_headers()
            self.wfile.close()
            return

        tail, query = self._get_tail_and_query()
        if tail is None:
            self.send_error(404, 'NotFound')
            return



        if tail == 'transcriptions/locales':
            self.path = urllib.parse.urlparse(self.path).path + '.json'
        elif tail.startswith('transcriptions'):
            jobname = tail[len('transcriptions/'):]
            if jobname.endswith('.json'):
                jobname = jobname[:-len('.json')]
            if jobname.endswith('/files'):
                jobname = jobname[:-len('/files')]

            self._validate_headers()
            if jobname not in self.server.jobs:
                self.send_error(404, 'NotFound')
                self.wfile.write(
                    'The specified entity cannot be found.'.encode()
                )
                return
        elif tail.startswith('outputs'):
            pass
        else:
            self.send_error(404, 'NotFound')
            self.wfile.write(
                'The resource you are looking for has been removed, had its '
                'name changed, or is temporarily unavailable.'.encode()
            )
            return

        super(MockRequestHandler, self).do_GET()

    def do_DELETE(self):
        if not isinstance(self.path, str):
            self.send_response(442)
            self.end_headers()
            self.wfile.close()
            return

        tail, query = self._get_tail_and_query()
        if tail is None:
            self.send_error(404, 'NotFound')
            return

        if query.get('api-version', [''])[0] != api_version:
            self.send_error(400, 'InvalidApiVersion')
            return

        if tail.startswith('transcriptions'):
            self._validate_headers()
            self.send_response(204)
            self.end_headers()
        else:
            self.send_error(404)

    def do_POST(self):
        tail, query = self._get_tail_and_query()
        if tail != 'transcriptions:submit':
            self.send_error(404, 'NotFound')
            return

        if query.get('api-version', [''])[0] != api_version:
            self.send_error(400, 'InvalidApiVersion')
            return

        if not self._validate_headers():
            return

        content_len = int(self.headers['Content-Length'])
        post_body = self.rfile.read(content_len)
        data = json.loads(post_body)

        if not data.get('contentUrls'):
            self.send_error(400, 'InvalidPayload')
            self.wfile.write(
                'Only absolute URIs containing a valid scheme, authority and '
                'path are allowed for transcription.recordingsUrl.'.encode()
            )
            return

        if not data.get('displayName'):
            self.send_error(400, 'InvalidPayload')
            self.wfile.write(
                'A non empty string is required for '
                'transcription.name.'.encode()
            )
            return

        if not data.get('locale'):
            self.send_error(400, 'InvalidPayload')
            self.wfile.write(
                'A non empty string is required for '
                'transcription.locale.'.encode()
            )
            return

        properties = data.get('properties')
        if not properties or properties.get('wordLevelTimestampsEnabled') is not True:
            raise Exception('Expected wordLevelTimestampsEnabled=True')

        if 'timeToLiveHours' not in properties:
            raise Exception('Expected timeToLiveHours')

        diarization = properties.get('diarization')
        if not isinstance(diarization, dict):
            raise Exception('Expected diarization object')
        if 'enabled' not in diarization:
            raise Exception('Expected diarization.enabled')

        recording_url = data.get('contentUrls')[0]
        if self.server.sas_enabled:
            if not recording_url.endswith('?' + account_sas):
                raise Exception('SAS enabled, but sas not found in recording URL.')
        elif '?' in recording_url:
            raise Exception('SAS disabled, but sas found in recording URL.')

        job_name = data.get('displayName')
        self.server.jobs.add(job_name)

        self.send_response(201)
        self.send_header(
            'Location',
            f'{origin}{self.server.base_url_path}transcriptions:submit/{job_name}?api-version={api_version}'
        )
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({
            'self': f'{origin}{self.server.base_url_path}transcriptions/{job_name}.json?api-version={api_version}'
        }).encode())


if __name__ == '__main__':
    unittest.main(verbosity=2)