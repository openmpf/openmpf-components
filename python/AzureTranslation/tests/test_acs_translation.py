#!/usr/bin/env python3
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
import json
import logging
import os
import pathlib
import sys
import queue
import shutil
import threading
import unittest
import urllib.parse
from typing import ClassVar, List, NamedTuple, TypedDict
from unittest import mock

import mpf_component_api as mpf

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from nlp_text_splitter import TextSplitterModel, TextSplitter
from acs_translation_component.acs_translation_component import (AcsTranslationComponent,
    get_azure_char_count, TranslationClient, NewLineBehavior, ChineseAndJapaneseCodePoints,
    AcsTranslateUrlBuilder, get_n_azure_chars)

from acs_translation_component.convert_language_code import iso_to_bcp

# Set to true to test the WtP canine-s-1l model locally
# Note, this will download ~1 GB to your local storage.
LOCAL_TEST_WTP_MODEL = False
SEEN_TRACE_IDS = set()

CHINESE_SAMPLE_TEXT = '你好，你叫什么名字？'
CHINESE_SAMPLE_TEXT_ENG_TRANSLATE = 'Hello, what\'s your name?'

SPANISH_SAMPLE_TEXT = '¿Dónde está la biblioteca?'
SPANISH_SAMPLE_TEXT_ENG_TRANSLATE = 'Where\'s the library?'

TEST_DATA = pathlib.Path(__file__).parent / 'data'

logging.basicConfig(level=logging.DEBUG)

class TestAcsTranslation(unittest.TestCase):

    mock_server: ClassVar['MockServer']
    wtp_model: ClassVar['TextSplitterModel']
    spacy_model: ClassVar['TextSplitterModel']

    @classmethod
    def setUpClass(cls):
        cls.mock_server = MockServer()
        cls.wtp_model = TextSplitterModel("wtp-bert-mini", "cpu", "en")
        if LOCAL_TEST_WTP_MODEL:
            cls.wtp_adv_model = TextSplitterModel("wtp-canine-s-1l", "cuda", "zh")
        cls.spacy_model = TextSplitterModel("xx_sent_ud_sm", "cpu", "en")


    @classmethod
    def tearDownClass(cls):
        cls.mock_server.shutdown()

    @classmethod
    def set_results_file(cls, file_name):
        cls.mock_server.set_results_path(TEST_DATA / file_name)

    @classmethod
    def get_request(cls) -> 'Request':
        return cls.mock_server.get_request_info()

    @classmethod
    def get_request_body(cls) -> List['AcsRequestEntry']:
        return cls.get_request().body


    def tearDown(self):
        self.mock_server.drain_queues()

    def test_iso_code_checker(self):
        self.assertEqual('zh-hans', iso_to_bcp("ZH"))
        self.assertEqual('zh-hans', iso_to_bcp("Zh"))
        self.assertEqual('zh-hans', iso_to_bcp("zh"))
        self.assertEqual('zh-hans', iso_to_bcp("ZHO"))

        self.assertEqual('zh-hant', iso_to_bcp("ZHO-HANT"))
        self.assertEqual('zh-hant', iso_to_bcp("Zho-haNT"))
        self.assertEqual('zh-hant', iso_to_bcp("ZH-Hant"))
        self.assertEqual('zh-hans', iso_to_bcp("zh-HANS"))
        self.assertEqual('fr-ca', iso_to_bcp("fr-ca"))

    def test_simple_jobs(self):
        def validate_results(results):
            try:
                self.assertEqual(1, len(results))
                result = results[0]
                self.assertEqual(CHINESE_SAMPLE_TEXT, result.detection_properties['TEXT'])

                self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                                 result.detection_properties['TRANSLATION'])
                self.assertEqual('EN', result.detection_properties['TRANSLATION TO LANGUAGE'])

                self.assertEqual('zh-Hans',
                                 result.detection_properties['TRANSLATION SOURCE LANGUAGE'])
                self.assertAlmostEqual(
                    1.0,
                    float(result.detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))
                self.assertNotIn('SKIPPED TRANSLATION', result.detection_properties)

                detect_request_body = self.get_request_body()
                self.assertEqual(1, len(detect_request_body))
                self.assertEqual(CHINESE_SAMPLE_TEXT, detect_request_body[0]['Text'])

                translate_url, translate_request_body = self.get_request()
                translate_query_string: str = urllib.parse.urlparse(translate_url).query
                translate_query_dict = urllib.parse.parse_qs(translate_query_string)
                self.assertEqual(['3.0'], translate_query_dict['api-version'])
                self.assertEqual(['en'], translate_query_dict['to'])
                self.assertEqual(['zh-Hans'], translate_query_dict['from'])

                self.assertEqual(1, len(translate_request_body))
                self.assertEqual(CHINESE_SAMPLE_TEXT, translate_request_body[0]['Text'])
            finally:
                self.mock_server.drain_queues()

        detect_results_file = 'chinese-detect-result.json'
        translate_results_file = 'results-chinese.json'

        with self.subTest('Image'):
            self.set_results_file(detect_results_file)
            self.set_results_file(translate_results_file)
            ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))
            job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {}, ff_loc)
            validate_results(list(AcsTranslationComponent().get_detections_from_image(job)))

        with self.subTest('Audio'):
            self.set_results_file(detect_results_file)
            self.set_results_file(translate_results_file)
            ff_track = mpf.AudioTrack(0, 1, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))
            job = mpf.AudioJob('Test', 'test.wav', 0, 1, get_test_properties(), {}, ff_track)
            validate_results(list(AcsTranslationComponent().get_detections_from_audio(job)))

        with self.subTest('Generic'):
            self.set_results_file(detect_results_file)
            self.set_results_file(translate_results_file)
            ff_track = mpf.GenericTrack(-1, dict(TEXT=CHINESE_SAMPLE_TEXT))
            job = mpf.GenericJob('Test', 'test.pdf', get_test_properties(), {}, ff_track)
            validate_results(list(AcsTranslationComponent().get_detections_from_generic(job)))

        with self.subTest('Plain text file'):
            self.set_results_file(detect_results_file)
            self.set_results_file(translate_results_file)
            job = mpf.GenericJob('Test', str(TEST_DATA / 'chinese-text.txt'),
                                 get_test_properties(), {})
            validate_results(list(AcsTranslationComponent().get_detections_from_generic(job)))


    def test_video_job(self):
        self.set_results_file('chinese-detect-result.json')
        self.set_results_file('results-chinese.json')
        self.set_results_file('spanish-detect-result.json')
        self.set_results_file('results-spanish.json')
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TEXT=SPANISH_SAMPLE_TEXT))
            },
            dict(TEXT=CHINESE_SAMPLE_TEXT))

        job = mpf.VideoJob('test', 'test.jpg', 0, 1, get_test_properties(), {}, ff_track)

        results = list(AcsTranslationComponent().get_detections_from_video(job))
        self.assertEqual(1, len(results))
        result = results[0]

        self.assertEqual(CHINESE_SAMPLE_TEXT, result.detection_properties['TEXT'])
        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                         result.detection_properties['TRANSLATION'])
        self.assertEqual('EN', result.detection_properties['TRANSLATION TO LANGUAGE'])
        self.assertEqual('zh-Hans', result.detection_properties['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(
            1.0, float(result.detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))


        self.assertEqual(CHINESE_SAMPLE_TEXT,
            result.frame_locations[0].detection_properties['TEXT'])
        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
            result.frame_locations[0].detection_properties['TRANSLATION'])
        self.assertEqual('EN',
            result.frame_locations[0].detection_properties['TRANSLATION TO LANGUAGE'])
        self.assertEqual('zh-Hans',
            result.frame_locations[0].detection_properties['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(
            1.0,
            float(result.frame_locations[0]\
                  .detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))


        self.assertEqual(SPANISH_SAMPLE_TEXT,
            result.frame_locations[1].detection_properties['TEXT'])
        self.assertEqual(SPANISH_SAMPLE_TEXT_ENG_TRANSLATE,
            result.frame_locations[1].detection_properties['TRANSLATION'])
        self.assertEqual('EN',
            result.frame_locations[1].detection_properties['TRANSLATION TO LANGUAGE'])
        self.assertEqual('es',
            result.frame_locations[1].detection_properties['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(
            1.0,
            float(result.frame_locations[1]\
                  .detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))

        request_body1 = self.get_request_body()
        self.assertEqual(1, len(request_body1))
        self.assertEqual(CHINESE_SAMPLE_TEXT, request_body1[0]['Text'])

        request_body2 = self.get_request_body()
        self.assertEqual(1, len(request_body2))
        self.assertEqual(CHINESE_SAMPLE_TEXT, request_body2[0]['Text'])

        request_body3 = self.get_request_body()
        self.assertEqual(1, len(request_body3))
        self.assertEqual(SPANISH_SAMPLE_TEXT, request_body3[0]['Text'])

        request_body4 = self.get_request_body()
        self.assertEqual(1, len(request_body4))
        self.assertEqual(SPANISH_SAMPLE_TEXT, request_body4[0]['Text'])

        with self.assertRaises(queue.Empty, msg='Only four requests should have been sent.'):
            self.get_request_body()


    def test_detect_lang_disabled(self):
        self.set_results_file('results-chinese-with-auto-detect.json')
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))
        job = mpf.ImageJob('Test', 'test.jpg',
                           get_test_properties(DETECT_BEFORE_TRANSLATE='FALSE'), {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))

        self.assertEqual(1, len(results))
        result = results[0]
        self.assertEqual(CHINESE_SAMPLE_TEXT, result.detection_properties['TEXT'])

        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                         result.detection_properties['TRANSLATION'])
        self.assertEqual('EN', result.detection_properties['TRANSLATION TO LANGUAGE'])

        self.assertEqual('zh-Hans', result.detection_properties['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(
            1.0, float(result.detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))


        translate_url, translate_request_body = self.get_request()
        translate_query_string: str = urllib.parse.urlparse(translate_url).query
        translate_query_dict = urllib.parse.parse_qs(translate_query_string)
        self.assertEqual(['3.0'], translate_query_dict['api-version'])
        self.assertEqual(['en'], translate_query_dict['to'])
        self.assertIsNone(translate_query_dict.get('from'))

        self.assertEqual(1, len(translate_request_body))
        self.assertEqual(CHINESE_SAMPLE_TEXT, translate_request_body[0]['Text'])


    def test_no_feed_forward_location(self):
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {})
        with self.assertRaises(mpf.DetectionException) as cm:
            list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)


    def test_no_feed_forward_track(self):
        job = mpf.VideoJob('test', 'test.jpg', 0, 1, get_test_properties(), {})
        with self.assertRaises(mpf.DetectionException) as cm:
            list(AcsTranslationComponent().get_detections_from_video(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)


    @mock.patch('time.sleep')
    def test_reports_error_when_server_error(self, _):
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))},
            dict(TEXT=CHINESE_SAMPLE_TEXT))

        job = mpf.VideoJob('test', 'test.jpg', 0, 1,
                           get_test_properties(ACS_URL='http://localhost:10670/bad_request'), {},
                           ff_track)

        with self.assertRaises(mpf.DetectionException) as cm:
            AcsTranslationComponent().get_detections_from_video(job)
        self.assertEqual(mpf.DetectionError.NETWORK_ERROR, cm.exception.error_code)


    def test_reports_error_when_missing_acs_props(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))

        test_props = get_test_properties()
        del test_props['ACS_URL']
        job = mpf.ImageJob('Test', 'test.jpg', test_props, {}, ff_loc)
        with self.assertRaises(mpf.DetectionException) as cm:
            AcsTranslationComponent().get_detections_from_image(job)
        self.assertEqual(mpf.DetectionError.MISSING_PROPERTY, cm.exception.error_code)

        test_props = get_test_properties()
        del test_props['ACS_SUBSCRIPTION_KEY']
        job = mpf.ImageJob('Test', 'test.jpg', test_props, {}, ff_loc)
        with self.assertRaises(mpf.DetectionException) as cm:
            AcsTranslationComponent().get_detections_from_image(job)
        self.assertEqual(mpf.DetectionError.MISSING_PROPERTY, cm.exception.error_code)


    def test_missing_required_properties(self):
        for required_prop in ('ACS_URL', 'ACS_SUBSCRIPTION_KEY'):
            test_props = get_test_properties()
            del test_props[required_prop]

            ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))
            job = mpf.ImageJob('Test', 'test.jpg', test_props, {}, ff_loc)

            with self.assertRaises(mpf.DetectionException) as cm:
                AcsTranslationComponent().get_detections_from_image(job)
            self.assertEqual(mpf.DetectionError.MISSING_PROPERTY, cm.exception.error_code)


    def test_translate_multiple_properties(self):
        self.set_results_file('spanish-detect-result.json')
        self.set_results_file('results-spanish.json')

        ff_loc = mpf.ImageLocation(
            0, 0, 10, 10, -1,
            dict(TEXT=CHINESE_SAMPLE_TEXT, MORE_TEXT=SPANISH_SAMPLE_TEXT))
        job = mpf.ImageJob(
            'Test', 'test.jpg',
            get_test_properties(FEED_FORWARD_PROP_TO_PROCESS='MORE_TEXT,TEXT'),
            {}, ff_loc)

        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))
        result = results[0]

        self.assertEqual(SPANISH_SAMPLE_TEXT, result.detection_properties['MORE_TEXT'])
        self.assertEqual(SPANISH_SAMPLE_TEXT_ENG_TRANSLATE,
                         result.detection_properties['TRANSLATION'])

        request_body1 = self.get_request_body()
        self.assertEqual(1, len(request_body1))
        self.assertEqual(SPANISH_SAMPLE_TEXT, request_body1[0]['Text'])

        request_body2 = self.get_request_body()
        self.assertEqual(1, len(request_body2))
        self.assertEqual(SPANISH_SAMPLE_TEXT, request_body2[0]['Text'])

        with self.assertRaises(queue.Empty, msg='Only two requests should have been sent.'):
            self.get_request_body()


    def test_translation_cache(self):
        self.set_results_file('chinese-detect-result.json')
        self.set_results_file('results-chinese.json')
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=CHINESE_SAMPLE_TEXT)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=CHINESE_SAMPLE_TEXT))
            },
            dict(TEXT=CHINESE_SAMPLE_TEXT))

        job = mpf.VideoJob('test', 'test.jpg', 0, 1, get_test_properties(), {}, ff_track)

        results = list(AcsTranslationComponent().get_detections_from_video(job))
        self.assertEqual(1, len(results))
        result = results[0]

        self.assertEqual(CHINESE_SAMPLE_TEXT, result.detection_properties['TEXT'])
        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                         result.detection_properties['TRANSLATION'])
        self.assertEqual('EN', result.detection_properties['TRANSLATION TO LANGUAGE'])

        detection1 = result.frame_locations[0]
        self.assertEqual(CHINESE_SAMPLE_TEXT, detection1.detection_properties['TEXT'])
        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                         detection1.detection_properties['TRANSLATION'])
        self.assertEqual('EN', detection1.detection_properties['TRANSLATION TO LANGUAGE'])

        detection2 = result.frame_locations[1]
        self.assertEqual(CHINESE_SAMPLE_TEXT, detection2.detection_properties['TRANSCRIPT'])
        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                         detection2.detection_properties['TRANSLATION'])
        self.assertEqual('EN', detection2.detection_properties['TRANSLATION TO LANGUAGE'])

        request_body = self.get_request_body()
        self.assertEqual(1, len(request_body))
        self.assertEqual(CHINESE_SAMPLE_TEXT, request_body[0]['Text'])

        request_body = self.get_request_body()
        self.assertEqual(1, len(request_body))
        self.assertEqual(CHINESE_SAMPLE_TEXT, request_body[0]['Text'])

        with self.assertRaises(queue.Empty, msg='Only two requests should have been sent.'):
            self.get_request_body()


    def test_different_to_language(self):
        self.set_results_file('eng-detect-result.json')
        self.set_results_file('results-eng-to-russian.json')
        eng_text = 'Do you speak English?'
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=eng_text))
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(TO_LANGUAGE='ru'), {}, ff_loc)

        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))

        result = results[0]
        self.assertEqual(eng_text, result.detection_properties['TEXT'])
        self.assertEqual('Вы говорите по-англиский?',
                         result.detection_properties['TRANSLATION'])
        self.assertEqual('RU', result.detection_properties['TRANSLATION TO LANGUAGE'])

        detect_request_body = self.get_request_body()
        self.assertEqual(1, len(detect_request_body))
        self.assertEqual(eng_text, detect_request_body[0]['Text'])

        translate_request_body = self.get_request_body()
        self.assertEqual(1, len(translate_request_body))
        self.assertEqual(eng_text, translate_request_body[0]['Text'])


    def test_url_formation(self):

        def assert_expected_url(job_properties, expected_to, expected_from, expected_query):
            builder = AcsTranslateUrlBuilder(job_properties)
            url_parts = urllib.parse.urlparse(builder.url)
            self.assertEqual('/test/translate', url_parts.path)
            self.assertEqual(expected_to, builder.to_language)
            self.assertEqual(expected_from, builder.from_language)

            expected_query.setdefault('api-version', '3.0')
            expected_query.setdefault('to', expected_to)
            if expected_from:
                expected_query.setdefault('from', expected_from)

            query_dict = urllib.parse.parse_qs(url_parts.query)
            self.assertEqual(len(expected_query), len(query_dict))

            for key, val in expected_query.items():
                self.assertEqual([val], query_dict[key], f'The "{key}" key is different.')


        with self.subTest('Adds default query string'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test'), 'en', None,
                {})


        with self.subTest('Does not replace api-version'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?api-version=4.5'), 'en', None,
                {'api-version': '4.5'})

        with self.subTest('Copies to lang from job properties'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=asdf', TO_LANGUAGE='zh'),
                'zh', None, {})


        with self.subTest('Uses "to" from query string when no job property'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=ru'),
                'ru', None, {})


        with self.subTest('Copies from lang from job properties'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=asdf', TO_LANGUAGE='zh',
                     FROM_LANGUAGE='ru'),
                'zh', 'ru', {'from': 'ru'})

        with self.subTest('Uses "from" from query string when no job property'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=asdf&from=ru', TO_LANGUAGE='zh'),
                'zh', 'ru', {'from': 'ru'})

        with self.subTest('FROM_LANGUAGE has no default value'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=asdf', TO_LANGUAGE='es'),
                'es', None, {})

        with self.subTest('Copies SUGGESTED_FROM_LANGUAGE from job properties'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=asdf', TO_LANGUAGE='zh',
                     FROM_LANGUAGE='ru', SUGGESTED_FROM_LANGUAGE='ja'),
                'zh', 'ru', {'suggestedFrom': 'ja'})

        with self.subTest('Uses "suggestedFrom" from query string when no job property'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?suggestedFrom=ru'),
                'en', None, {'suggestedFrom': 'ru'})


        with self.subTest('Copies CATEGORY from job properties'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?to=asdf', TO_LANGUAGE='zh',
                     FROM_LANGUAGE='ru', SUGGESTED_FROM_LANGUAGE='ja', CATEGORY='My category'),
                'zh', 'ru', {'suggestedFrom': 'ja', 'category': 'My category'})

        with self.subTest('Uses "category" from query string when no job property'):
            assert_expected_url(
                dict(ACS_URL='http://example.com/test?suggestedFrom=ru&category=whatever'),
                'en', None, {'suggestedFrom': 'ru', 'category': 'whatever'})


    @mock.patch.object(TranslationClient, 'DETECT_MAX_CHARS', new_callable=lambda: 150)
    def test_split_wtp_known_language(self, _):
        self.set_results_file('traditional-chinese-detect-result.json')
        self.set_results_file('split-sentence/art-of-war-translation-1.json')
        self.set_results_file('split-sentence/art-of-war-translation-2.json')
        self.set_results_file('split-sentence/art-of-war-translation-3.json')
        self.set_results_file('split-sentence/art-of-war-translation-4.json')

        text = (TEST_DATA / 'split-sentence/art-of-war.txt').read_text()
        detection_props = dict(TEXT=text)
        TranslationClient(get_test_properties(), self.wtp_model).add_translations(detection_props)

        self.assertEqual(5, len(detection_props))
        self.assertEqual(text, detection_props['TEXT'])

        expected_translation = (TEST_DATA / 'split-sentence/art-war-translation.txt') \
            .read_text().strip()
        self.assertEqual(expected_translation, detection_props['TRANSLATION'])
        self.assertEqual('EN', detection_props['TRANSLATION TO LANGUAGE'])

        self.assertEqual('zh-Hant', detection_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(1.0,
            float(detection_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))

        detect_request_text = self.get_request_body()[0]['Text']
        self.assertEqual(text[0:TranslationClient.DETECT_MAX_CHARS], detect_request_text)

        expected_chunk_lengths = [86, 116, 104, 114]
        self.assertEqual(sum(expected_chunk_lengths), len(text.replace('\n','')))
        translation_request1 = self.get_request_body()[0]['Text']
        self.assertTrue(translation_request1.startswith('兵者，'))
        self.assertTrue(translation_request1.endswith('而不危也；'))
        self.assertEqual(expected_chunk_lengths[0], len(translation_request1))
        self.assertNotIn('\n', translation_request1,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request1,
                            'Spaces should not be added to Chinese text.')


        translation_request2 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[1], len(translation_request2))
        self.assertTrue(translation_request2.startswith('天者，陰陽'))
        self.assertTrue(translation_request2.endswith('兵眾孰強？'))
        self.assertNotIn('\n', translation_request1,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request1,
                            'Spaces should not be added to Chinese text.')


        translation_request3 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[2], len(translation_request3))
        self.assertTrue(translation_request3.startswith('士卒孰練？'))
        self.assertTrue(translation_request3.endswith('遠而示之近。'))
        self.assertNotIn('\n', translation_request3,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request3,
                            'Spaces should not be added to Chinese text.')


        translation_request4 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[3], len(translation_request4))
        self.assertTrue(translation_request4.startswith('利而誘之，'))
        self.assertTrue(translation_request4.endswith('勝負見矣。'))
        self.assertNotIn('\n', translation_request4,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request4,
                            'Spaces should not be added to Chinese text.')


    def test_split_engine_difference(self):
        # Note: we can only use the WtP models for subsequent tests
        # involving Chinese text because only WtP's multilingual models
        # can detect some of '。' characters used for this language.
        text = (TEST_DATA / 'split-sentence/art-of-war.txt').read_text()

        text_without_newlines = text.replace('\n', '')

        actual = self.wtp_model._split_wtp(text_without_newlines)
        self.assertEqual(3, len(actual))
        for line in actual:
            self.assertTrue(line.endswith('。'))

        actual = self.spacy_model._split_spacy(text_without_newlines)
        self.assertEqual(1, len(actual))

        # However, WtP prefers newlines over the '。' character.
        actual = self.wtp_model._split_wtp(text)
        self.assertEqual(10, len(actual))

    @mock.patch.object(TranslationClient, 'DETECT_MAX_CHARS', new_callable=lambda: 150)
    def test_split_wtp_advanced_known_language(self, _):
        # This test should only be run manually outside of a Docker build.
        # The WtP canine model is ~1 GB and not worth downloading and adding to the pre-built Docker image.
        if not LOCAL_TEST_WTP_MODEL:
            return

        # For this test, we're more interested in the changes in behavior
        # caused by WtP split. So the translation files are mainly placeholders.
        self.set_results_file('traditional-chinese-detect-result.json')
        self.set_results_file('split-sentence/art-of-war-translation-1.json')
        self.set_results_file('split-sentence/art-of-war-translation-2.json')
        self.set_results_file('split-sentence/art-of-war-translation-3.json')
        self.set_results_file('split-sentence/art-of-war-translation-4.json')

        text = (TEST_DATA / 'split-sentence/art-of-war.txt').read_text()
        detection_props = dict(TEXT=text)
        TranslationClient(get_test_properties(SENTENCE_MODEL="wtp-canine-s-1l"), self.wtp_adv_model).add_translations(detection_props)

        self.assertEqual(5, len(detection_props))
        self.assertEqual(text, detection_props['TEXT'])

        expected_translation = (TEST_DATA / 'split-sentence/art-war-translation.txt') \
            .read_text().strip()
        self.assertEqual(expected_translation, detection_props['TRANSLATION'])
        self.assertEqual('EN', detection_props['TRANSLATION TO LANGUAGE'])
        self.assertEqual('zh-Hant', detection_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(1.0,
            float(detection_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))

        detect_request_text = self.get_request_body()[0]['Text']
        self.assertEqual(text[0:TranslationClient.DETECT_MAX_CHARS], detect_request_text)

        # Main test starts here:
        expected_chunk_lengths = [61, 150, 61, 148]
        self.assertEqual(sum(expected_chunk_lengths), len(text.replace('\n','')))
        translation_request1 = self.get_request_body()[0]['Text']
        self.assertTrue(translation_request1.startswith('兵者，'))
        self.assertTrue(translation_request1.endswith('四曰將，五曰法。'))
        self.assertEqual(expected_chunk_lengths[0], len(translation_request1))
        self.assertNotIn('\n', translation_request1,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request1,
                            'Spaces should not be added to Chinese text.')

        translation_request2 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[1], len(translation_request2))
        self.assertTrue(translation_request2.startswith('道者，令民於上同意'))
        self.assertTrue(translation_request2.endswith('賞罰孰明'))
        self.assertNotIn('\n', translation_request2,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request2,
                            'Spaces should not be added to Chinese text.')

        translation_request3 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[2], len(translation_request3))
        self.assertTrue(translation_request3.startswith('？吾以此知勝'))
        self.assertTrue(translation_request3.endswith('因利而制權也。'))
        self.assertNotIn('\n', translation_request3,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request3,
                            'Spaces should not be added to Chinese text.')

        translation_request4 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[3], len(translation_request4))
        self.assertTrue(translation_request4.startswith('兵者，詭道也。'))
        self.assertTrue(translation_request4.endswith('勝負見矣。'))
        self.assertNotIn('\n', translation_request4,
                            'Newlines were not properly removed')
        self.assertNotIn(' ', translation_request4,
                            'Spaces should not be added to Chinese text.')

    @mock.patch.object(TranslationClient, 'DETECT_MAX_CHARS', new_callable=lambda: 150)
    def test_split_wtp_unknown_lang(self, _):
        # Check that the text splitter does not have an issue
        # processing an unknown detected language.
        self.set_results_file('invalid-lang-detect-result.json')
        self.set_results_file('split-sentence/art-of-war-translation-1.json')
        self.set_results_file('split-sentence/art-of-war-translation-2.json')
        self.set_results_file('split-sentence/art-of-war-translation-3.json')
        self.set_results_file('split-sentence/art-of-war-translation-4.json')

        text = (TEST_DATA / 'split-sentence/art-of-war.txt').read_text()
        detection_props = dict(TEXT=text)
        TranslationClient(get_test_properties(), self.wtp_model).add_translations(detection_props)

        self.assertEqual(5, len(detection_props))
        self.assertEqual(text, detection_props['TEXT'])

        expected_translation = (TEST_DATA / 'split-sentence/art-war-translation.txt') \
            .read_text().strip()
        self.assertEqual(expected_translation, detection_props['TRANSLATION'])
        self.assertEqual('EN', detection_props['TRANSLATION TO LANGUAGE'])

        self.assertEqual('fake-lang', detection_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(1.0,
            float(detection_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))

        detect_request_text = self.get_request_body()[0]['Text']
        self.assertEqual(text[0:TranslationClient.DETECT_MAX_CHARS], detect_request_text)

        expected_chunk_lengths = [88, 118, 116, 106]
        self.assertEqual(sum(expected_chunk_lengths), len(text))

        # Due to an incorrect language detection, newlines are
        # not properly replaced for Chinese text, and
        # additional whitespace is present in the text.
        # This alters the behavior of WtP sentence splitting.
        translation_request1 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[0], len(translation_request1))
        self.assertTrue(translation_request1.startswith('兵者，'))
        self.assertTrue(translation_request1.endswith('而不危也；'))
        self.assertNotIn('\n', translation_request1,
                         'Newlines were not properly removed')
        self.assertIn(' ', translation_request1,
                      'Spaces should be kept due to incorrect language detection.')

        translation_request2 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[1], len(translation_request2))
        self.assertTrue(translation_request2.startswith('天者，陰陽'))
        self.assertTrue(translation_request2.endswith('兵眾孰強？'))
        self.assertNotIn('\n', translation_request2,
                         'Newlines were not properly removed')
        self.assertIn(' ', translation_request2,
                      'Spaces should be kept due to incorrect language detection.')

        translation_request3 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[2], len(translation_request3))
        self.assertTrue(translation_request3.startswith('士卒孰練？'))
        self.assertTrue(translation_request3.endswith('亂而取之， '))
        self.assertNotIn('\n', translation_request3,
                         'Newlines were not properly removed')
        self.assertIn(' ', translation_request3,
                      'Spaces should be kept due to incorrect language detection.')

        translation_request4 = self.get_request_body()[0]['Text']
        self.assertEqual(expected_chunk_lengths[3], len(translation_request4))
        self.assertTrue(translation_request4.startswith('實而備之，'))
        self.assertTrue(translation_request4.endswith('勝負見矣。 '))
        self.assertNotIn('\n', translation_request4,
                         'Newlines were not properly removed')
        self.assertIn(' ', translation_request4,
                      'Spaces should be kept due to incorrect language detection.')


    def test_newline_removal(self):

        def replace(text):
            # Just use a random result file. We are only interested in the request content.
            self.set_results_file('results-chinese.json')

            props = get_test_properties(DETECT_BEFORE_TRANSLATE='FALSE')
            TranslationClient(props, self.wtp_model).add_translations(dict(TEXT=text))
            return self.get_request_body()[0]['Text']

        with self.subTest('English'):
            self.assertEqual('My name is John.', replace('My name is John.'))

            self.assertEqual('My name is John.', replace('My name\nis John.'))
            self.assertEqual('My name is John.', replace('My name \nis John.'))
            self.assertEqual('My name is John.', replace('My name\n is John.'))
            self.assertEqual('My name  is John.', replace('My name \n is John.'))

            self.assertEqual('My name is John.\n\nUnrelated text.',
                             replace('My name is John.\n\nUnrelated text.'))

            self.assertEqual('My name is John.\n\nUnrelated text.',
                             replace('My name\nis John.\n\nUnrelated text.'))

        # Make sure we don't include Korean in languages without spaces, because its code points
        # are near Chinese and Japanese.
        with self.subTest('Korean'):
            self.assertEqual('제 이름은 존이에요.', replace('제 이름은 존이에요.'))
            self.assertEqual('제 이름은 존이에요.', replace('제 이름은\n존이에요.'))

        with self.subTest('Chinese'):
            self.assertEqual('我叫约翰。', replace('我叫约翰。'))

            self.assertEqual('我叫约翰。', replace('我叫\n约翰。'))
            self.assertEqual('我叫 约翰。', replace('我叫 \n约翰。'))
            self.assertEqual('我叫 约翰。', replace('我叫\n 约翰。'))
            self.assertEqual('我叫  约翰。', replace('我叫 \n 约翰。'))

            self.assertEqual('我叫约翰。\n\n不相关的文字。', replace('我叫约翰。\n\n不相关的文字。'))

            self.assertEqual('我叫约翰。\n\n不相关的文字。', replace('我\n叫约翰。\n\n不相关的文字。'))

        # Make sure we include Japanese in languages without spaces.
        with self.subTest('Japanese - Hiragana'):
            self.assertEqual('の名前はジョンです', replace('の名前はジョンです'))
            self.assertEqual('の名前はジョンです', replace('の名前\nはジョンです'))

        with self.subTest('Japanese - Katakana'):
            self.assertEqual('ジの名前はョンです', replace('ジの名前はョンです'))
            self.assertEqual('ジの名前はョンです', replace('ジの名前\nはョンです'))


    def test_job_prop_newline_behavior(self):
        with self.subTest('default'):
            behavior = NewLineBehavior.get({})
            self.assertEqual('My name is John.', behavior('My name is John.', None))
            self.assertEqual('My name is John.', behavior('My name\nis John.', None))

            self.assertEqual('我叫约翰。', behavior('我叫约翰。', None))
            self.assertEqual('我叫约翰。', behavior('我叫\n约翰。', None))

        with self.subTest('GUESS'):
            behavior = NewLineBehavior.get(dict(STRIP_NEW_LINE_BEHAVIOR='GUESS'))
            self.assertEqual('My name is John.', behavior('My name is John.', None))
            self.assertEqual('My name is John.', behavior('My name\nis John.', None))

            self.assertEqual('我叫约翰。', behavior('我叫约翰。', None))
            self.assertEqual('我叫约翰。', behavior('我叫\n约翰。', None))

        with self.subTest('REMOVE'):
            behavior = NewLineBehavior.get(dict(STRIP_NEW_LINE_BEHAVIOR='REMOVE'))
            self.assertEqual('My name is John.', behavior('My name is John.', None))
            self.assertEqual('My nameis John.', behavior('My name\nis John.', None))

            self.assertEqual('我叫约翰。', behavior('我叫约翰。', None))
            self.assertEqual('我叫约翰。', behavior('我叫\n约翰。', None))

        with self.subTest('SPACE'):
            behavior = NewLineBehavior.get(dict(STRIP_NEW_LINE_BEHAVIOR='SPACE'))
            self.assertEqual('My name is John.', behavior('My name is John.', None))
            self.assertEqual('My name is John.', behavior('My name\nis John.', None))

            self.assertEqual('我叫约翰。', behavior('我叫约翰。', None))
            self.assertEqual('我叫 约翰。', behavior('我叫\n约翰。', None))

        with self.subTest('NONE'):
            behavior = NewLineBehavior.get(dict(STRIP_NEW_LINE_BEHAVIOR='NONE'))
            self.assertEqual('My name is John.', behavior('My name is John.', None))
            self.assertEqual('My name\nis John.', behavior('My name\nis John.', None))

            self.assertEqual('我叫约翰。', behavior('我叫约翰。', None))
            self.assertEqual('我叫\n约翰。', behavior('我叫\n约翰。', None))

        with self.subTest('INVALID'):
            with self.assertRaises(mpf.DetectionException) as cm:
                NewLineBehavior.get(dict(STRIP_NEW_LINE_BEHAVIOR='INVALID'))
            self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)


    def test_newline_behavior_with_from_lang(self):
        behavior = NewLineBehavior.get({})
        self.assertEqual('我叫约翰。', behavior('我叫\n约翰。', 'zh-Hans'))
        self.assertEqual('My nameis John.', behavior('My name\nis John.', 'zh-Hans'))

        behavior = NewLineBehavior.get({})
        self.assertEqual('我叫 约翰。', behavior('我叫\n约翰。', 'en'))
        self.assertEqual('My name is John.', behavior('My name\nis John.', 'en'))


    def test_job_prop_overrides_from_lang(self):
        behavior = NewLineBehavior.get(dict(STRIP_NEW_LINE_BEHAVIOR='SPACE'))
        self.assertEqual('我叫 约翰。', behavior('我叫\n约翰。', 'zh-Hans'))
        self.assertEqual('My name is John.', behavior('My name\nis John.', 'zh-Hans'))


    def test_chinese_japanese_char_detection(self):
        art_of_war_text = (TEST_DATA / 'split-sentence/art-of-war.txt').read_text()
        self.assertTrue(all(ChineseAndJapaneseCodePoints.check_char(ch)
                            for ch in art_of_war_text if not ch.isspace()))


        self.assertFalse(ChineseAndJapaneseCodePoints.check_char('a'))
        self.assertFalse(ChineseAndJapaneseCodePoints.check_char('한'))

        self.assertTrue(ChineseAndJapaneseCodePoints.check_char('》'))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char('。'))

        self.assertTrue(ChineseAndJapaneseCodePoints.check_char('说'))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char('說'))

        self.assertFalse(ChineseAndJapaneseCodePoints.check_char(chr(0x2e80 - 1)))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x2e80)))

        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x16fe0)))
        self.assertFalse(ChineseAndJapaneseCodePoints.check_char(chr(0x16fe0 - 1)))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x16fe0 + 1)))

        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x30000)))
        self.assertFalse(ChineseAndJapaneseCodePoints.check_char(chr(0x30000 - 1)))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x30000 + 1)))

        self.assertFalse(ChineseAndJapaneseCodePoints.check_char(chr(0x3134b)))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x3134b - 1)))
        self.assertTrue(ChineseAndJapaneseCodePoints.check_char(chr(0x3134b - 2)))
        self.assertFalse(ChineseAndJapaneseCodePoints.check_char(chr(0x3134b + 1)))
        self.assertFalse(ChineseAndJapaneseCodePoints.check_char(chr(0x3134b + 2)))


    def test_category_and_explicit_from_language(self):
        self.set_results_file('explicit-from-results.json')

        ff_loc = mpf.ImageLocation(0, 0, 10, 20, -1, dict(TEXT=CHINESE_SAMPLE_TEXT))
        job = mpf.ImageJob(
            'Test', 'test.jpg',
            get_test_properties(FROM_LANGUAGE='zh-Hans', CATEGORY='My category'),
            {}, ff_loc)

        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))
        result = results[0]

        self.assertEqual(5, len(result.detection_properties))
        self.assertEqual(CHINESE_SAMPLE_TEXT, result.detection_properties['TEXT'])
        self.assertEqual(CHINESE_SAMPLE_TEXT_ENG_TRANSLATE,
                         result.detection_properties['TRANSLATION'])
        self.assertEqual('EN', result.detection_properties['TRANSLATION TO LANGUAGE'])
        self.assertEqual('zh-Hans', result.detection_properties['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(
            1.0,
            float(result.detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))

        request_url, request_body = self.get_request()
        self.assertEqual(1, len(request_body))
        self.assertEqual(CHINESE_SAMPLE_TEXT, request_body[0]['Text'])

        query_string: str = urllib.parse.urlparse(request_url).query
        query_dict = urllib.parse.parse_qs(query_string)
        self.assertEqual(4, len(query_dict))
        self.assertEqual(['zh-Hans'], query_dict['from'])
        self.assertEqual(['en'], query_dict['to'])
        self.assertEqual(['My category'], query_dict['category'])
        self.assertEqual(['3.0'], query_dict['api-version'])


    def test_suggested_from(self):
        self.set_results_file('suggested-from-result.json')

        input_text = '!@$%'

        ff_loc = mpf.ImageLocation(0, 0, 10, 20, -1, dict(TEXT=input_text))
        props = get_test_properties(SUGGESTED_FROM_LANGUAGE='ja', DETECT_BEFORE_TRANSLATE='false')
        job = mpf.ImageJob('Test', 'test.jpg', props, {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))
        result = results[0]

        self.assertEqual(input_text, result.detection_properties['TEXT'])
        self.assertEqual(input_text, result.detection_properties['TRANSLATION'])
        self.assertEqual('EN', result.detection_properties['TRANSLATION TO LANGUAGE'])

        self.assertEqual('ja', result.detection_properties['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(
            0.0, float(result.detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))

        request_url, request_body = self.get_request()
        self.assertEqual(1, len(request_body))
        self.assertEqual(input_text, request_body[0]['Text'])

        query_string = urllib.parse.urlparse(request_url).query
        query_dict = urllib.parse.parse_qs(query_string)
        self.assertEqual(['ja'], query_dict['suggestedFrom'])
        self.assertEqual(['en'], query_dict['to'])


    def test_guess_split_simple_sentence(self):
        input_text = 'Hello, what is your name? My name is John.'
        actual = list(TextSplitter.split(input_text,
            28,
            28,
            get_azure_char_count,
            self.wtp_model))
        self.assertEqual(input_text, ''.join(actual))
        self.assertEqual(2, len(actual))

        # "Hello, what is your name?"
        self.assertEqual('Hello, what is your name? ', actual[0])
        # " My name is John."
        self.assertEqual('My name is John.', actual[1])

        input_text = 'Hello, what is your name? My name is John.'
        actual = list(TextSplitter.split(input_text,
            28,
            28,
            get_azure_char_count,
            self.spacy_model))
        self.assertEqual(input_text, ''.join(actual))
        self.assertEqual(2, len(actual))

        # "Hello, what is your name?"
        self.assertEqual('Hello, what is your name? ', actual[0])
        # " My name is John."
        self.assertEqual('My name is John.', actual[1])



    def test_split_sentence_end_punctuation(self):
        input_text = 'Hello. How are you? asdfasdf'
        actual = list(TextSplitter.split(input_text,
            20,
            10,
            get_azure_char_count,
            self.wtp_model))

        self.assertEqual(input_text, ''.join(actual))
        self.assertEqual(2, len(actual))

        self.assertEqual('Hello. How are you? ', actual[0])
        self.assertEqual('asdfasdf', actual[1])

        actual = list(TextSplitter.split(input_text,
            20,
            10,
            get_azure_char_count,
            self.spacy_model))

        self.assertEqual(input_text, ''.join(actual))
        self.assertEqual(2, len(actual))

        self.assertEqual('Hello. How are you? ', actual[0])
        self.assertEqual('asdfasdf', actual[1])


    def test_guess_split_edge_cases(self):
        input_text = ("This is a sentence (Dr.Test). Is this,"
                      " a sentence as well? Maybe...maybe not?"
                      " \n All done, I think!")

        # Split using WtP model.
        actual = list(TextSplitter.split(input_text,
            30,
            30,
            get_azure_char_count,
            self.wtp_model))

        self.assertEqual(input_text, ''.join(actual))
        self.assertEqual(4, len(actual))

        # WtP should detect and split out each sentence
        self.assertEqual("This is a sentence (Dr.Test). ", actual[0])
        self.assertEqual("Is this, a sentence as well? ", actual[1])
        self.assertEqual("Maybe...maybe not? \n ", actual[2])
        self.assertEqual("All done, I think!", actual[3])

        actual = list(TextSplitter.split(input_text,
            35,
            35,
            get_azure_char_count,
            self.spacy_model))
        self.assertEqual(input_text, ''.join(actual))
        self.assertEqual(4, len(actual))

        # Split using spaCy model.
        self.assertEqual("This is a sentence (Dr.Test). ", actual[0])
        self.assertEqual("Is this, a sentence as well? ", actual[1])
        self.assertEqual("Maybe...maybe not? \n ", actual[2])
        self.assertEqual("All done, I think!", actual[3])


    def test_no_translate_no_detect_when_language_ff_prop_matches(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT='Hello', DECODED_LANGUAGE='eng'))
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))

        result_props = results[0].detection_properties
        self.assertNotIn('TRANSLATION', result_props)
        self.assertEqual('EN', result_props['TRANSLATION TO LANGUAGE'])
        self.assertEqual('en', result_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertEqual('TRUE', result_props['SKIPPED TRANSLATION'])


    def test_language_ff_prop_different(self):
        self.set_results_file('results-spanish.json')
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1,
                                   dict(TEXT=SPANISH_SAMPLE_TEXT, DECODED_LANGUAGE='spa'))
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))

        result = results[0]
        print(result.detection_properties)

        self.assertEqual(SPANISH_SAMPLE_TEXT, result.detection_properties['TEXT'])
        self.assertEqual(SPANISH_SAMPLE_TEXT_ENG_TRANSLATE,
                         result.detection_properties['TRANSLATION'])
        self.assertEqual('es', result.detection_properties['TRANSLATION SOURCE LANGUAGE'])

        request_body = self.get_request_body()
        self.assertEqual(1, len(request_body))
        self.assertEqual(SPANISH_SAMPLE_TEXT, request_body[0]['Text'])


    def test_unsupported_ff_prop(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1,
                                   dict(TEXT='ංහල', DECODED_LANGUAGE='si'))
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))

        result_props = results[0].detection_properties
        print(result_props)
        self.assertNotIn('TRANSLATION', result_props)
        self.assertEqual('EN', result_props['TRANSLATION TO LANGUAGE'])
        self.assertEqual('si', result_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(1, float(result_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))
        self.assertEqual('TRUE', result_props['SKIPPED TRANSLATION'])
        self.assertEqual('si', result_props['MISSING_LANGUAGE_MODELS'])


    def test_does_not_translate_when_to_and_from_are_same(self):
        self.set_results_file('eng-detect-result.json')
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT='Hello'))
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))

        result_props = results[0].detection_properties
        self.assertNotIn('TRANSLATION', result_props)
        self.assertEqual('EN', result_props['TRANSLATION TO LANGUAGE'])
        self.assertEqual('en', result_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertAlmostEqual(0.95, float(result_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE']))
        self.assertEqual('TRUE', result_props['SKIPPED TRANSLATION'])

        request_body = self.get_request_body()
        self.assertEqual('Hello', request_body[0]['Text'])

        with self.assertRaises(queue.Empty,
                               msg='Only one request (for /detect) should have been sent. '
                                   'No /translate should be sent.'):
            self.get_request_body()


    def test_multiple_detected_languages(self):
        self.set_results_file('eng-chinese-detect-result.json')
        self.set_results_file('eng-chinese-translate-results.json')
        input_text = 'What is your name? 你好'
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=input_text))
        job = mpf.ImageJob('Test', 'test.jpg', get_test_properties(), {}, ff_loc)
        results = list(AcsTranslationComponent().get_detections_from_image(job))
        self.assertEqual(1, len(results))

        result_props = results[0].detection_properties
        self.assertEqual('What is your name? Hello', result_props['TRANSLATION'])
        self.assertEqual('EN', result_props['TRANSLATION TO LANGUAGE'])
        self.assertEqual('en; zh-Hans', result_props['TRANSLATION SOURCE LANGUAGE'])
        self.assertEqual('0.67; 0.33', result_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE'])

        detect_request_body = self.get_request_body()
        self.assertEqual(input_text, detect_request_body[0]['Text'])

        translate_url, translate_request_body = self.get_request()
        self.assertEqual(input_text, translate_request_body[0]['Text'])
        self.assertIn('from=zh-Hans', translate_url)



    def test_get_n_chars(self):
        self.assertEqual('123', get_n_azure_chars('12345', 0, 3))

        text = '😀' * 5 + '👍' * 5
        self.assertEqual('😀' * 5, get_n_azure_chars(text, 0, 10))
        self.assertEqual('👍' * 5, get_n_azure_chars(text, 5, 10))

        text = '😀😀12345'
        self.assertEqual('😀😀1', get_n_azure_chars(text, 0, 5))
        self.assertEqual('😀', get_n_azure_chars(text, 0, 2))
        self.assertEqual('😀', get_n_azure_chars(text, 0, 3))

        text = '😀😀😀😀😀12345'
        self.assertEqual('😀😀', get_n_azure_chars(text, 0, 5))
        self.assertEqual('😀😀😀', get_n_azure_chars(text, 0, 6))
        self.assertEqual('😀😀😀', get_n_azure_chars(text, 0, 7))

        self.assertEqual('😀😀', get_n_azure_chars(text, 2, 5))
        self.assertEqual('😀😀😀', get_n_azure_chars(text, 2, 6))
        self.assertEqual('😀😀😀1', get_n_azure_chars(text, 2, 7))
        self.assertEqual('😀😀😀12', get_n_azure_chars(text, 2, 8))
        self.assertEqual('😀😀😀123', get_n_azure_chars(text, 2, 9))
        self.assertEqual('😀😀😀1234', get_n_azure_chars(text, 2, 10))
        self.assertEqual('😀😀😀12345', get_n_azure_chars(text, 2, 11))
        self.assertEqual('😀😀😀12345', get_n_azure_chars(text, 2, 12))

        text = '12345😀😀😀😀😀'
        self.assertEqual('12345', get_n_azure_chars(text, 0, 5))
        self.assertEqual('12345', get_n_azure_chars(text, 0, 6))
        self.assertEqual('12345😀', get_n_azure_chars(text, 0, 7))
        self.assertEqual('12345😀', get_n_azure_chars(text, 0, 8))
        self.assertEqual('12345😀😀', get_n_azure_chars(text, 0, 9))
        self.assertEqual(text, get_n_azure_chars(text, 0, 15))

        self.assertEqual('345😀', get_n_azure_chars(text, 2, 5))
        self.assertEqual('345😀', get_n_azure_chars(text, 2, 6))
        self.assertEqual('345😀😀', get_n_azure_chars(text, 2, 7))
        self.assertEqual('345😀😀', get_n_azure_chars(text, 2, 8))
        self.assertEqual('345😀😀😀', get_n_azure_chars(text, 2, 9))
        self.assertEqual('345😀😀😀', get_n_azure_chars(text, 2, 10))
        self.assertEqual('345😀😀😀😀', get_n_azure_chars(text, 2, 11))
        self.assertEqual('345😀😀😀😀', get_n_azure_chars(text, 2, 12))

        self.assertEqual('', get_n_azure_chars('', 0, 100))
        self.assertEqual('', get_n_azure_chars('', 10, 100))
        self.assertEqual('', get_n_azure_chars('Hello, world!', 5, 0))
        self.assertEqual('', get_n_azure_chars('Hello, world!', 0, 0))
        self.assertEqual('', get_n_azure_chars('Hello, world!', 20, 0))
        self.assertEqual('', get_n_azure_chars('Hello, world!', 20, 20))

    def test_azure_char_count(self):
        self.assertEqual(5, get_azure_char_count('12345'))
        self.assertEqual(15, get_azure_char_count('😀😀😀😀😀12345'))
        self.assertEqual(20, get_azure_char_count('😀' * 5 + '👍' * 5))


def get_test_properties(**extra_properties):
    return {
        'ACS_URL': os.getenv('ACS_URL', 'http://localhost:10670/translator'),
        'ACS_SUBSCRIPTION_KEY': os.getenv('ACS_SUBSCRIPTION_KEY', 'test_key'),
        **extra_properties
    }


class AcsRequestEntry(TypedDict):
    Text: str

class Request(NamedTuple):
    url: str
    body: List[AcsRequestEntry]


class MockServer(http.server.HTTPServer):
    _results_path_queue: 'queue.Queue[pathlib.Path]'
    _request_queue: 'queue.Queue[Request]'

    def __init__(self):
        super().__init__(('', 10670), MockRequestHandler)
        self._results_path_queue = queue.Queue()
        self._request_queue = queue.Queue()
        threading.Thread(target=self.serve_forever).start()

    def set_results_path(self, results_path: pathlib.Path) -> None:
        self._results_path_queue.put_nowait(results_path)

    def get_results_path(self) -> pathlib.Path:
        return self._results_path_queue.get_nowait()


    def get_request_info(self) -> Request:
        return self._request_queue.get_nowait()

    def set_request_info(self, request_body: Request) -> None:
        self._request_queue.put(request_body)


    def drain_queues(self):
        results_path_queue_empty = self._results_path_queue.empty()
        request_queue_empty = self._request_queue.empty()
        if results_path_queue_empty and request_queue_empty:
            return

        while not self._results_path_queue.empty():
            self._results_path_queue.get_nowait()

        while not self._request_queue.empty():
            self._request_queue.get_nowait()

        if not results_path_queue_empty:
            raise Exception('Expected results path queue to be empty.')
        elif not request_queue_empty:
            raise Exception('Expected request body queue to be empty.')



class MockRequestHandler(http.server.BaseHTTPRequestHandler):
    server: MockServer

    error_content_type = 'application/json; charset=UTF-8'
    error_message_format = '%(message)s'

    def do_POST(self):
        url_parts = urllib.parse.urlparse(self.path)

        is_detect = url_parts.path == '/translator/detect'
        is_translate = url_parts.path == '/translator/translate'
        if not is_detect and not is_translate:
            self._send_error(404, 000, 'NOT FOUND')
            return

        self._validate_headers()
        self._validate_query_string(url_parts.query, is_translate)
        max_chars = TranslationClient.DETECT_MAX_CHARS
        self._validate_body(max_chars)

        self.send_response(200)
        self.end_headers()
        with self.server.get_results_path().open('rb') as f:
            shutil.copyfileobj(f, self.wfile)


    def _validate_headers(self) -> None:
        if self.headers['Ocp-Apim-Subscription-Key'] != 'test_key':
            self._send_error(
                401, 0,
                'Expected the Ocp-Apim-Subscription-Key header to contain the subscription key.')

        if self.headers['Content-Type'] != 'application/json; charset=UTF-8':
            self._send_error(415, 0,
                             'Expected content type to be: "application/json; charset=UTF-8"')

        trace_id = self.headers['X-ClientTraceId']
        if not trace_id:
            self._send_error(400, 43, 'X-ClientTraceId is missing')

        global SEEN_TRACE_IDS
        if trace_id in SEEN_TRACE_IDS:
            self._send_error(400, 43, 'Duplicate X-ClientTraceId')
        SEEN_TRACE_IDS.add(trace_id)


    def _validate_query_string(self, query_string, is_translate) -> None:
        query_dict = urllib.parse.parse_qs(query_string)
        if query_dict['api-version'][0] != '3.0':
            self._send_error(400, 21, 'The API version parameter is missing or invalid.')
        if is_translate:
            if query_dict['to'][0] not in ('en', 'ru'):
                self._send_error(400, 36, 'The target language ("To" field) is missing or invalid.')
            if from_lang := query_dict.get('from'):
                if from_lang[0] == 'si':
                    self._send_error(400, 35, 'The source language is not valid.')


    def _validate_body(self, max_chars) -> None:
        content_len = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(content_len))
        full_url = f'http://{self.server.server_name}:{self.server.server_port}{self.path}'
        self.server.set_request_info(Request(full_url, body))

        if len(body) == 0:
            self._send_error(430, 0, 'Request body was empty array.')

        for translation_request in body:
            request_text = translation_request['Text']
            if len(request_text) == 0:
                self._send_error(430, 0, 'Translation request did not contain text.')
            elif len(request_text) > max_chars:
                self._send_error(429, 0, 'The server rejected the request because the client has '
                                 'exceeded request limits.')


    def _send_error(self, http_status, subcode, message):
        error_body = {
            'error': {
                'code': http_status * 1000 + subcode,
                'message': message
            }
        }
        self.send_error(http_status, json.dumps(error_body))
        # Stop further processing of the HTTP request.
        raise Exception()


if __name__ == '__main__':
    unittest.main(verbosity=2)
