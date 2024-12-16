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
from __future__ import annotations

import logging
import pathlib
import sys
import unittest
from typing import Sequence, Union

import mpf_component_api as mpf

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from fast_text_language_detection_component import FastTextLanguageDetectionComponent

logging.basicConfig(level=logging.DEBUG)

TEST_DATA = pathlib.Path(__file__).parent / 'data'

ENGLISH_SAMPLE_TEXT = 'Where is the library?'
SPANISH_SAMPLE_TEXT = '¿Dónde está la biblioteca?'
CHINESE_SAMPLE_TEXT = '你好，你叫什么名字？'
RUSSIAN_SAMPLE_TEXT = 'Привет, как тебя зовут?'

class TestComponent(unittest.TestCase):

    def test_selected_content(self):
        job = mpf.GenericJob('test job', '', {}, {'SELECTED_CONTENT': SPANISH_SAMPLE_TEXT})
        results = FastTextLanguageDetectionComponent().get_detections_from_generic(job)
        self._assert_results_match('SPA', 'LATN', 1.0, results)


    def test_plain_text_file(self):
        job = mpf.GenericJob('test job', str(TEST_DATA / 'art-of-war.txt'), {}, {})
        results = FastTextLanguageDetectionComponent().get_detections_from_generic(job)
        self._assert_results_match('ZHO', 'HANT', 0.772, results)


    def test_generic_job(self):
        ff_track = mpf.GenericTrack(1, {'TEXT': CHINESE_SAMPLE_TEXT})
        job = mpf.GenericJob('test job', '', {}, {}, ff_track)
        results = FastTextLanguageDetectionComponent().get_detections_from_generic(job)
        self._assert_results_match('ZHO', 'HANS', 0.9999, results)


    def test_empty_text_job(self):
        ff_track = mpf.GenericTrack(1, {'TEXT': ' '})
        job = mpf.GenericJob('test job', '', {}, {}, ff_track)
        results = FastTextLanguageDetectionComponent().get_detections_from_generic(job)
        self._assert_results_match('<UNKNOWN>', '<UNKNOWN>', 0, results)


    def test_emoji_text_job(self):
        ff_track = mpf.GenericTrack(1, {'TEXT': '😑🤦'})
        job = mpf.GenericJob('test job', '', {}, {}, ff_track)
        results = FastTextLanguageDetectionComponent().get_detections_from_generic(job)
        self._assert_results_match('<UNKNOWN>', '<UNKNOWN>', 0, results)


    def test_image_job(self):
        ff_location = mpf.ImageLocation(1, 2, 3, 4, 0.5, {'TEXT': ENGLISH_SAMPLE_TEXT})
        job = mpf.ImageJob('test job', '', {}, {}, ff_location)
        results = FastTextLanguageDetectionComponent().get_detections_from_image(job)
        self._assert_results_match('ENG', 'LATN', 1.0, results)

    def test_audio_job(self):
        ff_track = mpf.AudioTrack(0, 1, 0.5, {'TEXT': RUSSIAN_SAMPLE_TEXT})
        job = mpf.AudioJob('test job', '', 0, 1, {}, {}, ff_track)
        results = FastTextLanguageDetectionComponent().get_detections_from_audio(job)
        self._assert_results_match('RUS', 'CYRL', 0.9999, results)


    def test_video_job(self):
        ff_loc1 = mpf.ImageLocation(1, 2, 3, 4, 0.5, {'TEXT': ENGLISH_SAMPLE_TEXT})
        ff_loc2 = mpf.ImageLocation(1, 2, 3, 4, 0.5, {'TEXT': CHINESE_SAMPLE_TEXT})
        ff_loc3 = mpf.ImageLocation(1, 2, 3, 4, 0.5, {'TEXT': SPANISH_SAMPLE_TEXT})
        ff_track = mpf.VideoTrack(
            1, 3, 0.5, {1: ff_loc1, 2: ff_loc2, 3: ff_loc3}, {'TEXT': RUSSIAN_SAMPLE_TEXT})
        job = mpf.VideoJob('test job', '', 1, 3, {}, {}, ff_track)
        results = FastTextLanguageDetectionComponent().get_detections_from_video(job)

        self._assert_results_match('RUS', 'CYRL', 0.9999, results)
        frame_locs = results[0].frame_locations
        self._assert_results_match('ENG', 'LATN', 1.0, (frame_locs[1],))
        self._assert_results_match('ZHO', 'HANS', 0.9999, (frame_locs[2],))
        self._assert_results_match('SPA', 'LATN', 1.0, (frame_locs[3],))


    def _assert_results_match(
            self,
            language: str,
            script: str,
            confidence: float,
            tracks: Sequence[Union[
                mpf.GenericTrack, mpf.AudioTrack, mpf.ImageLocation, mpf.VideoTrack]]):
        self.assertEqual(1, len(tracks))
        properties = tracks[0].detection_properties
        self.assertEqual(language, properties['ISO_LANGUAGE'])
        self.assertEqual(script, properties['ISO_SCRIPT'])
        actual_confidence = float(properties['PRIMARY_LANGUAGE_CONFIDENCE'])
        self.assertAlmostEqual(confidence, actual_confidence, places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
