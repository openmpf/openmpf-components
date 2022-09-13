#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
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

from pathlib import Path
import sys
import unittest

import mpf_component_api as mpf

from argos_translation_component import ArgosTranslationComponent

LOCAL_PATH = Path(__file__).parent
sys.path.insert(0, str(LOCAL_PATH.parent))
TEST_DATA = LOCAL_PATH / 'data'

SPANISH_SHORT_SAMPLE = '¿Dónde está la biblioteca?'
RUSSIAN_SHORT_SAMPLE = "Где библиотека?"
CHINESE_SHORT_SAMPLE = "谢谢。"
SHORT_OUTPUT = "Where's the library?"
SHORT_OUTPUT_CHINESE = "Thanks."


class TestArgosTranslation(unittest.TestCase):

    def test_generic_job(self):
        ff_track = mpf.GenericTrack(-1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', dict(DEFAULT_SOURCE_LANGUAGE='ZH'), {}, ff_track)
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].detection_properties['TRANSLATION'])

    def test_plaintext_job(self):
        job = mpf.GenericJob('Test Plaintext', str(TEST_DATA / 'spanish_short.txt'),
                             dict(DEFAULT_SOURCE_LANGUAGE='ES'), {})
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].detection_properties['TRANSLATION'])

    def test_audio_job(self):
        ff_track = mpf.AudioTrack(0, 1, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES'))
        job = mpf.AudioJob('Test Audio', 'test.wav', 0, 1, dict(DEFAULT_SOURCE_LANGUAGE='ZH'), {}, ff_track)
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].detection_properties['TRANSLATION'])

    def test_image_job(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES'))
        job = mpf.ImageJob('Test Image', 'test.jpg', dict(DEFAULT_SOURCE_LANGUAGE='ZH'), {}, ff_loc)
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].detection_properties['TRANSLATION'])

    def test_video_job(self):
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES')),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TEXT=RUSSIAN_SHORT_SAMPLE, LANGUAGE='RU'))
            },
            dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES'))
        job = mpf.VideoJob('Test Video', 'test.mp4', 0, 1, dict(DEFAULT_SOURCE_LANGUAGE=''), {}, ff_track)
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_video(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual('es', result[0].frame_locations[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual('ru', result[0].frame_locations[1].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].detection_properties['TRANSLATION'])
        self.assertEqual(SHORT_OUTPUT, result[0].frame_locations[0].detection_properties['TRANSLATION'])
        self.assertEqual(SHORT_OUTPUT, result[0].frame_locations[1].detection_properties['TRANSLATION'])

    def test_language_behavior(self):
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TEXT=RUSSIAN_SHORT_SAMPLE, LANGUAGE='RU')),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES')),
                2: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TEXT=CHINESE_SHORT_SAMPLE, LANGUAGE='ZH')),
                3: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_OUTPUT, LANGUAGE='EN'))
            },
            dict(LANGUAGE='ES'))
        job = mpf.VideoJob('Test Language', 'test.mp4', 0, 1, {}, {}, ff_track)
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_video(job)

        self.assertEqual(1, len(result))

        # Should skip English tracks
        self.assertEqual('TRUE', result[0].frame_locations[3].detection_properties['SKIPPED_TRANSLATION'])

        # Should automatically select the correct language for other tracks
        self.assertEqual('ru', result[0].frame_locations[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual('es', result[0].frame_locations[1].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual('zh', result[0].frame_locations[2].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].frame_locations[0].detection_properties['TRANSLATION'])
        self.assertEqual(SHORT_OUTPUT, result[0].frame_locations[1].detection_properties['TRANSLATION'])
        self.assertEqual(SHORT_OUTPUT_CHINESE, result[0].frame_locations[2].detection_properties['TRANSLATION'])






