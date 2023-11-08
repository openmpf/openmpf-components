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
CHINESE_SHORT_SAMPLE = "你好，你叫什么名字？"
SHORT_OUTPUT = "Where's the library?"

# Observation, improvement in Argos-Chinese translation.
SHORT_OUTPUT_CHINESE = "Hello. What's your name?"

LONG_OUTPUT = (
    "We hold as evident these truths: that all men are created equal, "
    "that they are endowed by their Creator with certain inalienable rights, "
    "which among them are life, liberty and the pursuit of happiness. "
    "That in order to nurture these rights, governments are instituted among men, "
    "which derive their legitimate powers from the consent of the governed. "
    "Whenever a form of government becomes destroyer of these principles, "
    "the people have the right to reform or abolish it and to institute a new government "
    "that is founded on those principles, and to organize their powers in the way that in "
    "their opinion will offer the greatest chance of achieving their security and happiness."
)

MED_OUTPUT = (
    "Considering that the recognition of the inherent dignity and equal and "
    "inalienable rights of all members of the human family is the foundation "
    "of freedom, justice and peace for all; and"
)

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

        self.assertEqual('ru', result[0].frame_locations[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual('es', result[0].frame_locations[1].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual('zh', result[0].frame_locations[2].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].frame_locations[0].detection_properties['TRANSLATION'])
        self.assertEqual(SHORT_OUTPUT, result[0].frame_locations[1].detection_properties['TRANSLATION'])
        self.assertEqual(SHORT_OUTPUT_CHINESE, result[0].frame_locations[2].detection_properties['TRANSLATION'])

    def test_large_text(self):
        comp = ArgosTranslationComponent()
        job = mpf.GenericJob(
            job_name='Test Sentence Length',
            data_uri=str(TEST_DATA / 'spanish_long.txt'),
            job_properties=dict(DEFAULT_SOURCE_LANGUAGE='ES'),
            media_properties={},
            feed_forward_track=None
        )

        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])

        # TODO: Identify why the 1.0 spanish model occasionally switches words.
        # In this case,  words for nuture and nullify are sometimes switched depending on build environment.
        self.assertEqual(LONG_OUTPUT.replace("nullify","nurture"), result[0].detection_properties['TRANSLATION'])

    def test_medium_text(self):
        comp = ArgosTranslationComponent()
        job = mpf.GenericJob(
            job_name='Test Russian',
            data_uri=str(TEST_DATA / 'russian_medium.txt'),
            job_properties=dict(DEFAULT_SOURCE_LANGUAGE='RUS'),
            media_properties={},
            feed_forward_track=None
        )

        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))
        self.assertEqual('ru', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        print("Translation")
        print(result[0].detection_properties['TRANSLATION'])
        self.assertEqual(MED_OUTPUT, result[0].detection_properties['TRANSLATION'])

    def test_no_feed_forward_location(self):
        comp = ArgosTranslationComponent()
        job = mpf.ImageJob('Test', 'test.jpg', {}, {})

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_image(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

    def test_no_feed_forward_track(self):
        comp = ArgosTranslationComponent()
        job = mpf.VideoJob('test', 'test.mp4', 0, 1, {}, {})

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_video(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

        job = mpf.AudioJob('Test Audio', 'test.wav', 0, 1, {}, {})
        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_audio(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

    def test_unsupported_language(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='FAKE'))
        job = mpf.ImageJob('Test Image', 'test.jpg', dict(DEFAULT_SOURCE_LANGUAGE='es'), {}, ff_loc)
        comp = ArgosTranslationComponent()

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_image(job))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)


        job = mpf.GenericJob('Test Plaintext', str(TEST_DATA / 'spanish_short.txt'),
                             dict(DEFAULT_SOURCE_LANGUAGE='FAKE'), {})
        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)


        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANG='ES'))
        job = mpf.ImageJob('Test Image', 'test.jpg', dict(DEFAULT_SOURCE_LANGUAGE='FAKE'), {}, ff_loc)
        comp = ArgosTranslationComponent()

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_image(job))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)



    def test_iso_map(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='SPA'))
        job = mpf.ImageJob('Test Image', 'test.jpg', {}, {}, ff_loc)
        comp = ArgosTranslationComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['TRANSLATION_SOURCE_LANGUAGE'])
        self.assertEqual(SHORT_OUTPUT, result[0].detection_properties['TRANSLATION'])

    def test_translation_cache(self):
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES')),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=SPANISH_SHORT_SAMPLE, LANGUAGE='SPA'))
            },
            dict(TEXT=SPANISH_SHORT_SAMPLE, LANGUAGE='ES'))

        job = mpf.VideoJob('test', 'test.jpg', 0, 1, {}, {}, ff_track)

        comp = ArgosTranslationComponent()
        results = comp.get_detections_from_video(job)

        self.assertEqual(1, len(results))
        result = results[0]

        self.assertEqual(SPANISH_SHORT_SAMPLE, result.detection_properties['TEXT'])
        self.assertEqual(SHORT_OUTPUT, result.detection_properties['TRANSLATION'])
        self.assertEqual('es', result.detection_properties['TRANSLATION_SOURCE_LANGUAGE'])

        detection1 = result.frame_locations[0]
        self.assertEqual(SPANISH_SHORT_SAMPLE, detection1.detection_properties['TEXT'])
        self.assertEqual(SHORT_OUTPUT, detection1.detection_properties['TRANSLATION'])

        detection2 = result.frame_locations[1]
        self.assertEqual(SPANISH_SHORT_SAMPLE, detection2.detection_properties['TRANSCRIPT'])
        self.assertEqual(SHORT_OUTPUT, detection2.detection_properties['TRANSLATION'])

    def test_no_feed_forward_prop_no_default_lang(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SPANISH_SHORT_SAMPLE, LANG='ES'))
        job = mpf.ImageJob('Test Image', 'test.jpg', {}, {}, ff_loc)
        comp = ArgosTranslationComponent()

        with self.assertRaises(mpf.DetectionException) as cm:
            comp.get_detections_from_image(job)
        self.assertEqual(mpf.DetectionError.MISSING_PROPERTY, cm.exception.error_code)




if __name__ == '__main__':
    unittest.main()
