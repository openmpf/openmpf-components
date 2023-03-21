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

import unittest
import os

import mpf_component_api as mpf
import warnings
from whisper_detection_component import WhisperDetectionComponent


class TestWhisperDetection(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ResourceWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(
            os.path.dirname(__file__),
            'data',
            filename
        )

    def test_audio_job(self):
        job = mpf.AudioJob('test', self._get_test_file('left.wav'), 0, -1, {}, {})
        comp = WhisperDetectionComponent()
        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))
        self.assertEqual('en', result[0].detection_properties['DETECTED_LANGUAGE'])
        self.assertEqual('eng', result[0].detection_properties['ISO_LANGUAGE'])

    def test_video_job(self):
        media_properties = dict(FPS='24')

        job = mpf.VideoJob('test', self._get_test_file('left.avi'), 0, -1, {}, media_properties)
        comp = WhisperDetectionComponent()
        result = comp.get_detections_from_video(job)

        self.assertEqual(1, len(result))
        self.assertEqual('en', result[0].detection_properties['DETECTED_LANGUAGE'])
        self.assertEqual('eng', result[0].detection_properties['ISO_LANGUAGE'])

    def test_load_different_models_lang_detection(self):
        job = mpf.AudioJob('test', self._get_test_file('left.wav'), 0, -1, {}, {})
        comp = WhisperDetectionComponent()

        self.assertFalse(comp.wrapper.initialized)

        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))
        self.assertEqual('en', result[0].detection_properties['DETECTED_LANGUAGE'])
        self.assertEqual('eng', result[0].detection_properties['ISO_LANGUAGE'])

        self.assertEqual("multi", comp.wrapper.lang)
        self.assertEqual("base", comp.wrapper.size)
        self.assertTrue(comp.wrapper.initialized)

        job_props = dict(WHISPER_MODEL_SIZE='tiny', WHISPER_MODEL_LANG='multi')
        job = mpf.AudioJob('test', self._get_test_file('left.wav'), 0, -1, job_props, {})

        result = comp.get_detections_from_audio(job)

        self.assertEqual("multi", comp.wrapper.lang)
        self.assertEqual("tiny", comp.wrapper.size)
        self.assertTrue(comp.wrapper.initialized)

        self.assertEqual(1, len(result))
        self.assertEqual('en', result[0].detection_properties['DETECTED_LANGUAGE'])

    def test_transcribe(self):
        job_props = dict(WHISPER_MODE=1)
        job = mpf.AudioJob('test', self._get_test_file('left.wav'), 0, -1, job_props, {})

        comp = WhisperDetectionComponent()
        result = comp.get_detections_from_audio(job)

        expected_text = "There's three left on the left side, the one closest to us."

        self.assertEqual(1, len(result))
        self.assertEqual('en', result[0].detection_properties['DECODED_LANGUAGE'])
        self.assertEqual('eng', result[0].detection_properties['ISO_LANGUAGE'])
        self.assertEqual(expected_text, result[0].detection_properties["TRANSCRIPT"])

    def test_transcribe_given_language(self):
        expected_text = (
            "Me comunico con diversas personas en inglés y en español o mezclando ambas lenguas. "
            "Hay tantas razones. Yo lo hago por obviamente solidaridad, porque me reciben en esa comunidad. "
            "Como crecién el lado mexicano no acostumbraba en ese pasado. Luego, al cruzar la frontera metafórica "
            "porque no existe derecho, me di cuenta, hablan diferente, salpican verdad su conversación con "
            "palabras en inglés y porque no. Entonces no es fácil hacerlo porque he tratado de hacerlo y lo "
            "aprecio y lo entiendo mucho más por la experiencia que he tenido todos estos años. Y lo hago para "
            "tratar de pertenecer, para no ser diferente que me consideren, como parte de esa comunidad."
        )

        job_props = dict(WHISPER_MODE=1)
        job = mpf.AudioJob('test', self._get_test_file('bilingual.mp3'), 0, -1, job_props, {})

        comp = WhisperDetectionComponent()

        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['DECODED_LANGUAGE'])
        self.assertEqual('est', result[0].detection_properties['ISO_LANGUAGE'])

        # Results for the English portion of the audio are non-deterministic
        self.assertTrue(expected_text in result[0].detection_properties["TRANSCRIPT"])

        job_props = dict(WHISPER_MODE=1, AUDIO_LANGUAGE='es')
        job = mpf.AudioJob('test', self._get_test_file('bilingual.mp3'), 0, -1, job_props, {})

        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))

        # Results for the English portion of the audio are non-deterministic
        self.assertTrue(expected_text in result[0].detection_properties["TRANSCRIPT"])

    def test_translation(self):
        expected_text = (
            'I communicate with different people in English and Spanish or mixing both languages. '
            'There are so many reasons. I do it because obviously solidarity, because they receive me in that '
            'community. As the Mexican people believe, it is not used in that past. Then, when crossing the border, '
            'metaphorically, because there is no right, I realize, talking different, they get out of the truth, '
            'their conversation, with words in English. Why not? It is not easy to do it because I try to do it. '
            'I appreciate it and I understand it much more because of the experience I had all these years. '
            'I do it to try to keep it from being, not to be different, and I consider it to be a community.'
        )

        job_props = dict(WHISPER_MODE=2)
        job = mpf.AudioJob('test', self._get_test_file('bilingual.mp3'), 0, -1, job_props, {})

        comp = WhisperDetectionComponent()

        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))
        self.assertEqual('es', result[0].detection_properties['DECODED_LANGUAGE'])
        self.assertEqual('est', result[0].detection_properties['ISO_LANGUAGE'])
        self.assertTrue(expected_text in result[0].detection_properties["TRANSLATED_AUDIO"])

        job_props = dict(WHISPER_MODE=2, AUDIO_LANGUAGE='es')
        job = mpf.AudioJob('test', self._get_test_file('bilingual.mp3'), 0, -1, job_props, {})

        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))
        self.assertEqual(expected_text, result[0].detection_properties["TRANSLATED_AUDIO"])


if __name__ == '__main__':
    unittest.main(verbosity=2)
