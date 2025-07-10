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

import json
import logging
import mpf_component_api as mpf
import os
import unittest

from nllb_component import NllbTranslationComponent
from pathlib import Path
from typing import Sequence

logging.basicConfig(level=logging.DEBUG)

class TestNllbTranslation(unittest.TestCase):

    #get descriptor.json file path
    cur_path: str = os.path.dirname(__file__)
    descriptorFile: str = os.path.join(cur_path, '../plugin-files/descriptor/descriptor.json')

    #open descriptor.json and save off default props
    with open(descriptorFile) as file:
        descriptor: json = json.load(file)
    descriptorProperties: dict[str, str] = descriptor['algorithm']['providesCollection']['properties']
    defaultProps: dict[str, str] = {}
    for property in descriptor['algorithm']['providesCollection']['properties']:
        defaultProps[property['name']] = property['defaultValue']

    #test translation
    TRANSLATE_THIS_TEXT = (
        'Hallo, wie gehts Heute?' # "Hello, how are you today?"
    )
    #expected nllb translation
    TRANSLATION = (
        "Hi, how are you today?"
    )

    component = NllbTranslationComponent()

    def test_image_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT= self.TRANSLATE_THIS_TEXT))
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_image(job)

        props = result[0].detection_properties
        self.assertEqual(self.TRANSLATION, props["TRANSLATION"])

    def test_audio_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.AudioTrack(0, 1, -1, dict(TEXT= self.TRANSLATE_THIS_TEXT))
        job = mpf.AudioJob('Test Audio',
                           'test.wav', 0, 1,
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_audio(job)

        props = result[0].detection_properties
        self.assertEqual(self.TRANSLATION, props["TRANSLATION"])

    def test_video_job(self):

        TRANSCRIPT_INPUT_1 = (
           'Wie ist das Wetter?' # "How is the weather?"
        )
        EXPECTED_TRANSLATION_1 = (
            "How's the weather?"
        )
        TRANSCRIPT_INPUT_2 = (
           'Es regnet.' # "It's raining"
        )
        EXPECTED_TRANSLATION_2 = (
            "It's raining."
        )
        
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TRANSCRIPT=TRANSCRIPT_INPUT_1)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=TRANSCRIPT_INPUT_2))
            },
            dict(TEXT=self.TRANSLATE_THIS_TEXT))
        
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['TRANSLATE_ALL_FF_PROPERTIES'] = 'TRUE'

        job = mpf.VideoJob('Test Video',
                           'test.mp4', 0, 1,
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_video(job)

        props = result[0].detection_properties
        self.assertEqual(self.TRANSLATION, props["TEXT TRANSLATION"])
        frame_1_props = result[0].frame_locations[0].detection_properties
        self.assertEqual(EXPECTED_TRANSLATION_1, frame_1_props["TRANSCRIPT TRANSLATION"])
        frame_2_props = result[0].frame_locations[1].detection_properties
        self.assertEqual(EXPECTED_TRANSLATION_2, frame_2_props["TRANSCRIPT TRANSLATION"])

    def test_translate_only_text_ff_property_job(self):

        TRANSCRIPT_INPUT_1 = (
           'Wie ist das Wetter?' # "How is the weather?"
        )
        EXPECTED_TRANSLATION_1 = (
            "How's the weather?"
        )
        TRANSCRIPT_INPUT_2 = (
           'Es regnet.' # "It's raining"
        )
        EXPECTED_TRANSLATION_2 = (
            "It's raining."
        )
        
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TRANSCRIPT=TRANSCRIPT_INPUT_1)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=TRANSCRIPT_INPUT_2))
            },
            dict(TEXT=self.TRANSLATE_THIS_TEXT))
        
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        job = mpf.VideoJob('Test Video',
                           'test.mp4', 0, 1,
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_video(job)

        props = result[0].detection_properties
        self.assertEqual(self.TRANSLATION, props["TRANSLATION"])
        self.assertNotIn("TEXT TRANSLATION", props)
        frame_1_props = result[0].frame_locations[0].detection_properties
        self.assertNotIn("TRANSCRIPT TRANSLATION", frame_1_props)
        frame_2_props = result[0].frame_locations[1].detection_properties
        self.assertNotIn("TRANSCRIPT TRANSLATION", frame_2_props)

    def test_generic_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.TRANSLATE_THIS_TEXT))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.TRANSLATION, result_props["TRANSLATION"])

    def test_plaintext_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'

        job = mpf.GenericJob('Test Plaintext', 
                             str(Path(__file__).parent / 'data' / 'translation.txt'), 
                             test_generic_job_props,
                             {})
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.TRANSLATION, result_props["TRANSLATION"])

    def test_unsupported_source_language(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="ABC"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="DEF" 

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.TRANSLATE_THIS_TEXT))
        job = mpf.GenericJob('Test Plaintext', 'test.txt', test_generic_job_props, {}, ff_track)
        comp = NllbTranslationComponent()

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)

    def test_no_script_prop(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language but no script
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.TRANSLATE_THIS_TEXT))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.TRANSLATION, result_props["TRANSLATION"])

    def test_feed_forward_language(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.TRANSLATE_THIS_TEXT, 
                                             LANGUAGE='deu',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.TRANSLATION, result_props["TRANSLATION"])

    def test_numbers_only_not_translated(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT='1234', 
                                             LANGUAGE='spa',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual('1234', result_props["TRANSLATION"])

    def test_punctuation_only_not_translated(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT='!@#$%', 
                                             LANGUAGE='deu',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual('!@#$%', result_props["TRANSLATION"])

    def test_zho_punctuation_only_not_translated(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT='、。〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〞〟',
                                             LANGUAGE='zho',
                                             ISO_SCRIPT='Hans'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual('、。〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〞〟', result_props["TRANSLATION"])

    def test_eng_to_eng_translation(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT='This is English text that should not be translated.', 
                                             LANGUAGE='eng',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual('This is English text that should not be translated.', result_props["TRANSLATION"])

    def test_sentence_split_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        test_generic_job_props['SENTENCE_SPLITTER_CHAR_COUNT'] = '25'

        # translation to split into multiple sentences
        # with default sentence splitter (wtp-bert-mini)
        long_translation_text = (
            'Das ist Satz eins. Das ist Satz zwei. Und das ist Satz drei.'
        )
        expected_translation = "That's the first sentence. That's the second sentence. And that's the third sentence."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=long_translation_text))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(expected_translation, result_props["TRANSLATION"])

        test_generic_job_props['SOURCE_LANGUAGE'] = None
        test_generic_job_props['SENTENCE_MODEL_WTP_DEFAULT_ADAPTOR_LANGUAGE'] = 'en'
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(expected_translation, result_props["TRANSLATION"])
        # test sentence splitter (xx_sent_ud_sm)
        test_generic_job_props['SENTENCE_MODEL'] = 'xx_sent_ud_sm'
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(expected_translation, result_props["TRANSLATION"])

    def test_split_with_non_translate_segments(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'por'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        test_generic_job_props['SENTENCE_SPLITTER_CHAR_COUNT'] = '39'

         # excerpt from https://www.gutenberg.org/ebooks/16443
        pt_text="Os que são gentis são indispensáveis. 012345678901234567890123456789012345. 123456789012345678901234567890123456. Os caridosos são uma luz pra os outros."

        small_model_expected = "Those who are kind are indispensable. 012345678901234567890123456789012345.  123456789012345678901234567890123456.  Charitable people are a light to others."
        large_model_expected = "The kind ones are indispensable. 012345678901234567890123456789012345.  123456789012345678901234567890123456.  Charity workers are a light to others."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=pt_text,
                                             LANGUAGE='por',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(large_model_expected, result_props["TRANSLATION"])

    def test_sentence_split_job2(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'por'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'

        # excerpt from https://www.gutenberg.org/ebooks/16443
        pt_text="""Teimam de facto estes em que são indispensaveis os vividos raios do
nosso desanuviado sol, ou a face desassombrada da lua no firmamento
peninsular, onde não tem, como a de Londres--_a romper a custo um
plumbeo céo_--para verterem alegrias na alma e mandarem aos semblantes o
reflexo d'ellas; imaginam fatalmente perseguidos de _spleen_,
irremediavelmente lugubres e soturnos, como se a cada momento saíssem
das galerias subterraneas de uma mina de _pit-coul_, os nossos alliados
inglezes.

Como se enganam ou como pretendem enganar-nos!

É esta uma illusão ou má fé, contra a qual ha muito reclama debalde a
indelevel e accentuada expressão de beatitude, que transluz no rosto
illuminado dos homens de além da Mancha, os quaes parece caminharem
entre nós, envolvidos em densa atmosphera de perenne contentamento,
satisfeitos do mundo, satisfeitos dos homens e, muito especialmente,
satisfeitos de si.
"""
        small_model_expected = "They hold these in fact indispensable in which the vivid rays of our sunless sun, or the moon's dim face in the peninsular sky, where it has no, as in London--_a breaking at cost a plumb sky_--to pour joy into the soul and send its reflection to the semblance; imagine fatally pursued by _spleen_, hopelessly gloomy and submerged, as if at every moment they were coming out of the underground galleries of a mine of _pit-coul_, our English allies.  How they deceive themselves or how they intend to deceive us! This is an illusion or bad faith, against which there is much complaint about the indelible and accentuated expression of bliss, which transluces in the illuminated face of men beyond the Channel, the quaes seem to walk among us, enveloped in a dense atmosphere of perennial contentment, satisfied with the world, satisfied with men and, very especially, satisfied with themselves. "
        large_model_expected = "They fear, indeed, those in whom the vivid rays of our languid sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--to pour joys into the soul and send to the countenances the reflection d'elles; they imagine fatally pursued from spleen, hopelessly gloomy and sullen, as if at every moment they came out of the underground galleries of a mine of coulpit, our English allies. How they deceive themselves or how they try to deceive us! is this an illusion or bad faith, against which there is much complaint in vain the indelevel and accentuated expression of beatitude, which shines through the illuminated face of the men from beyond the Spot, who seem to walk among us, wrapped in a dense atmosphere of perennial contentment, satisfied with the world, satisfied with men and, very especially, satisfied with themselves."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=pt_text))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(large_model_expected, result_props["TRANSLATION"])

    @unittest.skip('Work in progress')
    def test_sentence_split_job3(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'por'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'

        job = mpf.GenericJob('Test Plaintext',
                             str(Path(__file__).parent / 'data' / 'pg16443.txt'),
                             test_generic_job_props,
                             {})
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.TRANSLATION, result_props["TRANSLATION"])

    def test_arb_eng(self):
        input_text="أعلن أوريول جونكيراس يوم السبت، وهو أحد القادة التسعة في كتالونيا المودع في السجن رهن المحاكمة منذ أواخر العام الماضي أنه سيرشح نفسه في انتخابات البرلمان الأوروبي في العام القادم."
        ground_truth_translation =  "On Saturday, Oriol Junqueras, one of nine Catalan leaders in pre-trial jail since late last year, announced he would run in European Parliament elections next year."
        small_model_expected     =  "O'Reilly Junqueras, one of nine Catalonia leaders jailed since late last year, announced on Saturday that he would run for the European Parliament election next year."
        large_model_expected     =  "Oriol Junqueras, one of nine leaders in Catalonia who has been in jail on trial since late last year, announced on Saturday that he will run in the European Parliament elections next year."
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT=input_text,
                                             LANGUAGE='arb',
                                             ISO_SCRIPT='Arab'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(large_model_expected, result_props["TRANSLATION"])

    def test_eng_aeb(self):
        input_text="Have you traveled anywhere recently?"
        small_model_expected = "هل سافر في اي مكان مؤخرا؟"
        large_model_expected = "سافرتي في أي بلاصة مؤخرا؟"
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['TARGET_LANGUAGE'] = 'aeb'
        test_generic_job_props['TARGET_SCRIPT'] = 'Arab'
        ff_track = mpf.GenericTrack(-1, dict(TEXT=input_text,
                                             LANGUAGE='eng',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(large_model_expected, result_props["TRANSLATION"])

    def test_aeb_eng(self):
        input_text="و يحبو يقعدوا غادي ولا مذبيهم يرجعوا لتونس؟"
        ground_truth_translation = "Do they want to stay there, or do they want to come back to Tunisia?"
        small_model_expected     = "Do they want to stay there or do their clients want to go back to Tunisia ?"
        large_model_expected     = "Do they want to stay there or do they want to go back to Tunisia ?"
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT=input_text,
                                             LANGUAGE='aeb',
                                             ISO_SCRIPT='Arab'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(large_model_expected, result_props["TRANSLATION"])

if __name__ == '__main__':
    unittest.main()
