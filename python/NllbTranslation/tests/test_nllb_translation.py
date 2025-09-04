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

from nllb_component.nllb_translation_component import should_translate

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
    SAMPLE_0 = (
        'Hallo, wie gehts Heute?' # "Hello, how are you today?"
    )
    #expected nllb translation
    OUTPUT_0 = (
        "Hi, how are you today?"
    )
    SAMPLE_1 = (
        'Wie ist das Wetter?' # "How is the weather?"
    )
    OUTPUT_1 = (
        "How's the weather?"
    )
    SAMPLE_2 = (
        'Es regnet.' # "It's raining"
    )
    OUTPUT_2 = (
        "It's raining."
    )

    component = NllbTranslationComponent()

    def test_image_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT= self.SAMPLE_0))
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_image(job)

        props = result[0].detection_properties
        self.assertEqual(self.OUTPUT_0, props["TRANSLATION"])

    def test_audio_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.AudioTrack(0, 1, -1, dict(TEXT= self.SAMPLE_0))
        job = mpf.AudioJob('Test Audio',
                           'test.wav', 0, 1,
                           test_generic_job_props,
                           {}, ff_track)
        result = self.component.get_detections_from_audio(job)

        props = result[0].detection_properties
        self.assertEqual(self.OUTPUT_0, props["TRANSLATION"])

    def test_video_job(self):
        
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TRANSCRIPT=self.SAMPLE_1)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=self.SAMPLE_2))
            },
            dict(TEXT=self.SAMPLE_0))
        
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
        self.assertEqual(self.OUTPUT_0, props["TEXT TRANSLATION"])
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertEqual(self.OUTPUT_1, frame_0_props["TRANSCRIPT TRANSLATION"])
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertEqual(self.OUTPUT_2, frame_1_props["TRANSCRIPT TRANSLATION"])

    def test_generic_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.OUTPUT_0, result_props["TRANSLATION"])

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
        self.assertEqual(self.OUTPUT_0, result_props["TRANSLATION"])

    def test_translate_first_ff_property(self):
        # set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['TRANSLATE_ALL_FF_PROPERTIES'] = 'FALSE' # default
        # set source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['FEED_FORWARD_PROP_TO_PROCESS'] = 'TEXT,TRANSCRIPT' # default

        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TRANSCRIPT=self.SAMPLE_1,TEXT=self.SAMPLE_0)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TEXT=self.SAMPLE_0,TRANSCRIPT=self.SAMPLE_2))
            },
            dict(TRANSCRIPT=self.SAMPLE_0))
        
        job = mpf.VideoJob('Test Video',
                        'test.mp4', 0, 1,
                        test_generic_job_props,
                        {}, ff_track)
        result = self.component.get_detections_from_video(job)

        props = result[0].detection_properties
        self.assertIn("TRANSLATION", props)
        self.assertNotIn("TRANSCRIPT TRANSLATION", props)
        self.assertEqual(self.OUTPUT_0, props["TRANSLATION"])
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertIn("TRANSLATION", frame_0_props)
        self.assertEqual(self.OUTPUT_0, frame_0_props["TRANSLATION"])
        self.assertNotIn("TEXT TRANSLATION", frame_0_props)
        self.assertNotIn("TRANSCRIPT TRANSLATION", frame_0_props)
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertIn("TRANSLATION", frame_1_props)
        self.assertEqual(self.OUTPUT_0, frame_1_props["TRANSLATION"])
        self.assertNotIn("TEXT TRANSLATION", frame_1_props)
        self.assertNotIn("TRANSCRIPT TRANSLATION", frame_1_props)

    def test_translate_all_ff_properties(self):
        # set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        # set source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['FEED_FORWARD_PROP_TO_PROCESS'] = 'TEXT,TRANSCRIPT' # default
        # set TRANSLATE_ALL_FF_PROPERTIES = 'TRUE'
        test_generic_job_props['TRANSLATE_ALL_FF_PROPERTIES'] = 'TRUE'

        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TRANSCRIPT=self.SAMPLE_1,TEXT=self.SAMPLE_0)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=self.SAMPLE_2,TEXT=self.SAMPLE_0)),
                2: mpf.ImageLocation(0, 20, 20, 20, -1, dict(OTHER=self.SAMPLE_0))
            },
            dict(TEXT=self.SAMPLE_0))

        job = mpf.VideoJob('Test Video',
                        'test.mp4', 0, 1,
                        test_generic_job_props,
                        {}, ff_track)
        result = self.component.get_detections_from_video(job)

        props = result[0].detection_properties
        self.assertIn("TEXT TRANSLATION", props)
        self.assertEqual(self.OUTPUT_0, props["TEXT TRANSLATION"])
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertIn("TRANSCRIPT TRANSLATION", frame_0_props)
        self.assertEqual(self.OUTPUT_1, frame_0_props["TRANSCRIPT TRANSLATION"])
        self.assertIn("TEXT TRANSLATION", frame_0_props)
        self.assertEqual(self.OUTPUT_0, frame_0_props["TEXT TRANSLATION"])
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertIn("TRANSCRIPT TRANSLATION", frame_1_props)
        self.assertEqual(self.OUTPUT_2, frame_1_props["TRANSCRIPT TRANSLATION"])
        self.assertIn("TEXT TRANSLATION", frame_1_props)
        self.assertEqual(self.OUTPUT_0, frame_1_props["TEXT TRANSLATION"])
        frame_2_props = result[0].frame_locations[2].detection_properties
        self.assertNotIn("OTHER TRANSLATION", frame_2_props)
    
    def test_translate_first_frame_location_property(self):
        # set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['TRANSLATE_ALL_FF_PROPERTIES'] = 'FALSE' # default
        # set source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['FEED_FORWARD_PROP_TO_PROCESS'] = 'TEXT,TRANSCRIPT' # default

        # Expected: only TEXT and TRANSCRIPT are processed in the detection properties
        #      AND nothing is processed in track properties.
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(OTHER_PROPERTY="Other prop text", TEXT=self.SAMPLE_1)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=self.SAMPLE_2))
            })
        
        job = mpf.VideoJob('Test Video',
                        'test.mp4', 0, 1,
                        test_generic_job_props,
                        {}, ff_track)
        result = self.component.get_detections_from_video(job)

        props = result[0].detection_properties
        self.assertNotIn("TRANSLATION", props)
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertIn("TRANSLATION", frame_0_props)
        self.assertEqual(self.OUTPUT_1, frame_0_props["TRANSLATION"])
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertIn("TRANSLATION", frame_1_props)
        self.assertEqual(self.OUTPUT_2, frame_1_props["TRANSLATION"])

    def test_unsupported_source_language(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="ABC"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="DEF" 

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
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

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.OUTPUT_0, result_props["TRANSLATION"])

    def test_feed_forward_language(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0, 
                                             LANGUAGE='deu',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(self.OUTPUT_0, result_props["TRANSLATION"])

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
        test_generic_job_props['SENTENCE_MODEL'] = 'wtp-bert-mini'

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

        pt_text_translation = "The kind ones are indispensable. 012345678901234567890123456789012345.  123456789012345678901234567890123456.  Charity workers are a light to others."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=pt_text,
                                             LANGUAGE='por',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(pt_text_translation, result_props["TRANSLATION"])

    def test_paragraph_split_job(self):
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
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable, to pour joy into the soul and send to the semblances the reflection of them; they imagine fatally pursued from _spleen_,  hopelessly gloomy and sullen, as if at every moment they were emerging from the subterranean galleries of a pit-coal mine, our British allies. How they deceive themselves or how they intend to deceive us! This is an illusion or bad faith, against which much is vainly complained the unlevel and accentuated expression of bliss, which shines through on the face. The European Parliament has been a great help to the people of Europe in the past, and it is a great help to us in the present."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=pt_text))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(pt_text_translation, result_props["TRANSLATION"])

    def test_should_translate(self):

        with self.subTest('OK to translate'):
            self.assertTrue(should_translate("Test 123."))         # Letters and numbers
            self.assertTrue(should_translate("abcdefg"))           # Only letters
            self.assertTrue(should_translate("123 Main St."))      # Contains letters
            self.assertTrue(should_translate("I have five (5) apples.")) # eng_Latn (English)
            self.assertTrue(should_translate("मेरे पास पाँच (5) सेब हैं।")) # awa_Deva (Awadhi)
            self.assertTrue(should_translate("Миндә биш (5) алма бар.")) # bak_Cyrl (Bashkir)
            self.assertTrue(should_translate("ང་ལ་ཀུ་ཤུ་ལྔ་(༥) ཡོད།")) # bod_Tibt (Tibetan)
            self.assertTrue(should_translate("મારી પાસે પાંચ (5) સફરજન છે.")) # guj_Gujr (Gujarati)
            self.assertTrue(should_translate("יש לי חמישה (5) תפוחים.")) # heb_Hebr (Hebrew)
            self.assertTrue(should_translate("मेरे पास पाँच (5) सेब हैं।")) # hin_Deva (Hindi)
            self.assertTrue(should_translate("Ես ունեմ հինգ (5) խնձոր։")) # hye_Armn (Armenian)
            self.assertTrue(should_translate("私はりんごを5個持っています。")) # jpn_Jpan (Japanese)
            self.assertTrue(should_translate("ನನಗೆ ಐದು (5) ಸೇಬುಗಳಿವೆ.")) # kan_Knda (Kannada)
            self.assertTrue(should_translate("მე მაქვს ხუთი (5) ვაშლი.")) # kat_Geor (Georgian)
            self.assertTrue(should_translate("ខ្ញុំមានផ្លែប៉ោមប្រាំ (5) ផ្លែ។")) # khm_Khmr (Khmer)
            self.assertTrue(should_translate("나는 사과 다섯 (5) 개가 있어요.")) # kor_Hang (Korean)
            self.assertTrue(should_translate("എനിക്ക് ആപ്പിളുകൾ അഞ്ചെ (5) ഉണ്ട്.")) # mal_Mlym (Malayalam)
            self.assertTrue(should_translate("ကျွန်တော်မှာ ပန်းသီး ငါး (5) လုံးရှိတယ်။")) # mya_Mymr (Burmese)
            self.assertTrue(should_translate("මට ආපල් පස් (5) තියෙනවා.")) # sin_Sinh (Sinhala)
            self.assertTrue(should_translate("எனக்கு ஐந்து (5) ஆப்பிள்கள் இருக்கின்றன.")) # tam_Taml (Tamil)
            self.assertTrue(should_translate("నాకు ఐదు (5) ఆపిళ్లు ఉన్నాయి.")) # tel_Telu (Telugu)
            self.assertTrue(should_translate("Ман панҷ (5) себ дорам.")) # tgk_Cyrl (Tajik)
            self.assertTrue(should_translate("ฉันมีแอปเปิ้ลห้า (5) ลูก")) # tha_Thai (Thai)
            self.assertTrue(should_translate("ኣነ ሓምሽተ (5) ፖም ኣሎኒ።")) # tir_Ethi (Tigrinya)
            self.assertTrue(should_translate("Mi gat five (5) apple.")) # tpi_Latn (Tok Pisin)
            self.assertTrue(should_translate("Mo ní ẹ̀pàlà márùn-ún (5).")) # yor_Latn (Yoruba)
            self.assertTrue(should_translate("我有五 (5) 個蘋果。")) # yue_Hant (Yue Chinese / Cantonese)

        with self.subTest('Do not translate'):
            # do not send to nllb
            self.assertFalse(should_translate('、。〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〞〟')) # Chinese punctuation and special characters
            self.assertFalse(should_translate("123.456 !"))         # Digits, punctuation, whitespace
            self.assertFalse(should_translate("\t-1,000,000.00\n")) # All three categories
            self.assertFalse(should_translate("()[]{}"))            # Only punctuation
            self.assertFalse(should_translate(" \n "))              # Only whitespace
            self.assertFalse(should_translate(""))                  # Empty string

        # Subtests:
        # A selection of test strings to cover all non-letter unicode character categories
        # see https://www.unicode.org/versions/Unicode16.0.0/core-spec/chapter-4/#G134153
        # see also https://www.unicode.org/Public/UCD/latest/ucd/PropList.txt
        #
        # Unicode category tests
        #
        with self.subTest('Decimal_Number: a decimal digit'):
            self.assertFalse(should_translate("0123456789"))        # Only digits
            self.assertFalse(should_translate("٠١٢٣٤٥٦٧٨٩")) #  Arabic-Indic digits (\u0660-\u0669)
            self.assertFalse(should_translate("۰۱۲۳۴۵۶۷۸۹")) #  Eastern Arabic-Indic digits (\u06F0-\u06F9)
            self.assertFalse(should_translate("߀߁߂߃߄߅߆߇߈߉")) #  NKo (Mangding) digits (\u07C0-\u07C9)
            self.assertFalse(should_translate("०१२३४५६७८९")) #  Devanagari digits (\u0966-\u096F)
            self.assertFalse(should_translate("০১২৩৪৫৬৭৮৯")) #  Bengali digits (\u09E6-\u09EF)
            self.assertFalse(should_translate("੦੧੨੩੪੫੬੭੮੯")) #  Gurmukhi digits (\u0A66-\u0A6F)
            self.assertFalse(should_translate("૦૧૨૩૪૫૬૭૮૯")) #  Gujarati digits (\u0AE6-\u0AEF)
            self.assertFalse(should_translate("୦୧୨୩୪୫୬୭୮୯")) #  Oriya digits (\u0B66-\u0B6F)
            self.assertFalse(should_translate("௦௧௨௩௪௫௬௭௮௯")) #  Tamil digits (\u0BE6-\u0BEF)
            self.assertFalse(should_translate("౦౧౨౩౪౫౬౭౮౯")) #  Telugu digits (\u0C66-\u0C6F)
            self.assertFalse(should_translate("೦೧೨೩೪೫೬೭೮")) #  Kannada digits (\u0CE6-\u0CEF)
            self.assertFalse(should_translate("೯൦൧൨൩൪൫൬൭൮൯")) #  Malayalam digits (\u0D66-\u0D6F)
            self.assertFalse(should_translate("෦෧෨෩෪෫෬෭෮෯")) #  Astrological digits (\u0DE6-\u0DEF)
            self.assertFalse(should_translate("๐๑๒๓๔๕๖๗๘๙")) #  Thai digits (\u0E50-\u0E59)
            self.assertFalse(should_translate("໐໑໒໓໔໕໖໗໘໙")) #  Lao digits (\u0ED0-\u0ED9)
            self.assertFalse(should_translate("༠༡༢༣༤༥༦༧༨༩")) #  Tibetan digits (\u0F20-\u0F29)
            self.assertFalse(should_translate("༪༫༬༭༮༯༰༱༲༳")) #  Tibetan half digits (\u0F20-\u0F29)
            self.assertFalse(should_translate("၀၁၂၃၄၅၆၇၈၉")) #  Myanmar digits (\u1040-\u1049)
            self.assertFalse(should_translate("႐႑႒႓႔႕႖႗႘႙")) #  Myanmar Shan digits (\u1090-\u1099)
            self.assertFalse(should_translate("፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼")) #  Ethiopic digits (\u1369-\u137C)
            self.assertFalse(should_translate("០១២៣៤៥៦៧៨៩")) #  Khmer digits (\u17E0-\u17E9)
            self.assertFalse(should_translate("᠐᠑᠒᠓᠔᠕᠖᠗᠘᠙")) #  Mongolian digits (\u1810-\u1819)
            self.assertFalse(should_translate("᥆᥇᥈᥉᥊᥋᥌᥍᥎᥏")) #  Limbu digits (\u1946-\u194F)
            self.assertFalse(should_translate("᧐᧑᧒᧓᧔᧕᧖᧗᧘᧙")) #  New Tai Lue digits (\u19D0-\u19D9)
            self.assertFalse(should_translate("᪀᪁᪂᪃᪄᪅᪆᪇᪈᪉")) #  Tai Tham Hora digits (\u1A80-\u1A89)
            self.assertFalse(should_translate("᪐᪑᪒᪓᪔᪕᪖᪗᪘᪙")) #  Tai Tham Tham digits (\u1A90-\u1A99)
            self.assertFalse(should_translate("᭐᭑᭒᭓᭔᭕᭖᭗᭘᭙")) #  Balinese digits (\u1B50-\u1B59)
            self.assertFalse(should_translate("᮰᮱᮲᮳᮴᮵᮶᮷᮸᮹")) #  Sundanese digits (\u1BB0-\u1BB9)
            self.assertFalse(should_translate("᱀᱁᱂᱃᱄᱅᱆᱇᱈᱉")) #  Lepcha digits (\u1C40-\u1C49)
            self.assertFalse(should_translate("᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙")) #  Ol Chiki digits (\u1C50-\u1C59)
            self.assertFalse(should_translate("꘠꘡꘢꘣꘤꘥꘦꘧꘨꘩")) #  Vai digits (\uA620-\uA629)
            self.assertFalse(should_translate("꣐꣑꣒꣓꣔꣕꣖꣗꣘꣙")) #  Saurashtra digits (\uA8D0-\uA8D9)
            self.assertFalse(should_translate("꤀꤁꤂꤃꤄꤅꤆꤇꤈꤉")) #  Kayah Li digits (\uA900-\uA909)
            self.assertFalse(should_translate("꧐꧑꧒꧓꧔꧕꧖꧗꧘꧙")) #  Javanese digits (\uA9D0-\uA9D9)
            self.assertFalse(should_translate("꧰꧱꧲꧳꧴꧵꧶꧷꧸꧹")) #  Tai Laing digits (\uA9F0-\uA9F9)
            self.assertFalse(should_translate("꩐꩑꩒꩓꩔꩕꩖꩗꩘꩙")) #  Cham digits (\uAA50-\uAA59)
            self.assertFalse(should_translate("꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹")) #  Meetei Mayek digits (\uABF0-\uABF9)
            self.assertFalse(should_translate("０１２３４５６７８９")) #  Full width digits (\uFF10-\uFF19)
            
        with self.subTest('Letter_Number: a letterlike numeric character'):
            letter_numbers = "ᛮᛯᛰⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩⅪⅫⅬⅭⅮⅯⅰⅱⅲⅳⅴⅵⅶⅷⅸⅹⅺⅻⅼⅽⅾⅿↀↁↂↅↆↇↈ〇〡〢〣〤〥〦〧〨〩〸〹〺ꛦꛧꛨꛩꛪꛫꛬꛭꛮꛯ"
            self.assertFalse(should_translate(letter_numbers))
        
        with self.subTest('Other_Number: a numeric character of other type'):
            other_numbers1 = "²³¹¼½¾৴৵৶৷৸৹୲୳୴୵୶୷௰௱௲౸౹౺౻౼౽౾൘൙൚൛൜൝൞൰൱൲൳൴൵൶൷൸༪༫༬༭༮༯༰༱༲༳፩፪፫፬፭፮፯፰፱፲፳፴፵፶፷፸፹፺፻፼"
            other_numbers2 = "៰៱៲៳៴៵៶៷៸៹᧚⁰⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅟↉①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
            other_numbers3 = "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛⓪⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴"
            other_numbers4 = "⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾⓿❶❷❸❹❺❻❼❽❾❿➀➁➂➃➄➅➆➇➈➉➊➋➌➍➎➏➐➑➒➓⳽㆒㆓㆔㆕㈠㈡㈢㈣㈤㈥㈦㈧㈨㈩㉈㉉㉊㉋㉌㉍㉎㉏"
            other_numbers5 = "㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊀㊁㊂㊃㊄㊅㊆㊇㊈㊉㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿꠰꠱꠲꠳꠴꠵"
            self.assertFalse(should_translate(other_numbers1))
            self.assertFalse(should_translate(other_numbers2))
            self.assertFalse(should_translate(other_numbers3))
            self.assertFalse(should_translate(other_numbers4))
            self.assertFalse(should_translate(other_numbers5))

        with self.subTest('# Nonspacing_Mark: a nonspacing combining mark, zero advance width (selected sample)'):
        # (NOTE: test string should always include \u1734, as \p{Nonspacing_Mark} fails to match it)
            nonspacing_marks = "\u0300\u0483\u0591\u0A01\u0B01\u0C00\u0D00\u0E31\u0F18\u1734\u1BAD\u2CEF\uFE2A\uFE2B\uFE2C\uFE2D\uFE2E\uFE2F"
            self.assertFalse(should_translate(nonspacing_marks))

        with self.subTest('# Spacing_Mark: a spacing combining mark (positive advance width)'):
            spacing_marks = "\u0903\u093B\u093E\u093F\u0940\uAA7D\uAAEB\uAAEE\uAAEF\uAAF5\uABE3\uABE4\uABE6\uABE7\uABE9\uABEA\uABEC"
            self.assertFalse(should_translate(spacing_marks))

        with self.subTest('# Enclosing_Mark: an enclosing combining mark'):
            enclosing_marks = "\u0488\u0489\u1ABE\u20DD\u20DE\u20DF\u20E0\u20E2\u20E3\u20E4\uA670\uA671\uA672"
            self.assertFalse(should_translate(enclosing_marks))

        with self.subTest('# Connector_Punctuation: a connecting punctuation mark, like a tie'):
            connector_punct = "_‿⁀⁔︳︴﹍﹎﹏＿"
            self.assertFalse(should_translate(connector_punct))

        with self.subTest('# Dash_Punctuation: a dash or hyphen punctuation mark'):
            dash_punct = "-֊־᐀᠆‐‑‒–—―⸗⸚⸺⸻⹀〜〰゠︱︲﹘﹣－"
            self.assertFalse(should_translate(dash_punct))

        with self.subTest('# Open_Punctuation: an opening punctuation mark (of a pair)'):
            open_punct = "([{༺༼᚛‚„⁅⁽₍⌈⌊〈❨❪❬❮❰❲❴⟅⟦⟨⟪⟬⟮⦃⦅⦇⦉⦋⦍⦏⦑⦓⦕⦗⧘⧚⧼⸢⸤⸦⸨⹂〈《「『【〔〖〘〚〝﴿︗︵︷︹︻︽︿﹁﹃﹇﹙﹛﹝（［｛｟｢"
            self.assertFalse(should_translate(open_punct))

        with self.subTest('# Close_Punctuation: a closing punctuation mark (of a pair)'):
            close_punct = ")]}༻༽᚜⁆⁾₎⌉⌋〉❩❫❭❯❱❳❵⟆⟧⟩⟫⟭⟯⦄⦆⦈⦊⦌⦎⦐⦒⦔⦖⦘⧙⧛⧽⸣⸥⸧⸩〉》」』】〕〗〙〛〞〟﴾︘︶︸︺︼︾﹀﹂﹄﹈﹚﹜﹞）］｝｠｣"
            self.assertFalse(should_translate(close_punct))

        with self.subTest('# Initial_Punctuation: an initial quotation mark'):
            initial_punct = "«‘‛“‟‹⸂⸄⸉⸌⸜⸠"
            self.assertFalse(should_translate(initial_punct))

        with self.subTest('# Final_Punctuation: a final quotation mark'):
            final_punct = "»’”›⸃⸅⸊⸍⸝⸡"
            self.assertFalse(should_translate(final_punct))

        with self.subTest('# Other_Punctuation: a punctuation mark of other type (selected sample)'):
            other_punct = "౷၌፦៙᪥᭛᳀᳆⁌⁍⳹⳺⳻⳼⸔⸕、。〃〽・꓾꓿꧁꧂"
            self.assertFalse(should_translate(other_punct))

        with self.subTest('# Math_Symbol: a symbol of mathematical use (selected sample)'):
            math_symbols = "∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳⊔⊕⩌⩍⩎⩏⩐⩑⩒⩓⩔⩕⩖⩗⩘⩙⩚⩛⩜⩝⩞⩟⩠⩡⩢⩣⩤⩥"
            self.assertFalse(should_translate(math_symbols))

        with self.subTest('# Currency_Symbol: a currency sign'):
            currency_symbols = "$¢£¤¥֏؋߾߿৲৳৻૱௹฿៛₠₡₢₣₤₥₦₧₨₩₪₫€₭₮₯₰₱₲₳₴₵₶₷₸₹₺₻₼₽₾₿꠸﷼﹩＄￠￡￥￦"
            self.assertFalse(should_translate(currency_symbols))

        with self.subTest('# Modifier_Symbol: non-letterlike modifier symbols'):
            modifier_symbols = "^`¨¯´¸˂˃˄˅˒˓˔˕˖˗˘˙˚˛˜˝˞˟˥˦˧˨˩˪˫˭˯˰˱˲˳˴˵˶˷˸˹˺˻˼˽˾˿͵΄΅᾽᾿῀῁῍῎῏῝῞῟῭΅`´῾゛゜꜀꜁꜂꜃꜄꜅꜆꜇꜈꜉꜊꜋꜌꜍꜎꜏꜐꜑꜒꜓꜔꜕꜖꜠꜡꞉꞊꭛꭪꭫﮲﮳﮴﮵﮶﮷﮸﮹﮺﮻﮼﮽﮾﮿﯀﯁＾｀￣"
            self.assertFalse(should_translate(modifier_symbols))

        with self.subTest('# Space_Separator: a space character (of various non-zero widths)'):
            space_separators = ("\u0020\u00A0\u1680\u2000\u2001\u2002\u2003\u2004\u2005" +
                                "\u2006\u2007\u2008\u2009\u200A\u202F\u205F\u3000")
            self.assertFalse(should_translate(space_separators))

        with self.subTest('# Line_Separator (U+2028) and Paragraph_Separator (U+2029)'):
            separators = "\u2028\u2029"
            self.assertFalse(should_translate(separators))

        with self.subTest('# Format: format control characters'):
            format_control = ("\u00AD\u0600\u0601\u0602\u0603\u0604\u0605\u061C\u06DD\u070F\u08E2\u180E" +
                            "\u200B\u200C\u200D\u200E\u200F\u202A\u202B\u202C\u202D\u202E\u2060\u2061" +
                            "\u2062\u2063\u2064\u2066\u2067\u2068\u2069\u206A\u206B\u206C\u206D\u206E" +
                            "\u206F\uFEFF\uFFF9\uFFFA\uFFFB")
            self.assertFalse(should_translate(format_control))

        with self.subTest('# test combinations of character categories'):
            do_not_translate = "\uFEFF₷႑႒႓\u0483\u093B\u2028\u0488︳︴\u0489〜\u2029༼༽\u3000⸠˽⸡꧁∑⓼Ⅷ꧂"
            self.assertFalse(should_translate(do_not_translate))
            do_translate = "ゴールドシップ は、日本の競走馬、種牡馬。" + do_not_translate
            self.assertTrue(should_translate(do_translate))

if __name__ == '__main__':
    unittest.main()
