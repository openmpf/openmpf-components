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

from nllb_component import NllbTranslationComponent, JobConfig
from pathlib import Path
from typing import Sequence
from nlp_text_splitter import WtpLanguageSettings

from nllb_component.nllb_translation_component import should_translate
from nllb_component.nllb_utils import NllbLanguageMapper

logging.basicConfig(level=logging.DEBUG)

# Certain tests are rather expensive, especially the Spanish dracula section.
# Disabling unless we are making specific changes to the component in future tests.
#
# These are also the only tests that assert EXACT model output. They are the golden
# baseline, generated with BUILD_TYPE=gpu (float16). A BUILD_TYPE=cpu image uses
# int8_float32 and legitimately words some sentences differently, so these will not
# pass there -- run them on a gpu build. Every other test asserts structurally (see
# assertTranslated) precisely so the suite is green on both build targets.
RUN_DEEP_TESTS = False

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

    def assertTranslated(self, translation, source, msg=None):
        """Assert a translation was produced, without pinning its exact wording.

        Exact output depends on the model precision the image was built with:
        BUILD_TYPE=gpu (float16) and BUILD_TYPE=cpu (int8_float32) legitimately
        word things differently -- measured at 24/30 agreement on a sample, and
        divergence is driven by lexical ambiguity rather than input length, so it
        cannot be avoided by choosing "simpler" test data. Tests that exercise
        plumbing or configuration therefore assert that translation *happened*,
        and leave exact-output checks to the RUN_DEEP_TESTS golden tests.
        """
        self.assertIsInstance(translation, str, msg)
        self.assertTrue(translation.strip(), msg or 'translation is empty')
        self.assertNotEqual(source.strip(), translation.strip(),
                            msg or 'source was passed through untranslated')

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

    # The single shared component for the whole suite. Each NllbTranslationComponent()
    # loads its own copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two
    # copies do not fit on a 16 GB card -- so tests reuse this one.
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
        # Verifies an ImageLocation's TEXT reaches TRANSLATION, not the wording of the output.
        self.assertTranslated(props["TRANSLATION"], self.SAMPLE_0)

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
        # Verifies an AudioTrack's TEXT reaches TRANSLATION, not the wording of the output.
        self.assertTranslated(props["TRANSLATION"], self.SAMPLE_0)

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
        # Verifies the track and each frame location get their own '<PROP> TRANSLATION'
        # key. Output wording is model-dependent and deliberately not asserted.
        self.assertTranslated(props["TEXT TRANSLATION"], self.SAMPLE_0)
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertTranslated(frame_0_props["TRANSCRIPT TRANSLATION"], self.SAMPLE_1)
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertTranslated(frame_1_props["TRANSCRIPT TRANSLATION"], self.SAMPLE_2)

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
        # Verifies a GenericTrack's TEXT reaches TRANSLATION, not the wording of the output.
        self.assertTranslated(result_props["TRANSLATION"], self.SAMPLE_0)

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
        # Verifies a plain-text file with no feed-forward track is read and translated.
        self.assertTranslated(result_props["TRANSLATION"], self.SAMPLE_0)

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
        # Verifies only the FIRST configured feed-forward property is translated.
        self.assertTranslated(props["TRANSLATION"], self.SAMPLE_0)
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertIn("TRANSLATION", frame_0_props)
        self.assertTranslated(frame_0_props["TRANSLATION"], self.SAMPLE_0)
        self.assertNotIn("TEXT TRANSLATION", frame_0_props)
        self.assertNotIn("TRANSCRIPT TRANSLATION", frame_0_props)
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertIn("TRANSLATION", frame_1_props)
        self.assertTranslated(frame_1_props["TRANSLATION"], self.SAMPLE_0)
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
        # Verifies EVERY configured property is translated into its own key.
        self.assertTranslated(props["TEXT TRANSLATION"], self.SAMPLE_0)
        frame_0_props = result[0].frame_locations[0].detection_properties
        self.assertIn("TRANSCRIPT TRANSLATION", frame_0_props)
        self.assertTranslated(frame_0_props["TRANSCRIPT TRANSLATION"], self.SAMPLE_1)
        self.assertIn("TEXT TRANSLATION", frame_0_props)
        self.assertTranslated(frame_0_props["TEXT TRANSLATION"], self.SAMPLE_0)
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertIn("TRANSCRIPT TRANSLATION", frame_1_props)
        self.assertTranslated(frame_1_props["TRANSCRIPT TRANSLATION"], self.SAMPLE_2)
        self.assertIn("TEXT TRANSLATION", frame_1_props)
        self.assertTranslated(frame_1_props["TEXT TRANSLATION"], self.SAMPLE_0)
        frame_2_props = result[0].frame_locations[2].detection_properties
        self.assertNotIn("OTHER TRANSLATION", frame_2_props)
        self.assertIn("OTHER", frame_2_props)

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
        self.assertIn("OTHER_PROPERTY", frame_0_props)
        # Verifies frame-location properties are translated.
        self.assertTranslated(frame_0_props["TRANSLATION"], self.SAMPLE_1)
        frame_1_props = result[0].frame_locations[1].detection_properties
        self.assertIn("TRANSLATION", frame_1_props)
        self.assertTranslated(frame_1_props["TRANSLATION"], self.SAMPLE_2)

    def test_unsupported_source_language(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="ABC"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="Latn"

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Plaintext', 'test.txt', test_generic_job_props, {}, ff_track)
        # Reuse the shared component: each NllbTranslationComponent() loads its own
        # copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two copies do not
        # fit on a 16 GB card. Nothing here needs a fresh instance.
        comp = self.component

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)
        self.assertEqual('Source language (ABC) is empty or unsupported (DetectionError.INVALID_PROPERTY)', str(cm.exception))

    def test_unsupported_target_language(self):
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="deu"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="Latn"
        test_generic_job_props['TARGET_LANGUAGE']="ABC"
        test_generic_job_props['TARGET_SCRIPT']="Latn"

        ff_track = mpf.GenericTrack(-1, dict(TEXT="Hello"))
        job = mpf.GenericJob('Test Plaintext', 'test.txt', test_generic_job_props, {}, ff_track)
        # Reuse the shared component: each NllbTranslationComponent() loads its own
        # copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two copies do not
        # fit on a 16 GB card. Nothing here needs a fresh instance.
        comp = self.component

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)
        self.assertEqual('Target language (ABC) is not supported (DetectionError.INVALID_PROPERTY)', str(cm.exception))

    def test_unsupported_source_script(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="deu"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="BadScript"

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Plaintext', 'test.txt', test_generic_job_props, {}, ff_track)
        # Reuse the shared component: each NllbTranslationComponent() loads its own
        # copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two copies do not
        # fit on a 16 GB card. Nothing here needs a fresh instance.
        comp = self.component

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)
        self.assertEqual('Language/script combination (deu_BadScript) is invalid or not supported (DetectionError.INVALID_PROPERTY)', str(cm.exception))

    def test_unsupported_target_script(self):
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="deu"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="Latn"
        test_generic_job_props['TARGET_LANGUAGE']="eng"
        test_generic_job_props['TARGET_SCRIPT']="BadScript"

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Plaintext', 'test.txt', test_generic_job_props, {}, ff_track)
        # Reuse the shared component: each NllbTranslationComponent() loads its own
        # copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two copies do not
        # fit on a 16 GB card. Nothing here needs a fresh instance.
        comp = self.component

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)
        self.assertEqual('Language/script combination (eng_BadScript) is invalid or not supported (DetectionError.INVALID_PROPERTY)', str(cm.exception))

    def test_invalid_script_lang_combination(self):
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE']="spa"
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT']="Cyrl"

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Plaintext', 'test.txt', test_generic_job_props, {}, ff_track)
        # Reuse the shared component: each NllbTranslationComponent() loads its own
        # copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two copies do not
        # fit on a 16 GB card. Nothing here needs a fresh instance.
        comp = self.component

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_generic(job))
        self.assertEqual(mpf.DetectionError.INVALID_PROPERTY, cm.exception.error_code)
        self.assertEqual('Language/script combination (spa_Cyrl) is invalid or not supported (DetectionError.INVALID_PROPERTY)', str(cm.exception))

    def test_no_script_prop(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language but no script
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        # Verifies the default script is applied when none is supplied; the language
        # resolving is the subject, not the wording of the output.
        self.assertTranslated(result_props["TRANSLATION"], self.SAMPLE_0)

    def test_language_script_codes_case(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language but no script
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'DEU'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'LATN'

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        # Verifies language/script codes resolve case-insensitively.
        self.assertTranslated(result_props["TRANSLATION"], self.SAMPLE_0)

    def test_feed_forward_language(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0,
                                             LANGUAGE='deu',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        # Verifies the source language is taken from the feed-forward track.
        self.assertTranslated(result_props["TRANSLATION"], self.SAMPLE_0)

    def test_eng_to_eng_translation(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        ff_track = mpf.GenericTrack(-1, dict(TEXT='This is English text that should not be translated.',
                                             LANGUAGE='eng',
                                             ISO_SCRIPT='Latn'))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        # Same-language jobs short-circuit: the model is never invoked, so the text is
        # returned byte-identical. This is exact on every build precisely because no
        # model runs -- without the short-circuit NLLB paraphrases eng->eng into
        # "This is an English text...".
        self.assertEqual('This is English text that should not be translated.',
                         result_props["TRANSLATION"])

    def test_sentence_split_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        test_generic_job_props['USE_NLLB_TOKEN_LENGTH']='FALSE'
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
        # Verifies the text is split and every sentence is translated. Chunk wording is
        # model-dependent, so assert delivery rather than exact output.
        self.assertTranslated(result_props["TRANSLATION"], long_translation_text)

        test_generic_job_props['SOURCE_LANGUAGE'] = None
        test_generic_job_props['SENTENCE_MODEL_WTP_DEFAULT_ADAPTOR_LANGUAGE'] = 'en'
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertTranslated(result_props["TRANSLATION"], long_translation_text)
        # test sentence splitter (xx_sent_ud_sm)
        test_generic_job_props['SENTENCE_MODEL'] = 'xx_sent_ud_sm'
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertTranslated(result_props["TRANSLATION"], long_translation_text)

    def test_split_with_non_translate_segments(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'por'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        test_generic_job_props['USE_NLLB_TOKEN_LENGTH']='FALSE'
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
        translation = result_props["TRANSLATION"]
        # The subject here is that segments with nothing to translate pass through
        # VERBATIM; the wording of the surrounding translated segments is
        # model-dependent and deliberately not asserted.
        self.assertIn("012345678901234567890123456789012345.", translation)
        self.assertIn("123456789012345678901234567890123456.", translation)
        self.assertTranslated(translation, pt_text)

    def test_paragraph_split_job(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'por'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        test_generic_job_props['USE_NLLB_TOKEN_LENGTH']='FALSE'
        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'DEFAULT'
        test_generic_job_props['SENTENCE_SPLITTER_NEWLINE_BEHAVIOR'] = 'GUESS'
        test_generic_job_props['SENTENCE_MODEL'] = 'wtp-bert-mini'

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
        ff_track = mpf.GenericTrack(-1, dict(TEXT=pt_text))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable, to pour joy into the soul and send to the semblances the reflection of them; they imagine fatally pursued from _spleen_,  hopelessly gloomy and dreary, as if every moment they came out of the underground galleries of a pit-coal mine, How they deceive or how they intend to deceive us! is this an illusion or bad faith, against which there is much claim in vain the indelevel and accentuated expression of beatitude, which shines on the illuminated face of the men from beyond the Manch, who seem to walk among us, wrapped in dense atmosphere of perennial contentment, satisfied with the world, satisfied with men and, most of all, satisfied with themselves."
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)
        result_props: dict[str, str] = result_track[0].detection_properties
        # Verifies paragraph splitting produces a translation for the whole input.
        self.assertTranslated(result_props["TRANSLATION"], pt_text)

        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'SENTENCE'
        test_generic_job_props['SENTENCE_SPLITTER_NEWLINE_BEHAVIOR'] = 'GUESS'

        # In general, translation irregularities are more noticeable in SENTENCE splitting mode.
        # If the text fragment is too small, this can even include additional hallucinated text.
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable to pour joy into the soul and send to the countenances the reflection of them; They imagine themselves fatally haunted by spleen, hopelessly gloomy and sullen, as if at every moment they were emerging from the underground galleries of a pit-coal mine, Our British allies. How they deceive themselves or how they intend to deceive us! Is this an illusion or bad faith, against which there is much to be lamented in vain the indelevel and accentuated expression of beatitude, which shines through the illuminated faces of the men from beyond the Channel, who seem to walk among us, wrapped in a dense atmosphere of perennial contentment, satisfied with the world, satisfied with men and, very especially, satisfied with themselves? Yes , please ."
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)
        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertTranslated(result_props["TRANSLATION"], pt_text)


        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'DEFAULT'
        test_generic_job_props['SENTENCE_SPLITTER_NEWLINE_BEHAVIOR'] = 'NONE'
        pt_text_translation = "They fear, indeed, those in whom the vivid rays of our unblinking sun, or the unclouded face of the moon in the peninsular firmament, where it has not, like that of London--to break at the cost of a plumbeo heaven--are indispensable, to pour joy into the soul and send to the semblances the reflection of them; they imagine fatally pursued from _spleen_,  hopelessly gloomy and sullen, as if at every moment they were emerging from the subterranean galleries of a pit-coal mine, our British allies. How they deceive themselves or how they intend to deceive us! This is an illusion or bad faith, against which much is vainly complained the unlevel and accentuated expression of bliss, which shines through on the face. The European Parliament has been a great help to the people of Europe in the past, and it is a great help to us in the present."
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)
        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertTranslated(result_props["TRANSLATION"], pt_text)




    def test_wtp_with_flores_iso_lookup(self):
        #set default props
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)
        #load source language
        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'arz'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Arab'
        test_generic_job_props['USE_NLLB_TOKEN_LENGTH']='FALSE'
        test_generic_job_props['SENTENCE_SPLITTER_CHAR_COUNT'] = '100'
        test_generic_job_props['SENTENCE_SPLITTER_INCLUDE_INPUT_LANG'] = 'True'
        test_generic_job_props['PROCESS_DIFFICULT_LANGUAGES'] = "disabled"

        arz_text="هناك استياء بين بعض أعضاء جمعية ويلز الوطنية من الاقتراح بتغيير مسماهم الوظيفي إلى MWPs (أعضاء في برلمان ويلز). وقد نشأ ذلك بسبب وجود خطط لتغيير اسم الجمعية إلى برلمان ويلز."

        arz_text_translation = "Some members of the National Assembly for Wales were dissatisfied with the proposal to change their functional designation to MWPs (Members of the National Assembly for Wales). This arose from plans to change the name of the assembly to the Parliament of Wales."

        ff_track = mpf.GenericTrack(-1, dict(TEXT=arz_text))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        # Verifies a FLORES code with no direct wtpsplit equivalent (arz) still resolves
        # an adaptor language and splits; output wording is not the subject.
        self.assertTranslated(result_props["TRANSLATION"], arz_text)


    @unittest.skipIf(not RUN_DEEP_TESTS, "RUN_DEEP_TESTS is disabled. Please set RUN_DEEP_TESTS=True to evaluate a longer text sample (Recommended once per component update).")
    def test_long_spanish(self):
        # Excerpt of Dracula (Spanish):
        dracula_long_spa ='''
DRÁCULA

Bram Stoker

I. Del diario de Jonathan Harker
Bistritz, 3 de mayo

Salí de Munich a las 8:35 de la noche del primero de mayo, llegando a Viena temprano a la mañana siguiente; debí haber llegado a las 6:46, pero el tren llevaba una hora de retraso. Budapest parece un lugar maravilloso, según el vistazo que pude obtener desde el tren y el poco tiempo que caminé por sus calles. Temí alejarme demasiado de la estación, ya que llegamos tarde y saldríamos lo más cerca posible de la hora fijada.

La impresión que tuve fue que estábamos abandonando el Oeste y entrando en el Este; el más occidental de los espléndidos puentes sobre el Danubio, que aquí es de gran anchura y profundidad, nos condujo a las tradiciones del dominio turco.

Salimos con bastante buen tiempo, y llegamos después del anochecer a Klausenburg. Allí me detuve por la noche en el Hotel Royale. Para la cena, o más bien para la comida nocturna, tomé pollo preparado de algún modo con pimiento rojo, que estaba muy sabroso, pero me dio mucha sed. (Nota: obtener la receta para Mina.) Le pregunté al camarero, y me dijo que se llamaba "paprika hendl," y que, siendo un plato nacional, podría conseguirlo en cualquier lugar de los Cárpatos.

Mis escasos conocimientos de alemán me fueron muy útiles aquí; de hecho, no sé cómo me las habría arreglado sin ellos.

Como tuve algo de tiempo disponible cuando estuve en Londres, visité el Museo Británico e investigué en los libros y mapas de la biblioteca acerca de Transilvania; se me había ocurrido que cierto conocimiento previo del país difícilmente podría dejar de ser importante al tratar con un noble de esa región.

Descubrí que el distrito que él mencionó está en el extremo oriental del país, justo en las fronteras de tres estados: Transilvania, Moldavia y Bukovina, en medio de los montes Cárpatos; una de las partes más salvajes y menos conocidas de Europa.

No pude encontrar ningún mapa ni obra que indicara la localización exacta del castillo de Drácula, ya que no existen mapas en este país que puedan compararse en exactitud con nuestros mapas del Ordnance Survey; sin embargo, descubrí que Bistritz, el pueblo postal mencionado por el conde Drácula, es un lugar bastante conocido. Anotaré aquí algunas de mis notas, ya que podrían refrescar mi memoria cuando relate mis viajes a Mina.

En la población de Transilvania hay cuatro nacionalidades distintas: sajones en el sur, mezclados con los valacos, que son descendientes de los dacios; magiares al oeste y székelys al este y norte. Yo me dirijo hacia estos últimos, quienes afirman ser descendientes de Atila y los hunos. Esto podría ser cierto, ya que cuando los magiares conquistaron el país en el siglo XI encontraron asentados a los hunos.

He leído que todas las supersticiones conocidas del mundo se encuentran reunidas en la herradura de los Cárpatos, como si fuese el centro de una especie de torbellino imaginativo; si es así, mi estancia podría resultar muy interesante. (Nota: Debo preguntarle al conde todo acerca de ellas.)

No dormí bien, aunque mi cama era bastante cómoda, pues tuve toda clase de sueños extraños. Un perro estuvo aullando toda la noche bajo mi ventana, lo que podría haber tenido algo que ver; o quizás fue el paprika, pues tuve que beberme toda el agua de la jarra y aun así seguía sediento. Hacia la mañana logré dormir, y fui despertado por continuos golpes en mi puerta, por lo que supongo que entonces dormía profundamente.

Desayuné más paprika y una especie de gachas de harina de maíz que llamaban "mamaliga," y berenjena rellena de carne picada, un excelente plato que llaman "impletata." (Nota: conseguir también esta receta.)

Tuve que apresurar el desayuno, pues el tren salía poco antes de las ocho, o más bien debería haberlo hecho, ya que después de apresurarme a la estación a las 7:30 tuve que esperar en el vagón durante más de una hora antes de que comenzáramos a movernos.

Me parece que cuanto más al este se viaja, más impuntuales son los trenes. ¿Cómo serán entonces en China?

'''
        test_generic_job_props: dict[str, str] = dict(self.defaultProps)

        test_generic_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'spa'
        test_generic_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'

        text_translation = '''Dracula is dead. Bram Stoker is here . I would. From the diary of Jonathan Harker Bistritz, May 3 I left Munich at 8:35 p.m. on May 1, arriving in Vienna early the next morning; I should have arrived at 6:46, but the train was an hour late. Budapest seems like a wonderful place, from the view I was able to get from the train and the short time I walked through its streets. I was afraid to get too far from the station, since we arrived late and would leave as close as possible to the appointed time. The impression I had was that we were leaving the West and entering the East; the westernmost of the splendid bridges over the Danube, which here is of great width and depth, led us to the traditions of Turkish rule. We left in fairly good weather, and arrived after dark in Klausenburg. There I stopped for the night at the Hotel Royale. For dinner, or rather for the evening meal, I had chicken prepared somehow with red pepper, which was very tasty, but made me very thirsty. (Note: get the recipe for Mina.) I asked the waiter, and he told me that it was called "paprika hendl", and that, being a national dish, I could get it anywhere in the Carpathians. My limited knowledge of German came in very handy here; in fact, I don't know how I would have managed without it. Since I had some free time when I was in London, I visited the British Museum and researched the library books and maps about Transylvania; it had occurred to me that some prior knowledge of the country could hardly be less important when dealing with a nobleman of that region. I discovered that the district he mentioned is in the far eastern part of the country, right on the borders of three states - Transylvania, Moldavia, and Bukovina - in the middle of the Carpathian Mountains, one of the wildest and least known parts of Europe. I could not find any map or work indicating the exact location of Dracula's castle, as there are no maps in this country that can be accurately compared with our Ordnance Survey maps; however, I found that Bistritz, the postal town mentioned by Count Dracula, is a fairly well-known place. I'll write down some of my notes here, as they might refresh my memory when I recount my travels to Mina. In the population of Transylvania there are four distinct nationalities: Saxons in the south, mixed with the Vlachs, who are descendants of the Dacians; Magyars in the west; and Székelys in the east and north. I turn to the latter, who claim to be descendants of Attila and the Huns. This could be true, for when the Magyars conquered the country in the 11th century they found the Huns settled. I have read that all the known superstitions of the world are gathered in the Carpathian horseshoe, as if it were the center of a kind of imaginative whirlwind; if so, my stay could be very interesting. (Note: I must ask the Count everything about them.) I didn't sleep well, although my bed was quite comfortable, for I had all sorts of strange dreams. A dog was howling all night under my window, which might have had something to do with it; or perhaps it was the paprika, for I had to drink all the water from the jug and was still thirsty. By morning I managed to fall asleep, and I was awakened by continuous knocking on my door, so I guess I was then sound asleep. I had more paprika for breakfast and some kind of cornmeal porridge they called "mamaliga", and eggplant stuffed with minced meat, a great dish they call "impletata". (Note: get this recipe too.) I had to hurry up with breakfast, for the train left shortly before eight o'clock, or rather I should have, for after hurrying to the station at 7:30 I had to wait in the car for over an hour before we started moving. It seems to me that the farther east you travel, the more untimely the trains are. What will they be like in China then?'''
        ff_track = mpf.GenericTrack(-1, dict(TEXT=dracula_long_spa))
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        # Golden assertion: exact output, baselined on the gpu (float16) build.
        self.assertEqual(text_translation, result_props["TRANSLATION"])

        # By increasing the soft limit past recommended levels, translation quality drops sharply
        # -- output falls from ~3816 to ~1846 characters.
        #
        # This scenario MUST pin SENTENCE_SPLITTER_MODE=DEFAULT. The shipped default is now
        # SENTENCE, which puts one sentence in every chunk and therefore never consults the soft
        # limit; without this line both runs return identical text and the test asserts nothing.
        text_translation = '''DRACULA Bram Stoker I. From the diary of Jonathan Harker Bistritz, 3rd May I left Munich at 8:35 on the night of the first of May, arriving in Vienna early the next morning; I should have arrived at 6:46, but the train was an hour late. Budapest seems a wonderful place, from the view I could get from the train and the little time I walked through its streets. I was afraid to get too far from the station, as we arrived late and would leave as close as possible to the set time. The impression I had was that we were leaving the West and entering the East; the westernmost of the splendid bridges over the Danube, which here is of great width and depth, could lead us to the Discoveries of the Turkish dominion. We left in good time, and after a certain evening we arrived in Vienna. I stopped here for dinner at the Hotel Molotov, and the little time I walked through its streets. I could not find any map or work indicating the exact location of Dracula's castle, as there are no maps in this country that can be compared in accuracy with our Ordnance Survey maps; however, I discovered that Bistritz, the postal town mentioned by Count Dracula, is a fairly well-known place. I will write down some of my notes here, as they might refresh my memory when I relate my travels to Mina. In the population of Transylvania there are four distinct nationalities: Saxons in the south, mixed with the Valacs, who are descendants of the Dacians; Prussians in the west and Székelys in the east and north. I turn to these latter, as I could have claimed to be descendants of Attila and the Huns. This could be quite surprising, since when the Magyars conquered the country in the 11th century they found the Hungarians settled there. It seems to me that the farther east you travel, the more untimely the trains are. What will they be like in China then?'''
        test_generic_job_props['SENTENCE_SPLITTER_MODE'] = 'DEFAULT'
        test_generic_job_props['NLLB_TRANSLATION_TOKEN_SOFT_LIMIT'] = '512'
        job = mpf.GenericJob('Test Generic', 'test.pdf', test_generic_job_props, {}, ff_track)
        result_track: Sequence[mpf.GenericTrack] = self.component.get_detections_from_generic(job)

        result_props: dict[str, str] = result_track[0].detection_properties
        self.assertEqual(text_translation, result_props["TRANSLATION"])


    def test_difficult_language_token_limit_overrides_soft_limit_not_hard_limit(self):
        base_job_props: dict[str, str] = dict(self.defaultProps)
        base_job_props['DEFAULT_SOURCE_LANGUAGE'] = 'spa'
        base_job_props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        base_job_props['TARGET_LANGUAGE'] = 'fra'
        base_job_props['TARGET_SCRIPT'] = 'Latn'
        base_job_props['USE_NLLB_TOKEN_LENGTH'] = 'TRUE'
        base_job_props['NLLB_TRANSLATION_TOKEN_LIMIT'] = '200'
        base_job_props['NLLB_TRANSLATION_TOKEN_SOFT_LIMIT'] = '130'
        base_job_props['DIFFICULT_LANGUAGE_TOKEN_LIMIT'] = '50'

        text = (
            'Para la cena, o más bien para la comida nocturna, tomé pollo preparado '
            'de algún modo con pimiento rojo, que estaba muy sabroso, pero me dio '
            'mucha sed.'
        )

        config = JobConfig(base_job_props, ff_props={})
        # Reuse the shared component: each NllbTranslationComponent() loads its own
        # copy of the model, and at BUILD_TYPE=gpu (float16, 6.7 GB) two copies do not
        # fit on a 16 GB card. Nothing here needs a fresh instance.
        component = self.component
        component._check_model(config)
        component._load_tokenizer(config)

        source_token_count = component._tokenizer.count_tokens(text)
        self.assertLessEqual(source_token_count, 50)

        # Normal path: difficult-language handling disabled
        normal_ff_track = mpf.GenericTrack(-1, dict(TEXT=text))
        normal_props = dict(base_job_props)
        normal_props['PROCESS_DIFFICULT_LANGUAGES'] = 'disabled'
        normal_job = mpf.GenericJob('Test Generic', 'test.pdf', normal_props, {}, normal_ff_track)
        normal_result = component.get_detections_from_generic(normal_job)[0]
        normal_translation = normal_result.detection_properties["TRANSLATION"]

        normal_target_token_count = component._tokenizer.count_tokens(normal_translation)
        self.assertGreater(normal_target_token_count, 50)

        # Difficult-language path: Spanish explicitly treated as "difficult"
        difficult_ff_track = mpf.GenericTrack(-1, dict(TEXT=text))
        difficult_props = dict(base_job_props)
        difficult_props['PROCESS_DIFFICULT_LANGUAGES'] = 'spa'
        difficult_job = mpf.GenericJob('Test Generic', 'test.pdf', difficult_props, {}, difficult_ff_track)
        difficult_result = component.get_detections_from_generic(difficult_job)[0]
        difficult_translation = difficult_result.detection_properties["TRANSLATION"]

        difficult_target_token_count = component._tokenizer.count_tokens(difficult_translation)
        self.assertGreater(difficult_target_token_count, 50)

        # If difficult-language handling only overrides the soft limit, the output should match.
        self.assertEqual(normal_translation, difficult_translation)

    # ---- 7.2: tokenizer backend parity -----------------------------------------
    # The two backends must be interchangeable, because NLLB_TOKENIZER selects between
    # them at job level. Two divergences are KNOWN and asserted as expected rather than
    # treated as regressions -- see nllb_component/tokenizers.py.
    PARITY_CORPUS = [
        'Hallo, wie gehts Heute?',
        'Wie ist das Wetter?',
        'Es regnet.',
        'Isto e uma frase de teste, com acentuacao e pontuacao!',
        'Ich habe 3,50 EUR bezahlt - z.B. fuer Kaffee.',
    ]

    def _both_backends(self):
        from nllb_component import tokenizers
        model_dir = os.path.join('/models', self.component._current_model_name)
        try:
            hf = tokenizers.create_backend(tokenizers.HUGGINGFACE, model_dir)
        except Exception as e:
            self.skipTest(f'HuggingFace backend unavailable: {e}')
        return tokenizers.create_backend(tokenizers.SENTENCEPIECE, model_dir), hf

    def test_tokenizer_backends_agree_on_token_counts(self):
        # count_tokens drives sentence splitting, so disagreement here would move chunk
        # boundaries when the backend changes. This is the parity that must be exact.
        sp, hf = self._both_backends()
        for text in self.PARITY_CORPUS:
            with self.subTest(text=text):
                self.assertEqual(sp.count_tokens(text), hf.count_tokens(text))

    def test_tokenizer_backends_produce_ct2_token_strings(self):
        # CTranslate2 consumes token STRINGS, and requires the source-language token
        # first and </s> last. Both backends must satisfy that shape.
        sp, hf = self._both_backends()
        for backend in (sp, hf):
            with self.subTest(backend=backend.name):
                encoded = backend.encode(self.PARITY_CORPUS, 'deu_Latn')
                self.assertEqual(len(encoded), len(self.PARITY_CORPUS))
                for tokens in encoded:
                    self.assertEqual('deu_Latn', tokens[0])
                    self.assertEqual('</s>', tokens[-1])
                    self.assertTrue(all(isinstance(t, str) for t in tokens))

    def test_tokenizer_backends_agree_except_on_unknown_characters(self):
        # Known divergence: a character absent from the vocabulary (e.g. an em dash) is
        # emitted by SentencePiece as its raw surface form and by HuggingFace as <unk>.
        # Token COUNTS still match and CTranslate2 maps both to <unk>, so translations
        # are unaffected. Asserted as expected so it cannot be mistaken for a regression.
        sp, hf = self._both_backends()
        plain = 'Wie ist das Wetter?'
        self.assertEqual(sp.encode([plain], 'deu_Latn'), hf.encode([plain], 'deu_Latn'))

        em_dash = 'Ich habe bezahlt \u2014 zum Beispiel.'
        sp_tokens = sp.encode([em_dash], 'deu_Latn')[0]
        hf_tokens = hf.encode([em_dash], 'deu_Latn')[0]
        self.assertEqual(len(sp_tokens), len(hf_tokens))
        self.assertIn('\u2014', sp_tokens)
        self.assertIn('<unk>', hf_tokens)

    def test_tokenizer_round_trips(self):
        # decode(encode(x)) should recover the text, modulo the language/EOS markers the
        # encoder adds. Guards the decode path each backend implements differently.
        sp, _ = self._both_backends()
        encoded = sp.encode(self.PARITY_CORPUS, 'deu_Latn')
        stripped = [[t for t in toks if t not in ('deu_Latn', '</s>')] for toks in encoded]
        for original, decoded in zip(self.PARITY_CORPUS, sp.decode(stripped)):
            with self.subTest(text=original):
                self.assertEqual(original, decoded)

    def test_unrecognised_tokenizer_falls_back(self):
        # An unusable NLLB_TOKENIZER should not fail the job -- it warns and uses the
        # default, because a tokenizer typo is not worth losing a translation over.
        from nllb_component import tokenizers
        model_dir = os.path.join('/models', self.component._current_model_name)
        self.assertEqual(tokenizers.SENTENCEPIECE,
                         tokenizers.create_backend('NOT-A-BACKEND', model_dir).name)

    # ---- 7.3: NLLB_MODEL is honoured -------------------------------------------
    def test_nllb_model_property_is_honoured(self):
        # Regression guard for the defect that made NLLB_MODEL a no-op: _check_model
        # only tested model_is_loaded, which is never false for a live Translator, so
        # the baked-in default served every request regardless of the job property.
        alt = '/models/nllb-model-swap-test'
        real = os.path.join('/models', self.component._current_model_name)
        original = self.component._current_model_name
        if not os.path.isdir(alt):
            try:
                os.makedirs(alt, exist_ok=True)
                for f in os.listdir(real):
                    link = os.path.join(alt, f)
                    if not os.path.exists(link):
                        os.symlink(os.path.join(real, f), link)
            except OSError as e:
                self.skipTest(f'cannot stage an alternate model directory: {e}')
        try:
            props = dict(self.defaultProps)
            props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
            props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
            props['NLLB_MODEL'] = 'nllb-model-swap-test'
            ff_track = mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))
            job = mpf.GenericJob('Test Generic', 'test.pdf', props, {}, ff_track)
            result = self.component.get_detections_from_generic(job)[0]

            self.assertEqual('nllb-model-swap-test', self.component._current_model_name)
            self.assertTranslated(result.detection_properties['TRANSLATION'], self.SAMPLE_0)
        finally:
            # Restore the shared component for the rest of the suite.
            restore = dict(self.defaultProps)
            restore['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
            restore['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
            restore['NLLB_MODEL'] = original
            self.component.get_detections_from_generic(
                mpf.GenericJob('Restore', 'test.pdf', restore, {},
                               mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))))

    def test_unknown_nllb_model_raises_instead_of_falling_back(self):
        # A bad model name must fail loudly. Silently serving a different model is how
        # the original bug stayed hidden through a whole evaluation run.
        original = self.component._current_model_name
        props = dict(self.defaultProps)
        props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        props['NLLB_MODEL'] = 'this-model-does-not-exist'
        job = mpf.GenericJob('Test Generic', 'test.pdf', props, {},
                             mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0)))
        try:
            with self.assertRaises(mpf.DetectionException) as cm:
                self.component.get_detections_from_generic(job)
            self.assertEqual(mpf.DetectionError.COULD_NOT_READ_DATAFILE, cm.exception.error_code)
            # After a failed load the component must not claim to hold a model.
            self.assertIsNone(self.component._current_model_name)
        finally:
            restore = dict(self.defaultProps)
            restore['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
            restore['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
            restore['NLLB_MODEL'] = original
            self.component.get_detections_from_generic(
                mpf.GenericJob('Restore', 'test.pdf', restore, {},
                               mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))))

    # ---- job properties added for the CTranslate2 engine ------------------------
    # These are tested by asserting the value REACHES CTranslate2, not by diffing
    # translations. Output-diffing is unreliable here for two independent reasons:
    # easy input does not discriminate (beam 1 and beam 4 agree on short sentences),
    # and the gpu/cpu builds legitimately word things differently. Checking the call
    # is deterministic on both.

    class _RecordingTranslator:
        """Forwards to the real Translator while recording translate_batch kwargs."""

        def __init__(self, inner):
            self._inner = inner
            self.calls = []

        def translate_batch(self, *args, **kwargs):
            self.calls.append(kwargs)
            return self._inner.translate_batch(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _record_translate_batch(self, props, text=None):
        job = mpf.GenericJob('Test Generic', 'test.pdf', props, {},
                             mpf.GenericTrack(-1, dict(TEXT=text or self.SAMPLE_0)))
        original = self.component._model
        spy = self._RecordingTranslator(original)
        self.component._model = spy
        try:
            self.component.get_detections_from_generic(job)
        finally:
            self.component._model = original
        self.assertTrue(spy.calls, 'translate_batch was never called')
        return spy.calls[-1]

    def _decode_props(self, **overrides):
        props = dict(self.defaultProps)
        props['DEFAULT_SOURCE_LANGUAGE'] = 'deu'
        props['DEFAULT_SOURCE_SCRIPT'] = 'Latn'
        props.update(overrides)
        return props

    def test_decode_properties_reach_ctranslate2(self):
        kwargs = self._record_translate_batch(self._decode_props(
            NLLB_BEAM_SIZE='2',
            NLLB_MAX_BATCH_SIZE='512',
            NLLB_BATCH_TYPE='examples'))
        self.assertEqual(2, kwargs['beam_size'])
        self.assertEqual(512, kwargs['max_batch_size'])
        self.assertEqual('examples', kwargs['batch_type'])

    def test_decode_property_defaults_reach_ctranslate2(self):
        # Guards the defaults themselves, so a job that sets nothing still gets the
        # decoding the evaluation measured (beam 4 in particular).
        kwargs = self._record_translate_batch(self._decode_props())
        self.assertEqual(4, kwargs['beam_size'])
        self.assertEqual(2024, kwargs['max_batch_size'])
        self.assertEqual('tokens', kwargs['batch_type'])

    def test_max_decoding_length_follows_the_token_limit(self):
        # The chunk limit doubles as CTranslate2's max_decoding_length; if these ever
        # drift apart the model could be allowed to emit more than a chunk's budget.
        kwargs = self._record_translate_batch(self._decode_props(
            USE_NLLB_TOKEN_LENGTH='TRUE', NLLB_TRANSLATION_TOKEN_LIMIT='321'))
        self.assertEqual(321, kwargs['max_decoding_length'])

    def test_nllb_tokenizer_property_selects_the_backend(self):
        from nllb_component import tokenizers
        for requested, expected in ((tokenizers.HUGGINGFACE, tokenizers.HUGGINGFACE),
                                    (tokenizers.SENTENCEPIECE, tokenizers.SENTENCEPIECE),
                                    ('nonsense', tokenizers.SENTENCEPIECE)):
            with self.subTest(requested=requested):
                job = mpf.GenericJob(
                    'Test Generic', 'test.pdf',
                    self._decode_props(NLLB_TOKENIZER=requested), {},
                    mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0)))
                try:
                    self.component.get_detections_from_generic(job)
                except mpf.DetectionException as e:
                    self.skipTest(f'backend {requested} unavailable: {e}')
                self.assertEqual(expected, self.component._tokenizer.name)
        # leave the shared component on the default backend
        self.component.get_detections_from_generic(mpf.GenericJob(
            'Restore', 'test.pdf', self._decode_props(), {},
            mpf.GenericTrack(-1, dict(TEXT=self.SAMPLE_0))))

    def test_threading_properties_are_parsed_and_reported_when_unappliable(self):
        # Threading is fixed when the Translator is constructed, so a job asking for
        # different values cannot be honoured. It must say so rather than silently
        # ignoring the request -- that silence is exactly how the NLLB_MODEL bug hid.
        config = JobConfig(self._decode_props(NLLB_INTER_THREADS='3',
                                              NLLB_INTRA_THREADS='7'), ff_props={})
        self.assertEqual(3, config.nllb_inter_threads)
        self.assertEqual(7, config.nllb_intra_threads)

        with self.assertLogs('NllbTranslationComponent', level='WARNING') as logs:
            self.component._check_model(config)
        self.assertTrue(any('INTRA_THREADS' in line for line in logs.output),
                        f'no threading warning logged: {logs.output}')

    def test_job_config_defaults_match_the_descriptor(self):
        # The descriptor is the documented contract. If a JobConfig default drifts from
        # its defaultValue, the documentation is silently wrong for anyone who does not
        # set the property explicitly.
        minimal = {'DEFAULT_SOURCE_LANGUAGE': 'deu', 'DEFAULT_SOURCE_SCRIPT': 'Latn'}
        config = JobConfig(minimal, ff_props={})
        documented = {p['name']: p['defaultValue'] for p in self.descriptorProperties}
        attr_for = {
            'NLLB_MODEL': 'nllb_model',
            'NLLB_TOKENIZER': 'nllb_tokenizer',
            'NLLB_BEAM_SIZE': 'nllb_beam_size',
            'NLLB_MAX_BATCH_SIZE': 'nllb_max_batch_size',
            'NLLB_BATCH_TYPE': 'nllb_batch_type',
            'NLLB_INTER_THREADS': 'nllb_inter_threads',
            'NLLB_INTRA_THREADS': 'nllb_intra_threads',
            'NLLB_TRANSLATION_TOKEN_LIMIT': 'nllb_token_limit',
            'NLLB_TRANSLATION_TOKEN_SOFT_LIMIT': 'nllb_token_soft_limit',
            'SENTENCE_SPLITTER_CHAR_COUNT': 'nllb_character_limit',
            'DIFFICULT_LANGUAGE_TOKEN_LIMIT': 'difficult_language_token_limit',
        }
        for prop, attr in attr_for.items():
            with self.subTest(property=prop):
                self.assertIn(prop, documented, f'{prop} is not in descriptor.json')
                actual = getattr(config, attr)
                # Compare in the type the component actually uses.
                self.assertEqual(type(actual)(documented[prop]), actual)

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

    def test_wtp_iso_conversion(self):
        # checks ISO normalization and WTP ("Where's The Point" Sentence Splitter) lookup
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ace_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ace_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('acm_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('acq_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('aeb_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('afr_Latn'), 'af')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ajp_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('amh_Ethi'), 'am')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('apc_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('arb_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ars_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ary_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('arz_Arab'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('asm_Beng'), 'bn')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ast_Latn'), 'es')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('awa_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ayr_Latn'), 'es')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('azb_Arab'), 'az')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('azj_Latn'), 'az')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bak_Cyrl'), 'ru')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bam_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ban_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bel_Cyrl'), 'be')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ben_Beng'), 'bn')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bho_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bjn_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bug_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('bul_Cyrl'), 'bg')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('cat_Latn'), 'ca')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ceb_Latn'), 'ceb')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ces_Latn'), 'cs')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('cjk_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ckb_Arab'), 'ku')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('crh_Latn'), 'tr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('cym_Latn'), 'cy')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('dan_Latn'), 'da')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('deu_Latn'), 'de')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('dik_Latn'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('dyu_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ell_Grek'), 'el')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('eng_Latn'), 'en')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('epo_Latn'), 'eo')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('est_Latn'), 'et')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('eus_Latn'), 'eu')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('fin_Latn'), 'fi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('fon_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('fra_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('fur_Latn'), 'it')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('fuv_Latn'), 'ha')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('gla_Latn'), 'gd')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('gle_Latn'), 'ga')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('glg_Latn'), 'gl')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('grn_Latn'), 'es')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('guj_Gujr'), 'gu')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('hat_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('hau_Latn'), 'ha')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('heb_Hebr'), 'he')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('hin_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('hne_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('hun_Latn'), 'hu')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('hye_Armn'), 'hy')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ibo_Latn'), 'ig')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ind_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('isl_Latn'), 'is')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ita_Latn'), 'it')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('jav_Latn'), 'jv')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('jpn_Jpan'), 'ja')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kab_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kac_Latn'), 'my')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kan_Knda'), 'kn')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kas_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kat_Geor'), 'ka')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kbp_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kea_Latn'), 'pt')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('khm_Khmr'), 'km')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('khk_Cyrl'), 'mn')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kir_Cyrl'), 'ky')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kmb_Latn'), 'pt')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kmr_Latn'), 'ku')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('knc_Latn'), 'ha')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kon_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('kor_Hang'), 'ko')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lij_Latn'), 'it')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lim_Latn'), 'nl')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lin_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lit_Latn'), 'lt')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lmo_Latn'), 'it')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ltg_Latn'), 'lv')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lua_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lus_Latn'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('lvs_Latn'), 'lv')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mag_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mai_Deva'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mal_Mlym'), 'ml')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mar_Deva'), 'mr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('min_Latn'), 'id')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mkd_Cyrl'), 'mk')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mlt_Latn'), 'mt')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mni_Beng'), 'bn')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mos_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('mya_Mymr'), 'my')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('nld_Latn'), 'nl')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('nno_Latn'), 'no')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('nob_Latn'), 'no')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('npi_Deva'), 'ne')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('nus_Latn'), 'ar')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('pan_Guru'), 'pa')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('pap_Latn'), 'es')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('pbt_Arab'), 'ps')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('pes_Arab'), 'fa')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('plt_Latn'), 'mg')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('pol_Latn'), 'pl')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('por_Latn'), 'pt')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('prs_Arab'), 'fa')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ron_Latn'), 'ro')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('rus_Cyrl'), 'ru')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('sag_Latn'), 'fr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('sat_Olck'), 'hi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('scn_Latn'), 'it')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('shn_Mymr'), 'my')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('sin_Sinh'), 'si')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('slk_Latn'), 'sk')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('slv_Latn'), 'sl')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('spa_Latn'), 'es')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('als_Latn'), 'sq')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('srp_Cyrl'), 'sr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('swe_Latn'), 'sv')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('szl_Latn'), 'pl')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('tam_Taml'), 'ta')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('tel_Telu'), 'te')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('tgk_Cyrl'), 'tg')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('tha_Thai'), 'th')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('tur_Latn'), 'tr')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ukr_Cyrl'), 'uk')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('umb_Latn'), 'pt')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('urd_Arab'), 'ur')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('uzn_Latn'), 'uz')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('vec_Latn'), 'it')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('vie_Latn'), 'vi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('xho_Latn'), 'xh')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('ydd_Hebr'), 'yi')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('yor_Latn'), 'yo')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('yue_Hant'), 'zh')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('zho_Hans'), 'zh')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('zsm_Latn'), 'ms')
        self.assertEqual(WtpLanguageSettings.convert_to_iso('zul_Latn'), 'zu')

        # languages supported by NLLB but not supported by WTP Splitter
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('aka_Latn')) # 'ak'  Akan
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('bem_Latn')) # 'sw'  Bemba
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('bod_Tibt')) # 'bo'  Tibetan
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('bos_Latn')) # 'bs'  Bosnian
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('dzo_Tibt')) # 'dz'  Dzongkha
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('ewe_Latn')) # 'ee'  Ewe
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('fao_Latn')) # 'fo'  Faroese
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('fij_Latn')) # 'fj'  Fijian
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('gaz_Latn')) # 'om'  Oromo
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('hrv_Latn')) # 'hr'  Croatian
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('ilo_Latn')) # 'tl'  Ilocano
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('kam_Latn')) # 'sw'  Kamba
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('kik_Latn')) # 'sw'  Kikuyu
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('kin_Latn')) # 'rw'  Kinyarwanda
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('lao_Laoo')) # 'lo'  Lao
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('ltz_Latn')) # 'lb'  Luxembourgish
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('lug_Latn')) # 'lg'  Ganda
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('luo_Latn')) # 'luo' Luo
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('mri_Latn')) # 'mi'  Maori
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('nso_Latn')) # 'st'  Northern Sotho
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('nya_Latn')) # 'ny'  Chichewa
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('oci_Latn')) # 'oc'  Occitan
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('ory_Orya')) # 'or'  Odia
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('pag_Latn')) # 'tl'  Pangasinan
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('quy_Latn')) # 'qu'  Quechua
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('run_Latn')) # 'rn'  Rundi
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('san_Deva')) # 'sa'  Sanskrit
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('smo_Latn')) # 'sm'  Samoan
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('sna_Latn')) # 'sn'  Shona
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('snd_Arab')) # 'sd'  Sindhi
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('som_Latn')) # 'so'  Somali
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('sot_Latn')) # 'st'  Southern Sotho
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('srd_Latn')) # 'sc'  Sardinian
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('ssw_Latn')) # 'ss'  Swati
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('sun_Latn')) # 'su'  Sundanese
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('swh_Latn')) # 'sw'  Swahili
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('taq_Latn')) # 'ber' Tamasheq
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tat_Cyrl')) # 'tt'  Tatar
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tgl_Latn')) # 'tl'  Tagalog
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tir_Ethi')) # 'ti'  Tigrinya
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tpi_Latn')) # 'tpi' Tok Pisin
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tsn_Latn')) # 'tn'  Tswana
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tso_Latn')) # 'ts'  Tsonga
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tuk_Latn')) # 'tk'  Turkmen
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tum_Latn')) # 'ny'  Tumbuka
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('twi_Latn')) # 'ak'  Twi
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('tzm_Tfng')) # 'ber' Central Atlas Tamazight (Berber)
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('uig_Arab')) # 'ug'  Uyghur
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('war_Latn')) # 'tl'  Waray
        self.assertIsNone(WtpLanguageSettings.convert_to_iso('wol_Latn')) # 'wo'  Wolof

if __name__ == '__main__':
    unittest.main()
