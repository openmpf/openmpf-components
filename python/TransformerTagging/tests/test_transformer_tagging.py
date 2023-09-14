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

from transformer_tagging_component import TransformerTaggingComponent

LOCAL_PATH = Path(__file__).parent
sys.path.insert(0, str(LOCAL_PATH.parent))
TEST_DATA = LOCAL_PATH / 'data'

SHORT_SAMPLE = (
    'I drove to the beach today and will be staying overnight at a hotel. '
    'I texted my friend before I left so she could look after my cats. '
    'She will drop by to check on them after stopping by the bank. '
    'I plan to spend all day at the beach tomorrow.'
)

SHORT_SAMPLE_TAGS = "TRAVEL"
SHORT_SAMPLE_TRIGGER_SENTENCES = "I drove to the beach today and will be staying overnight at a hotel."
SHORT_SAMPLE_OFFSET = "0-67"
SHORT_SAMPLE_SCORE = "0.4680028557777405"


class TestArgosTranslation(unittest.TestCase):

    def test_generic_job(self):
        ff_track = mpf.GenericTrack(-1, dict(TEXT=SHORT_SAMPLE))
        job = mpf.GenericJob('Test Generic', 'test.pdf', {}, {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

    def test_plaintext_job(self):
        job = mpf.GenericJob('Test Plaintext', str(TEST_DATA / 'simple_input.txt'), {}, {})
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

    def test_audio_job(self):
        ff_track = mpf.AudioTrack(0, 1, -1, dict(TEXT=SHORT_SAMPLE))
        job = mpf.AudioJob('Test Audio', 'test.wav', 0, 1, {}, {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

    def test_image_job(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_SAMPLE))
        job = mpf.ImageJob('Test Image', 'test.jpg', {}, {}, ff_loc)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

    def test_video_job(self):
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_SAMPLE)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=SHORT_SAMPLE))
            },
            dict(TEXT=SHORT_SAMPLE))
        job = mpf.VideoJob('Test Video', 'test.mp4', 0, 1, {}, {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_video(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties
        self.assertEqual(SHORT_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

        frame_1_props = result[0].frame_locations[0].detection_properties
        self.assertEqual(SHORT_SAMPLE_TAGS, frame_1_props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, frame_1_props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, frame_1_props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, frame_1_props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

        frame_2_props = result[0].frame_locations[1].detection_properties
        self.assertEqual(SHORT_SAMPLE_TAGS, frame_2_props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, frame_2_props["TRANSCRIPT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, frame_2_props["TRANSCRIPT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, frame_2_props["TRANSCRIPT TRAVEL TRIGGER SENTENCES SCORE"])

    def test_no_feed_forward_location(self):
        comp = TransformerTaggingComponent()
        job = mpf.ImageJob('Test', 'test.jpg', {}, {})

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_image(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

    def test_no_feed_forward_track(self):
        comp = TransformerTaggingComponent()
        job = mpf.VideoJob('test', 'test.mp4', 0, 1, {}, {})

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_video(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

        job = mpf.AudioJob('Test Audio', 'test.wav', 0, 1, {}, {})
        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_audio(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

    def test_custom_confidence_threshold(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_SAMPLE))
        job = mpf.ImageJob('Test Image', 'test.jpg', dict(SCORE_THRESHOLD=".2"), {}, ff_loc)

        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual("TRAVEL; FINANCIAL", props["TAGS"])
        self.assertEqual(SHORT_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(SHORT_SAMPLE_SCORE, props["TEXT TRAVEL TRIGGER SENTENCES SCORE"])

        custom_threshold_sentence = "She will drop by to check on them after stopping by the bank."
        custom_threshold_sentence_offset = "135-195"
        custom_threshold_sentence_score = "0.2906474769115448"

        self.assertEqual(custom_threshold_sentence, props["TEXT FINANCIAL TRIGGER SENTENCES"])
        self.assertEqual(custom_threshold_sentence_offset, props["TEXT FINANCIAL TRIGGER SENTENCES OFFSET"])
        self.assertEqual(custom_threshold_sentence_score, props["TEXT FINANCIAL TRIGGER SENTENCES SCORE"])

    def test_custom_tagging_file(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_SAMPLE))
        job = mpf.ImageJob('Test Image', 'test.jpg',
                           dict(TRANSFORMER_TAGGING_CORPUS="config/custom_corpus.json"), {}, ff_loc)

        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))

    def test_debugging_show_matches(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_SAMPLE))
        job = mpf.ImageJob('Test Image', 'test.jpg', {}, {}, ff_loc)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))
        props = result[0].detection_properties
        self.assertTrue("TEXT TRAVEL TRIGGER SENTENCES MATCHES" not in props)

        job = mpf.ImageJob('Test Image', 'test.jpg', dict(ENABLE_DEBUG="TRUE"), {}, ff_loc)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))
        props = result[0].detection_properties
        self.assertTrue("TEXT TRAVEL TRIGGER SENTENCES MATCHES" in props)
        self.assertEqual("This sentence is hotel.", props["TEXT TRAVEL TRIGGER SENTENCES MATCHES"])

    def test_missing_property_to_process(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(INPUT="some input"))
        job = mpf.ImageJob('Test Image', 'test.jpg', {}, {}, ff_loc)

        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))

        self.assertEqual(ff_loc.x_left_upper, result[0].x_left_upper)
        self.assertEqual(ff_loc.y_left_upper, result[0].y_left_upper)
        self.assertEqual(ff_loc.width, result[0].width)
        self.assertEqual(ff_loc.height, result[0].height)
        self.assertEqual(ff_loc.confidence, result[0].confidence)
        self.assertEqual(ff_loc.detection_properties, result[0].detection_properties)

    def test_missing_text_to_process(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=""))
        job = mpf.ImageJob('Test Image', 'test.jpg', {}, {}, ff_loc)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))


if __name__ == '__main__':
    unittest.main()
