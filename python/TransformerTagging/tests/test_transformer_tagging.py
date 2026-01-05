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

from pathlib import Path
import logging
import unittest

import mpf_component_api as mpf

from transformer_tagging_component import TransformerTaggingComponent


TEST_DATA = Path(__file__).parent / 'data'
TEST_CONFIG = Path(__file__).parent / 'config'

logging.basicConfig(level=logging.DEBUG)

SHORT_BEACH_SAMPLE = (
    'I drove to the beach today and will be staying overnight at a hotel. '
    'I texted my friend before I left so she could look after my cats. '
    'She will drop by to check on them after stopping by the bank. '
    'I plan to spend all day at the beach tomorrow.'
)
SHORT_BEACH_SAMPLE_TAGS = "TRAVEL"
SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES = "I drove to the beach today and will be staying overnight at a hotel."
SHORT_BEACH_SAMPLE_OFFSET = "0-67"
SHORT_BEACH_SAMPLE_SCORE = 0.4680028557777405

SHORT_PARK_SAMPLE = 'I rode my bike to the park.'
SHORT_PARK_SAMPLE_TAGS = "VEHICLE"
SHORT_PARK_SAMPLE_TRIGGER_SENTENCES = "I rode my bike to the park."
SHORT_PARK_SAMPLE_OFFSET = "0-26"
SHORT_PARK_SAMPLE_SCORE = 0.47114357352256775


class TestTransformerTagging(unittest.TestCase):

    def test_generic_job(self):
        ff_track = mpf.GenericTrack(-1, dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.GenericJob('Test Generic', 'test.pdf',
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                             {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)


    def test_plaintext_job(self):
        job = mpf.GenericJob('Test Plaintext',
                             str(TEST_DATA / 'simple_input.txt'), 
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                             {})
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)


    def test_audio_job(self):
        ff_track = mpf.AudioTrack(0, 1, -1, dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.AudioJob('Test Audio',
                           'test.wav', 0, 1,
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_audio(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)


    def test_image_job(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {}, ff_loc)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)


    def test_video_job(self):
        ff_track = mpf.VideoTrack(
            0, 1, -1,
            {
                0: mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_BEACH_SAMPLE)),
                1: mpf.ImageLocation(0, 10, 10, 10, -1, dict(TRANSCRIPT=SHORT_BEACH_SAMPLE))
            },
            dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.VideoJob('Test Video',
                           'test.mp4', 0, 1,
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_video(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties
        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)

        frame_1_props = result[0].frame_locations[0].detection_properties
        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, frame_1_props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, frame_1_props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, frame_1_props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)

        frame_2_props = result[0].frame_locations[1].detection_properties
        self.assertEqual(SHORT_BEACH_SAMPLE_TAGS, frame_2_props["TAGS"])
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, frame_2_props["TRANSCRIPT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, frame_2_props["TRANSCRIPT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(frame_2_props["TRANSCRIPT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)


    def test_no_feed_forward_location(self):
        comp = TransformerTaggingComponent()
        job = mpf.ImageJob('Test',
                           'test.jpg',
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {})

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_image(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)


    def test_no_feed_forward_track(self):
        comp = TransformerTaggingComponent()
        job = mpf.VideoJob('test',
                           'test.mp4', 0, 1,
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {})

        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_video(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)

        job = mpf.AudioJob('Test Audio', 'test.wav', 0, 1, {}, {})
        with self.assertRaises(mpf.DetectionException) as cm:
            list(comp.get_detections_from_audio(job))
        self.assertEqual(mpf.DetectionError.UNSUPPORTED_DATA_TYPE, cm.exception.error_code)


    def test_custom_score_threshold(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.ImageJob('Test Image',
                           'test.jpg', 
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json"),
                                SCORE_THRESHOLD=".2"),
                           {}, ff_loc)

        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual("FINANCIAL; TRAVEL", props["TAGS"]) # tags in lexicographic order
        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)

        custom_threshold_sentence = "She will drop by to check on them after stopping by the bank."
        custom_threshold_sentence_offset = "135-195"
        custom_threshold_sentence_score = 0.2906474769115448

        self.assertEqual(custom_threshold_sentence, props["TEXT FINANCIAL TRIGGER SENTENCES"])
        self.assertEqual(custom_threshold_sentence_offset, props["TEXT FINANCIAL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(custom_threshold_sentence_score, float(props["TEXT FINANCIAL TRIGGER SENTENCES SCORE"]), places=3)


    def test_debugging_show_matches(self):
        ff_loc = mpf.ImageLocation(0, 0, 10, 10, -1, dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {}, ff_loc)
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
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {}, ff_loc)

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
        job = mpf.ImageJob('Test Image',
                           'test.jpg',
                           dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                           {}, ff_loc)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_image(job)

        self.assertEqual(1, len(result))


    def test_maintain_tags_from_earlier_feedforward_task(self):
        ff_track = mpf.GenericTrack(-1, dict(TEXT=SHORT_BEACH_SAMPLE))
        job = mpf.GenericJob('Test Generic',
                             'test.pdf',
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                             {}, ff_track)

        job.feed_forward_track.detection_properties["TAGS"] = "VEL; TRAVEL MAP; ALPHA"
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties
        expected_tags = "ALPHA; TRAVEL; TRAVEL MAP; VEL" # tags in lexicographic order

        self.assertEqual(expected_tags, props["TAGS"])


    def test_matches_with_semicolons(self):
        SEMICOLON_SAMPLE = (
            'I rode my bike to the beach today; it was a long trip. '
        )
        ff_track = mpf.GenericTrack(-1, dict(TEXT=SEMICOLON_SAMPLE))
        job = mpf.GenericJob('Test Generic',
                             'test.pdf',
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                             {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))
        props = result[0].detection_properties

        expected_output = "I rode my bike to the beach today[;] it was a long trip."
        self.assertEqual(expected_output, props["TEXT VEHICLE TRIGGER SENTENCES"])


    def test_repeat_trigger_job(self):
        sample = (
            'I drove to the beach today and will be staying overnight at a hotel. '
            'I drove to the beach today and will be staying overnight at a hotel. '
            'I texted my friend before I left so she could look after my cats. '
            'I am going to the airport tomorrow. '
            'I plan to spend all day at the beach tomorrow. '
            'This airline serves peanuts. '
            'I am going to the airport tomorrow. '
        )

        trigger_sentences = (
            'I drove to the beach today and will be staying overnight at a hotel.; '
            'I am going to the airport tomorrow.; '
            'This airline serves peanuts.'
        )

        offsets = "0-67, 69-136; 204-238, 316-350; 287-314"

        score_1 = 0.4680027663707733
        score_2 = 0.5079247951507568
        score_3 = 0.5265363454818726

        matches = (
            'This sentence is hotel.; '
            'This sentence is airport.; '
            'This sentence is airline.'
        )

        ff_track = mpf.GenericTrack(-1, dict(TEXT=sample))
        job = mpf.GenericJob('Test Repeat',
                             'test.pdf',
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json"),
                                  SCORE_THRESHOLD='0.4',
                                  ENABLE_DEBUG='true'),
                             {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties

        self.assertEqual("TRAVEL", props["TAGS"])
        self.assertEqual(trigger_sentences, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(offsets, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])

        score_result_1, score_result_2, score_result_3 = props["TEXT TRAVEL TRIGGER SENTENCES SCORE"].split(";")
        self.assertAlmostEqual(score_1, float(score_result_1), places=3)
        self.assertAlmostEqual(score_2, float(score_result_2), places=3)
        self.assertAlmostEqual(score_3, float(score_result_3), places=3)

        self.assertAlmostEqual(matches, props["TEXT TRAVEL TRIGGER SENTENCES MATCHES"])


    def test_newline_split(self):
        sample = (
            'This first sentence is about driving to the beach\n'
            'Another sentence about driving to the beach\n\n\n\n'
            'This sentence is also about driving to the beach and ends in a period.\n'
            '   Beach sentence begins and ends with \t whitespace   \n'
            'Final beach sentence!'
        )

        trigger_sentences = (
            'This first sentence is about driving to the beach; '
            'Another sentence about driving to the beach; '
            'This sentence is also about driving to the beach and ends in a period.; '
            'Beach sentence begins and ends with \t whitespace; '
            'Final beach sentence!'
        )

        offsets = '0-48; 50-92; 97-166; 171-218; 223-243'

        ff_track = mpf.GenericTrack(-1, dict(TEXT=sample))
        job = mpf.GenericJob('Test Generic',
                             'test.txt',
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json"),
                                  ENABLE_NEWLINE_SPLIT='true',
                                  ENABLE_DEBUG='true'),
                             {}, ff_track)
        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        props = result[0].detection_properties

        self.assertEqual(trigger_sentences, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(offsets, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])


    def test_process_all_properties(self):
        ff_track = mpf.GenericTrack(-1,
                                    dict(TEXT=SHORT_BEACH_SAMPLE,
                                         TRANSLATION=SHORT_PARK_SAMPLE,
                                         TRANSCRIPT=SHORT_BEACH_SAMPLE + ' ' + SHORT_PARK_SAMPLE))
        job = mpf.GenericJob('Test Generic',
                             'test.pdf',
                             dict(TRANSFORMER_TAGGING_CORPUS=str(TEST_CONFIG / "minimal_corpus.json")),
                             {}, ff_track)

        comp = TransformerTaggingComponent()
        result = comp.get_detections_from_generic(job)

        self.assertEqual(1, len(result))

        props = result[0].detection_properties
        self.assertEqual('TRAVEL; VEHICLE', props["TAGS"]) # tags in lexicographic order

        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TEXT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TEXT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TEXT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)

        self.assertEqual(SHORT_PARK_SAMPLE_TRIGGER_SENTENCES, props["TRANSLATION VEHICLE TRIGGER SENTENCES"])
        self.assertEqual(SHORT_PARK_SAMPLE_OFFSET, props["TRANSLATION VEHICLE TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_PARK_SAMPLE_SCORE, float(props["TRANSLATION VEHICLE TRIGGER SENTENCES SCORE"]), places=3)

        self.assertEqual(SHORT_BEACH_SAMPLE_TRIGGER_SENTENCES, props["TRANSCRIPT TRAVEL TRIGGER SENTENCES"])
        self.assertEqual(SHORT_BEACH_SAMPLE_OFFSET, props["TRANSCRIPT TRAVEL TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_BEACH_SAMPLE_SCORE, float(props["TRANSCRIPT TRAVEL TRIGGER SENTENCES SCORE"]), places=3)

        self.assertEqual(SHORT_PARK_SAMPLE_TRIGGER_SENTENCES, props["TRANSCRIPT VEHICLE TRIGGER SENTENCES"])
        self.assertEqual('244-270', props["TRANSCRIPT VEHICLE TRIGGER SENTENCES OFFSET"])
        self.assertAlmostEqual(SHORT_PARK_SAMPLE_SCORE, float(props["TRANSCRIPT VEHICLE TRIGGER SENTENCES SCORE"]), places=3)


if __name__ == '__main__':
    unittest.main()
