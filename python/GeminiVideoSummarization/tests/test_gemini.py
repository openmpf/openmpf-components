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

import sys
import os
import logging
import json
from pathlib import Path

import unittest
import unittest.mock

# Add gemini_video_summarization_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gemini_video_summarization_component.gemini_video_summarization_component import GeminiVideoSummarizationComponent

import unittest
import mpf_component_api as mpf
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

logging.basicConfig(level=logging.ERROR)
USE_MOCKS = True
TEST_DATA = Path("data")

# Replace with your own desired model name
MODEL_NAME = "gemini-2.5-flash"
OPENAI_MODEL_NAME = "o4-mini"

# Replace with your own path to the Google Application Credentials JSON file
GOOGLE_APPLICATION_CREDENTIALS="../application_default_credentials.json"
OPENAI_APPLICATION_CREDENTIALS="../openai_api_key.txt"

job_properties=dict(
    APPLICATION_CREDENTIALS=None,
    GENERATION_PROMPT_PATH="../gemini_video_summarization_component/data/default_prompt.txt"
)

CAT_TIMELINE = {
    "video_summary": "A cat is sitting on a cobblestone street, looking around as people walk by.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "0:04",
            "description": "The cat is sitting on the cobblestone street, looking around."
        },
        {
            "timestamp_start": "0:04",
            "timestamp_end": "0:06",
            "description": "The cat looks back at the camera and then walks away."
        }
    ]
}

INVALID_CAT_TIMELINE = {
    "video_summary": "A cat is sitting on a cobblestone street, looking around as people walk by.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "0:04",
            "description": "The cat is sitting on the cobblestone street, looking around."
        },
        {
            "timestamp_start": "0:04",
            "timestamp_end": "0:17",
            "description": "The cat looks back at the camera and then walks away."
        }
    ]
}

CAT_VIDEO_PROPERTIES = {
    'DURATION': '6840',
    'FPS': '25',
    'FRAME_COUNT': '171',
    'FRAME_HEIGHT': '640',
    'FRAME_WIDTH': '360',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

DOG_TIMELINE = {
    "video_summary": "A dog sitting by a window and looking around.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "0:06",
            "description": "Dog sitting by the window."
        }
    ]
}

DOG_VIDEO_PROPERTIES = {
    'DURATION': '6170',
    'FPS': '25',
    'FRAME_COUNT': '154',
    'FRAME_HEIGHT': '240',
    'FRAME_WIDTH': '426',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

SHORT_TIMELINE = {
    "video_summary": "A person is running around.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "0:01",
            "description": "A person running."
        }
    ]
}

SHORT_VIDEO_PROPERTIES = {
    'DURATION': '1000',
    'FPS': '30.0',
    'FRAME_COUNT': '31',
    'FRAME_HEIGHT': '1920',
    'FRAME_WIDTH': '1080',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

MISSILE_TIMELINE = {
    "video_summary": "A missile streaks downward across the frame and strikes the ground, producing a large fireball.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "0:01",
            "description": "A fast-moving projectile descends and impacts, creating an explosion."
        }
    ]
}

MISSILE_VIDEO_PROPERTIES = {
    'DURATION': '1000',
    'FPS': '30.0',
    'FRAME_COUNT': '30',
    'FRAME_HEIGHT': '852',
    'FRAME_WIDTH': '720',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

DRONE_TIMELINE_SEGMENT_1 = {
    "video_summary": "The video shows a street in Turkey with traffic and pedestrians.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "1:01",
            "description": "An aerial view of a city with traffic and pedestrians."
        },
        {
            "timestamp_start": "1:02",
            "timestamp_end": "1:29",
            "description": "Traffic continues to pass through with little change to the scenery."
        },
        {
            "timestamp_start": "1:30",
            "timestamp_end": "2:00",
            "description": "The video continues as pedestrains and street traffic pass through the city center."
        }
    ]
}

# Valid timeline for timeline integrity test (ends at 3:36 to fit 5393 frames at 25fps = 215.72s)
DRONE_TIMELINE_SEGMENT_1_VALID = {
    "video_summary": "The video shows a street in Turkey with traffic and pedestrians.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "1:01",
            "description": "An aerial view of a city with traffic and pedestrians."
        },
        {
            "timestamp_start": "1:02",
            "timestamp_end": "1:29",
            "description": "Traffic continues to pass through with little change to the scenery."
        },
        {
            "timestamp_start": "1:30",
            "timestamp_end": "3:36",
            "description": "The video continues as pedestrains and street traffic pass through the city center."
        }
    ]
}

# events beyond video length
DRONE_TIMELINE_SEGMENT_2 = {
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "1:58",
            "description": "An aerial view of a city with traffic and pedestrians."
        },
        {
            "timestamp_start": "1:58",
            "timestamp_end": "2:10",
            "description": "Traffic continues to pass through with little change to the scenery."
        },
        {
            "timestamp_start": "2:11",
            "timestamp_end": "2:40",
            "description": "The video continues as pedestrains and street traffic pass through the city center."
        }
    ]
}

# Short timeline for segment length validation test
DRONE_TIMELINE_SEGMENT_2_SHORT = {
    "video_summary": "An aerial view of a city with traffic and pedestrians.",
    "video_event_timeline": [
        {
            "timestamp_start": "0:00",
            "timestamp_end": "0:35",
            "description": "An aerial view of a city with traffic and pedestrians."
        }
    ]
}

DRONE_VIDEO_PROPERTIES = {
    'DURATION': '120000',
    'FPS': '25.0',
    'FRAME_COUNT': '3001',
    'FRAME_HEIGHT': '1080',
    'FRAME_WIDTH': '1920',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

class TestGemini(unittest.TestCase):

    def test_local_chat_template_falls_back_when_processor_template_missing(self):
        class FakeProcessor:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                raise ValueError(
                    "Cannot use chat template functions because tokenizer.chat_template "
                    "is not set and no template argument was passed!"
                )

        component = GeminiVideoSummarizationComponent(
            model=object(),
            processor=FakeProcessor(),
            device="cpu"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/tmp/video.mp4"},
                    {"type": "text", "text": "Summarize."}
                ]
            }
        ]

        self.assertEqual(
            "<bos><|turn>user\n\n\n<|video|>\n\nSummarize.<turn|>\n<|turn>model\n",
            component._apply_local_chat_template(messages)
        )

    def test_local_num_frames_avoids_torchvision_endpoint_index(self):
        self.assertEqual(
            29,
            GeminiVideoSummarizationComponent._avoid_torchvision_endpoint_frame_index(290, 30)
        )
        self.assertEqual(
            32,
            GeminiVideoSummarizationComponent._avoid_torchvision_endpoint_frame_index(290, 32)
        )

    def test_local_model_uses_preprocessed_frame_count_for_num_frames(self):
        response = json.dumps(MISSILE_TIMELINE)

        class FakeInputs(dict):
            def __init__(self):
                super().__init__(input_ids=[[1, 2]])

            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, device):
                return self

        class FakeModel:
            def generate(self, **kwargs):
                return [[1, 2, 3]]

        class FakeVideoProcessor:
            num_frames = 32

        class FakeProcessor:
            video_processor = FakeVideoProcessor()

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                return "<|video|> Summarize."

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return FakeInputs()

            def batch_decode(self, generated_ids, skip_special_tokens=True):
                return [response]

        processor = FakeProcessor()
        component = GeminiVideoSummarizationComponent(
            model=FakeModel(),
            processor=processor,
            device="cpu"
        )
        job = mpf.VideoJob(
            "short local model job",
            str(TEST_DATA / "falling-missile-cropped-speedup-trimmed 2.mp4"),
            0,
            29,
            {},
            MISSILE_VIDEO_PROPERTIES
        )

        self.assertEqual(response, component._local_get_response(job, "Summarize."))
        self.assertEqual(32, processor.call_kwargs["num_frames"])

    def run_patched_job(self, component, job, response):
        if not USE_MOCKS:
            return component.get_detections_from_video(job)
        
        if USE_MOCKS:
            if component.model is not None:
                response_method = "_local_get_response"
            elif component.api == "OpenAI":
                response_method = "_openai_response"
            elif component.api == "Google":
                response_method = "_google_response"
            else:
                response_method = "_get_response"

            patch_path = (
                "gemini_video_summarization_component.gemini_video_summarization_component."
                f"GeminiVideoSummarizationComponent.{response_method}"
            )
            with unittest.mock.patch(patch_path, return_value=response):
                return component.get_detections_from_video(job)
    
    def assert_detection_region(self, detection, frame_width, frame_height):    
        self.assertEqual(0, detection.x_left_upper)
        self.assertEqual(0, detection.y_left_upper)
        self.assertEqual(frame_width, detection.width)
        self.assertEqual(frame_height, detection.height)


    def assert_first_middle_last_detections(self, track, frame_width, frame_height):    
        self.assertIn(track.start_frame, track.frame_locations)
        self.assert_detection_region(track.frame_locations[track.start_frame], frame_width, frame_height)

        self.assertIn(track.stop_frame, track.frame_locations)
        self.assert_detection_region(track.frame_locations[track.stop_frame], frame_width, frame_height)

        middle_frame = int((track.stop_frame - track.start_frame) / 2) + track.start_frame
        self.assertIn(middle_frame, track.frame_locations)
        self.assert_detection_region(track.frame_locations[middle_frame], frame_width, frame_height)

    def test_openai_api_routes_to_openai_response(self):
        component = GeminiVideoSummarizationComponent(API="OpenAI")

        job = mpf.VideoJob('openai cat job', str(TEST_DATA / 'cat.mp4'), 0, 170,
            {
                "APPLICATION_CREDENTIALS": OPENAI_APPLICATION_CREDENTIALS,
                "MODEL_NAME": OPENAI_MODEL_NAME,
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
            },
            CAT_VIDEO_PROPERTIES)
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        with unittest.mock.patch(
            "gemini_video_summarization_component.gemini_video_summarization_component."
            "GeminiVideoSummarizationComponent._openai_response",
            return_value=json.dumps(CAT_TIMELINE)
        ) as openai_response, unittest.mock.patch(
            "gemini_video_summarization_component.gemini_video_summarization_component."
            "GeminiVideoSummarizationComponent._google_response"
        ) as google_response:
            results = component.get_detections_from_video(job)

        openai_response.assert_called_once()
        google_response.assert_not_called()
        self.assertEqual(3, len(results))
        self.assertEqual('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertIn("looking around as people walk by.", results[0].detection_properties["TEXT"])
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

    def test_openai_api_invalid_json_response(self):
        component = GeminiVideoSummarizationComponent(API="OpenAI")

        job = mpf.VideoJob('openai invalid cat job JSON', str(TEST_DATA / 'cat.mp4'), 0, 100,
            {
                "APPLICATION_CREDENTIALS": OPENAI_APPLICATION_CREDENTIALS,
                "MODEL_NAME": OPENAI_MODEL_NAME,
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
            },
            CAT_VIDEO_PROPERTIES)

        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job, "garbage xyz")

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("not valid JSON", str(cm.exception))

    def test_multiple_videos(self):
        component = GeminiVideoSummarizationComponent(API="Google")
        job_props = job_properties.copy()
        job_props["APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
        
        job = mpf.VideoJob('valid cat job', str(TEST_DATA / 'cat.mp4'), 0, 170, job_props, CAT_VIDEO_PROPERTIES)
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])


        results = self.run_patched_job(component, job, json.dumps(CAT_TIMELINE))
        self.assertEqual(3, len(results))
        self.assertEqual('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertIn("looking around as people walk by.", results[0].detection_properties["TEXT"])
        self.assertEqual(0, results[0].start_frame)
        self.assertEqual(170, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("looking around.", results[1].detection_properties["TEXT"])
        self.assertEqual(0, results[1].start_frame) # 0 * 25
        self.assertEqual(99, results[1].stop_frame) # (4 * 25) - 1
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)

        self.assertIn("looks back at the camera", results[2].detection_properties["TEXT"])
        self.assertEqual(100, results[2].start_frame) # 4 * 25
        self.assertEqual(149, results[2].stop_frame) # (6 * 25) - 1 
        self.assert_first_middle_last_detections(results[2], frame_width, frame_height)


        job = mpf.VideoJob('valid dog job', str(TEST_DATA / 'dog.mp4'), 0, 153, job_props, DOG_VIDEO_PROPERTIES)
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        results = self.run_patched_job(component, job, json.dumps(DOG_TIMELINE))
        self.assertEqual(2, len(results))

        self.assertEqual('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertIn("sitting by a window and looking around", results[0].detection_properties["TEXT"])
        self.assertEqual(0, results[0].start_frame)
        self.assertEqual(153, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("sitting by the window.", results[1].detection_properties["TEXT"])
        self.assertEqual(0, results[1].start_frame) # 0 * 25
        self.assertEqual(149, results[1].stop_frame) # (6 * 25) - 1
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)

        job = mpf.VideoJob('short job', str(TEST_DATA / 'short.mp4'), 0, 0, job_props, SHORT_VIDEO_PROPERTIES)
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        results = self.run_patched_job(component, job, json.dumps(SHORT_TIMELINE))
        self.assertEqual(2, len(results))

        self.assertEqual('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertIn("A person is running around.", results[0].detection_properties["TEXT"])
        self.assertEqual(0, results[0].start_frame)
        self.assertEqual(0, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("A person running.", results[1].detection_properties["TEXT"])
        self.assertEqual(0, results[1].start_frame)
        self.assertEqual(0, results[1].stop_frame) 
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)

        job = mpf.VideoJob(
            'missile job',
            str(TEST_DATA / 'falling-missile-cropped-speedup-trimmed 2.mp4'),
            0,
            29,
            job_properties,
            MISSILE_VIDEO_PROPERTIES
        )
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        results = self.run_patched_job(component, job, json.dumps(MISSILE_TIMELINE))
        self.assertEqual(2, len(results))

        self.assertEqual('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertIn("missile", results[0].detection_properties["TEXT"].lower())
        self.assertEqual(0, results[0].start_frame)
        self.assertEqual(29, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("explosion", results[1].detection_properties["TEXT"].lower())
        self.assertEqual(0, results[1].start_frame)
        self.assertEqual(29, results[1].stop_frame)
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)

    def test_invalid_timeline(self):
        component = GeminiVideoSummarizationComponent(API="Google")

        job = mpf.VideoJob('invalid cat job', str(TEST_DATA / 'cat.mp4'), 0, 15000,
            { 
                "APPLICATION_CREDENTIALS": GOOGLE_APPLICATION_CREDENTIALS,
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
            }, 
            CAT_VIDEO_PROPERTIES)
        
        with self.assertRaises(mpf.DetectionException) as cm:
             self.run_patched_job(component, job, json.dumps(INVALID_CAT_TIMELINE)) # don't care about results

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Max timeline event end time not close enough to segment stop time.", str(cm.exception))

        # test disabling time check
        job = mpf.VideoJob('invalid cat job', str(TEST_DATA / 'cat.mp4'), 0, 15000, 
            {
                "GOOGLE_APPLICATION_CREDENTIALS": GOOGLE_APPLICATION_CREDENTIALS,
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
                "TIMELINE_CHECK_TARGET_THRESHOLD" : "-1"
            },
            CAT_VIDEO_PROPERTIES)
            
        results = self.run_patched_job(component, job, json.dumps(INVALID_CAT_TIMELINE))

        self.assertIn("cat", results[0].detection_properties["TEXT"])

    def test_invalid_json_response(self):
        component = GeminiVideoSummarizationComponent(API="Google")

        job = mpf.VideoJob('invalid cat job JSON', str(TEST_DATA / 'cat.mp4'), 0, 100,
            {
                "APPLICATION_CREDENTIALS": GOOGLE_APPLICATION_CREDENTIALS,
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
            },
            CAT_VIDEO_PROPERTIES)

        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job, "garbage xyz") # don't care about results
        
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("not valid JSON", str(cm.exception))

    def test_empty_response(self):
        component = GeminiVideoSummarizationComponent(API="Google")

        job = mpf.VideoJob('empty cat job', str(TEST_DATA / 'cat.mp4'), 0,  170,
            {
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
            },
            CAT_VIDEO_PROPERTIES)

        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job, "") # don't care about results

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Empty response", str(cm.exception))

if __name__ == "__main__":
    unittest.main(verbosity=2)