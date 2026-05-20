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

logging.basicConfig(level=logging.DEBUG)
USE_MOCKS = True
TEST_DATA = Path("data")

# Replace with your own desired model name
MODEL_NAME = "gemini-2.5-flash"

# Replace with your own path to the Google Application Credentials JSON file
GOOGLE_APPLICATION_CREDENTIALS="../application_default_credentials.json"

job_properties=dict(
    GOOGLE_APPLICATION_CREDENTIALS=GOOGLE_APPLICATION_CREDENTIALS,
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
    'DURATION': '6890',
    'FPS': '25',
    'FRAME_COUNT': '172',
    'FRAME_HEIGHT': '360',
    'FRAME_WIDTH': '640',
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

    def run_patched_job(self, component, job, response):
        if not USE_MOCKS:
            return
        
        if USE_MOCKS:
            with unittest.mock.patch("gemini_video_summarization_component.gemini_video_summarization_component.GeminiVideoSummarizationComponent._get_gemini_response", return_value=response):
                return component.get_detections_from_video(job)
        
    def test_local_model(self):
        model_id = "google/gemma-4-E2B-it"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        
        component = GeminiVideoSummarizationComponent(model=model, processor=processor)

        job = mpf.VideoJob('local model cat job', str(TEST_DATA / 'cat.mp4'), 0, 171,
            {
                "GENERATION_PROMPT_PATH":"../gemini_video_summarization_component/data/default_prompt.txt",
                "GENERATION_MAX_ATTEMPTS" : "1",
                "MODEL_NAME": MODEL_NAME
            },
            CAT_VIDEO_PROPERTIES, {})
        
        results = self.run_patched_job(component, job, json.dumps(CAT_TIMELINE))
        self.assertEqual(3, len(results))

if __name__ == "__main__":
    unittest.main(verbosity=0)
    