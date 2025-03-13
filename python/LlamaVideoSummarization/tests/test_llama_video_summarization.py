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
import os
import pathlib
import sys
import unittest
import unittest.mock

import mpf_component_api as mpf

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from llama_video_summarization_component import LlamaVideoSummarizationComponent, ChildProcess

logging.basicConfig(level=logging.DEBUG)

USE_MOCKS = eval(os.environ.get('USE_MOCKS', 'True'))
TEST_DATA = pathlib.Path(__file__).parent / 'data'

DUMMY_TIMELINE = \
[
    {
        "timestamp_start": 0,
        "timestamp_end": 50,
        "description": "Something happened."
    },
    {
        "timestamp_start": 51,
        "timestamp_end": 99,
        "description": "Something else happened."
    }
]

class TestComponent(unittest.TestCase):

    def run_patched_job(self, component, job, response):
        detection_func = component.get_detections_from_video

        if not USE_MOCKS:
            return detection_func(job)

        with unittest.mock.patch.object(ChildProcess, '__init__', lambda: None):
            with unittest.mock.patch.object(ChildProcess, '__del__', lambda: None):
                with unittest.mock.patch.object(ChildProcess, 'send_job_get_response', lambda x, y: response):
                    return detection_func(job)


    def test_multiple_videos(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100, {}, {})
        results = self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a cat.",
                "video_length": 100,
                "video_event_timeline": DUMMY_TIMELINE
            }))
        
        job = mpf.VideoJob('dog job', str(TEST_DATA / 'dog.mp4'), 0, 100, {}, {})
        results = self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a dog.",
                "video_length": 100,
                "video_event_timeline": DUMMY_TIMELINE
            }))
        self.assertIn("dog", results[0].detection_properties["TEXT"])


    def test_invalid_timeline(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100,
            { 
                "GENERATION_MAX_ATTEMPTS" : "1"
            }, 
            {})
        with self.assertRaises(mpf.DetectionException) as cm:
            results = self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a cat.",
                "video_length": 500,
                "video_event_timeline": DUMMY_TIMELINE
            }))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Video length", str(cm.exception))

        # test disabling time check
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100, 
            {
                "GENERATION_MAX_ATTEMPTS" : "1",
                "TIMELINE_CHECK_THRESHOLD" : "-1"
            },
            {})
        results = self.run_patched_job(component, job, json.dumps(
        {
            "video_summary": "This is a video of a cat.",
            "video_length": 500,
            "video_event_timeline": DUMMY_TIMELINE
        }))
        self.assertIn("cat", results[0].detection_properties["TEXT"])


    def test_invalid_json_response(self):
        component = LlamaVideoSummarizationComponent()
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100,
            {
                "GENERATION_MAX_ATTEMPTS" : "1"
            },
            {})

        with self.assertRaises(mpf.DetectionException) as cm:
            results = self.run_patched_job(component, job, "garbage xyz")
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("not valid JSON", str(cm.exception))


    def test_empty_response(self):
        component = LlamaVideoSummarizationComponent()
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100,
            {
                "GENERATION_MAX_ATTEMPTS" : "1"
            },
            {})

        with self.assertRaises(mpf.DetectionException) as cm:
            results = self.run_patched_job(component, job, "")
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Empty response", str(cm.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)