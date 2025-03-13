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

class TestComponent(unittest.TestCase):

    def run_patched_job(self, component, job, response):
        detection_func = component.get_detections_from_video

        if not USE_MOCKS:
            return detection_func(job)

        with unittest.mock.patch.object(ChildProcess, '__init__', lambda: None):
            with unittest.mock.patch.object(ChildProcess, '__del__', lambda: None):
                with unittest.mock.patch.object(ChildProcess, 'send_job_get_response', lambda x, y: response):
                    return detection_func(job)

    def test_video(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100, {}, {})
        results = self.run_patched_job(component, job,
            {
                "video_summary": "This is a video of a cat.",
                "video_length": 100,
                "video_event_timeline": "{json}"
            })
        self.assertIn("cat", results[0].detection_properties["TEXT"])

        job = mpf.VideoJob('dog job', str(TEST_DATA / 'dog.mp4'), 0, 100, {}, {})
        results = self.run_patched_job(component, job,
            {
                "video_summary": "This is a video of a dog.",
                "video_length": 200,
                "video_event_timeline": "{json}"
            })
        self.assertIn("dog", results[0].detection_properties["TEXT"])

if __name__ == "__main__":
    unittest.main(verbosity=2)