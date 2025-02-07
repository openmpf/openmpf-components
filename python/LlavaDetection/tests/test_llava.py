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
import warnings
import json

# Add clip_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llava_component.llava_component import LlavaComponent

import unittest
import unittest.mock
from unittest.mock import MagicMock, Mock
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)
USE_MOCKS = False

class TestLlava(unittest.TestCase):

    def run_patched_job(self, component, job):
        if isinstance(job, mpf.ImageJob):
            detection_func = component.get_detections_from_image
        elif isinstance(job, mpf.VideoJob):
            detection_func = component.get_detections_from_video
        else:
            raise Exception("Must be image or video job.")
    
        if not USE_MOCKS:
            return detection_func(job)
        
        def _read_generate_outputs(fpath):
            outputs = dict()
            with open(fpath) as f:
                data = json.load(f)
                for job in data:
                    job_name, responses = job['jobName'], job['responses']
                    outputs[job_name] = dict()
                    for response_dict in responses:
                        outputs[job_name][response_dict['prompt']] = response_dict['response']
            return outputs

        mock_container = MagicMock()
        with unittest.mock.patch("ollama.Client", return_value=mock_container):
            outputs = _read_generate_outputs(self._get_test_file('generate_outputs.json'))
            mock_container.generate = Mock(side_effect=lambda model, prompt, images: {"response": f"{outputs[job.job_name][prompt]}"})#get_generate_output(outputs, job.job_name, prompt))

            results = list(detection_func(job))
            return results

    def test_image_file(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                OLLAMA_SERVER='localhost:11434'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]
        
        self.assertTrue("CLOTHING" in result.detection_properties and "ACTIVITY" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["CLOTHING"]) > 0 and len(result.detection_properties["ACTIVITY"]) > 0)

    def test_image_file_no_prompts(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-image-no-prompts',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_SERVER='localhost:11434'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]

    def test_custom_config(self):
        ff_loc = mpf.ImageLocation(0, 0, 900, 1600, -1, dict(CLASSIFICATION="DOG"))
        job = mpf.ImageJob(
            job_name='test-custom',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_SERVER='localhost:11434'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]
        
        self.assertTrue("DESCRIPTION" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["DESCRIPTION"]) > 0)

    def test_video_file(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

        ff_track =  mpf.VideoTrack(0, 0, -1, {}, {'CLASSIFICATION': 'DOG'})
        ff_track.frame_locations[0] = mpf.ImageLocation(0, 0, 3456, 5184, -1, {'CLASSIFICATION': 'DOG', 'CLASSIFICATION CONFIDENCE LIST': '-1', 'CLASSIFICATION LIST': 'DOG'})

        job = mpf.VideoJob(
            job_name='test-video',
            data_uri=self._get_test_file('test_video.mp4'),
            start_frame=0,
            stop_frame=0,
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_SERVER='localhost:11434',
                GENERATE_FRAME_RATE_CAP='-1'
            ),
            media_properties={},
            feed_forward_track=ff_track
        )
        component = LlavaComponent()
        result = list(self.run_patched_job(component, job))[0]
        for ff_location in result.frame_locations.values():
            self.assertTrue("DESCRIPTION" in ff_location.detection_properties)
            self.assertTrue(len(ff_location.detection_properties['DESCRIPTION']) > 0)

    def test_full_frame_image(self):
        job = mpf.ImageJob(
            job_name='test-full-frame-image',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_SERVER='localhost:11434'
            ),
            media_properties={},
            feed_forward_location=None
        )
        component = LlavaComponent()
        results = self.run_patched_job(component, job)
        for result in results:
            self.assertTrue("LOCATION" in result.detection_properties and "DESCRIPTION" in result.detection_properties)
            self.assertTrue(len(result.detection_properties["LOCATION"]) > 0 and len(result.detection_properties["DESCRIPTION"]) > 0)

    def test_full_frame_video(self):
        job = mpf.VideoJob(
            job_name='test-full-frame-video',
            data_uri=self._get_test_file('test_video.mp4'),
            start_frame=0,
            stop_frame=14,
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_SERVER='localhost:11434',
                GENERATE_FRAME_RATE_CAP='-1'
            ),
            media_properties={},
            feed_forward_track=None
        )  
        component = LlavaComponent()
        results = self.run_patched_job(component, job)
        for result in results:    
            self.assertTrue("LOCATION" in result.detection_properties and "DESCRIPTION" in result.detection_properties)
            self.assertTrue(len(result.detection_properties["LOCATION"]) > 0 and len(result.detection_properties["DESCRIPTION"]) > 0)

    def test_json_response_image(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        # ff_loc = mpf.ImageLocation(0, 0, 262, 192, -1, dict(CLASSIFICATION="CAR"))
        job = mpf.ImageJob(
            job_name='test-json-response-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                OLLAMA_SERVER='localhost:11434',
                ENABLE_JSON_PROMPT_FORMAT='True'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]

    def test_json_response_video(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

        ff_track =  mpf.VideoTrack(0, 0, -1, {}, {'CLASSIFICATION': 'DOG'})
        ff_track.frame_locations[0] = mpf.ImageLocation(0, 0, 3456, 5184, -1, {'CLASSIFICATION': 'DOG', 'CLASSIFICATION CONFIDENCE LIST': '-1', 'CLASSIFICATION LIST': 'DOG'})

        job = mpf.VideoJob(
            job_name='test-json-response-video',
            data_uri=self._get_test_file('test_video.mp4'),
            start_frame=0,
            stop_frame=0,
            job_properties=dict(
                OLLAMA_SERVER='localhost:11434',
                ENABLE_JSON_PROMPT_FORMAT='True',
                JSON_PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_json_prompts.json'),
                GENERATE_FRAME_RATE_CAP='-1'
            ),
            media_properties={},
            feed_forward_track=ff_track
        )
        component = LlavaComponent()
        result = list(self.run_patched_job(component, job))[0]
    
    def test_video_file_nth_frame(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

        ff_track =  mpf.VideoTrack(0, 0, -1, {}, {'CLASSIFICATION': 'DOG'})
        for i in range(5):
            ff_track.frame_locations[i] = mpf.ImageLocation(0, 0, 3456, 5184, -1, {'CLASSIFICATION': 'DOG', 'CLASSIFICATION CONFIDENCE LIST': '-1', 'CLASSIFICATION LIST': 'DOG'})

        job = mpf.VideoJob(
            job_name='test-video-nth-frame',
            data_uri=self._get_test_file('test_video.mp4'),
            start_frame=0,
            stop_frame=4,
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_SERVER='localhost:11434',
                GENERATE_FRAME_RATE_CAP='1.0'
            ),
            media_properties={
                'FPS': '2'
            },
            feed_forward_track=ff_track
        )
        component = LlavaComponent()
        result = list(self.run_patched_job(component, job))[0]

        for i, ff_location in result.frame_locations.items():
            if i % 2 == 0:
                self.assertTrue("DESCRIPTION" in ff_location.detection_properties)
                self.assertTrue(len(ff_location.detection_properties['DESCRIPTION']) > 0)
            else:
                self.assertTrue("DESCRIPTION" not in ff_location.detection_properties)

    def test_video_file_nth_frame_json(self):
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

        ff_track =  mpf.VideoTrack(0, 0, -1, {}, {'CLASSIFICATION': 'DOG'})
        for i in range(5):
            ff_track.frame_locations[i] = mpf.ImageLocation(0, 0, 3456, 5184, -1, {'CLASSIFICATION': 'DOG', 'CLASSIFICATION CONFIDENCE LIST': '-1', 'CLASSIFICATION LIST': 'DOG'})

        job = mpf.VideoJob(
            job_name='test-video-nth-frame-json',
            data_uri=self._get_test_file('test_video.mp4'),
            start_frame=0,
            stop_frame=4,
            job_properties=dict(
                JSON_PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_json_prompts.json'),
                ENABLE_JSON_PROMPT_FORMAT='True',
                OLLAMA_SERVER='localhost:11434'
            ),
            media_properties={
                'FPS': '2'
            },
            feed_forward_track=ff_track
        )
        component = LlavaComponent()
        result = list(self.run_patched_job(component, job))[0]

        for i, ff_location in result.frame_locations.items():
            if i % 2 == 0:
                self.assertTrue("LLAVA" in ff_location.detection_properties)
            else:
                self.assertTrue("LLAVA" not in ff_location.detection_properties)

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)

if __name__ == '__main__':
    unittest.main(verbosity=2)