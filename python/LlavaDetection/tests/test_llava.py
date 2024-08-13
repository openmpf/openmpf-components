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
import ollama
import httpx

# Add clip_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llava_component.llava_component import LlavaComponent

import unittest
import unittest.mock
from unittest.mock import MagicMock, Mock
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)
USE_MOCKS = True

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
        
        def get_generate_output(job_name, prompt):
            outputs = {
                'test-image': {"Describe what this person is wearing": "The person in the image is wearing a dark suit with a matching tie. The shirt underneath appears to be light-colored, possibly white or off-white. He has glasses on his face and is smiling as he shakes hands with someone who isn't fully visible in the frame. His attire suggests a formal setting, possibly for business or an event that requires professional dress code.", "Describe what this person is doing": "The person in the image appears to be shaking someone's hand. They are wearing a suit and tie, which suggests they may be in a professional or formal setting. The context of the photo is not clear from this angle, but it looks like they could be at an event or gathering where such interactions are common."},
                'test-custom': {"Describe the color and breed of the dog.": "The dog in the image appears to be a Golden Retriever. The breed is known for its golden-colored fur, which can range from pale blonde to deeper golden shades, often with some darker feathering around the ears and along the tail. This specific dog has a beautiful golden coat that suggests it may be younger or well-groomed. The facial features of Golden Retriever dogs are also quite distinctive, such as their expressive eyes and long, floppy ears. They are medium to large-sized breed with a friendly and intelligent disposition."},
                'test-video': {"Describe the color and breed of the dog.": "The dog in the image appears to be a Border Collie. The breed is characterized by its black and white color pattern, which you can see here with distinct patches of black fur against a mostly white background. Border Collies are known for their intelligent eyes and expressive faces, which they use to work livestock. They also have a double coat that is thick and wavy in texture. In this photo, the dog looks well-groomed and healthy."}
            }
            return {"response": f"{outputs[job_name][prompt]}"}

        mock_container = MagicMock()
        with unittest.mock.patch("ollama.Client", return_value=mock_container):
            mock_container.generate = Mock(side_effect=lambda model, prompt, images: get_generate_output(job.job_name, prompt))

            results = list(detection_func(job))
            return results

    def test_image_file(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                OLLAMA_CLIENT_HOST_URL='localhost:11434'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]
        
        self.assertTrue("CLOTHING" in result.detection_properties and "ACTIVITY" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["CLOTHING"]) > 0 and len(result.detection_properties["ACTIVITY"]) > 0)

        print("Test Image File:\n" + "-"*16)
        print(f"CLOTHING: \n\n{result.detection_properties['CLOTHING']}\n")
        print(f"ACTIVITY: \n\n{result.detection_properties['ACTIVITY']}\n\n")
    
    def test_custom_config(self):
        ff_loc = mpf.ImageLocation(0, 0, 900, 1600, -1, dict(CLASSIFICATION="DOG"))
        job = mpf.ImageJob(
            job_name='test-custom',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                OLLAMA_CLIENT_HOST_URL='localhost:11434'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]
        
        self.assertTrue("DESCRIPTION" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["DESCRIPTION"]) > 0)

        print("Test Custom Config:\n" + "-"*19)
        print(f"DESCRIPTION: \n\n{result.detection_properties['DESCRIPTION']}\n\n")

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
                OLLAMA_CLIENT_HOST_URL='localhost:11434'
            ),
            media_properties={},
            feed_forward_track=ff_track
        )
        component = LlavaComponent()
        result = self.run_patched_job(component, job)[0]

        print("Test Video File:\n" + "-"*16)
        for ff_location in result.frame_locations.values():
            self.assertTrue("DESCRIPTION" in ff_location.detection_properties)
            self.assertTrue(len(ff_location.detection_properties['DESCRIPTION']) > 0)
        
            print(f"DESCRIPTION: \n\n{ff_location.detection_properties['DESCRIPTION']}\n")

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)