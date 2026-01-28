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
import numpy as np

# Add gemini_component to path.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from gemini_component.gemini_component import GeminiComponent

import unittest
import unittest.mock
from unittest.mock import MagicMock, Mock
import mpf_component_api as mpf

logging.basicConfig(level=logging.DEBUG)
USE_MOCKS = True

# Replace with your own API key
GEMINI_API_KEY = ''

# Replace with your own desired model name
MODEL_NAME = "gemma-3-27-it"

class TestGemini(unittest.TestCase):
    def run_patched_job(self, component, job, side_effect_function):
        if isinstance(job, mpf.ImageJob):
            detection_func = component.get_detections_from_image
        elif isinstance(job, mpf.VideoJob):
            detection_func = component.get_detections_from_video
        else:
            raise Exception("Must be image or video job.")
    
        if not USE_MOCKS:
            return detection_func(job)

        with unittest.mock.patch("gemini_component.gemini_component.GeminiComponent._get_gemini_response", side_effect=side_effect_function):
            results = list(detection_func(job))
            return results

    def test_image_file(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            if prompt == "Describe what this person is wearing":
                response = "The person in the image is wearing a dark suit with a matching tie. The shirt underneath appears to be light-colored, possibly white or off-white. He has glasses on his face and is smiling as he shakes hands with someone who isn't fully visible in the frame. His attire suggests a formal setting, possibly for business or an event that requires professional dress code."
            elif prompt == "Describe what this person is doing":
                response = "The person in the image appears to be shaking someone's hand. They are wearing a suit and tie, which suggests they may be in a professional or formal setting. The context of the photo is not clear from this angle, but it looks like they could be at an event or gathering where such interactions are common."

            close_unlink_shm(data_uri)
            return {"response": f"{response}"}

        result = self.run_patched_job(component, job, side_effect_function)[0]
        
        self.assertTrue("CLOTHING" in result.detection_properties and "ACTIVITY" in result.detection_properties)
        self.assertTrue(len(result.detection_properties["CLOTHING"]) > 0 and len(result.detection_properties["ACTIVITY"]) > 0)
        
    def test_image_file_no_prompts(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-image-no-prompts',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            close_unlink_shm(data_uri)
            return {"response": ""}

        result = self.run_patched_job(component, job, side_effect_function)[0]
        self.assertTrue(len(result.detection_properties) == 1 and result.detection_properties['CLASSIFICATION'] == 'PERSON')

    def test_custom_config(self):
        ff_loc = mpf.ImageLocation(0, 0, 900, 1600, -1, dict(CLASSIFICATION="DOG"))
        job = mpf.ImageJob(
            job_name='test-custom',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()
    
        def side_effect_function(model_name, data_uri, prompt):
            if prompt == "Describe the color and breed of the dog.":
                response = "The dog in the image appears to be a Golden Retriever. The breed is known for its golden-colored fur, which can range from pale blonde to deeper golden shades, often with some darker feathering around the ears and along the tail. This specific dog has a beautiful golden coat that suggests it may be younger or well-groomed. The facial features of Golden Retriever dogs are also quite distinctive, such as their expressive eyes and long, floppy ears. They are medium to large-sized breed with a friendly and intelligent disposition."
            
            close_unlink_shm(data_uri)
            return {"response": f"{response}"}
        
        result = self.run_patched_job(component, job, side_effect_function)[0]
        
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
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                GENERATE_FRAME_RATE_CAP='-1'
            ),
            media_properties={},
            feed_forward_track=ff_track
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            if prompt == "Describe the color and breed of the dog.":
                response = "The dog in the image appears to be a Border Collie. The breed is characterized by its black and white color pattern, which you can see here with distinct patches of black fur against a mostly white background. Border Collies are known for their intelligent eyes and expressive faces, which they use to work livestock. They also have a double coat that is thick and wavy in texture. In this photo, the dog looks well-groomed and healthy."

            close_unlink_shm(data_uri)
            return {"response": f"{response}"}
        
        result = list(self.run_patched_job(component, job, side_effect_function))[0]
        for ff_location in result.frame_locations.values():
            self.assertTrue("DESCRIPTION" in ff_location.detection_properties)
            self.assertTrue(len(ff_location.detection_properties['DESCRIPTION']) > 0)

    def test_full_frame_image(self):
        job = mpf.ImageJob(
            job_name='test-full-frame-image',
            data_uri=self._get_test_file('dog.jpg'),
            job_properties=dict(
                PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_prompts.json'),
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME
            ),
            media_properties={},
            feed_forward_location=None
        )
        component = GeminiComponent()
        def side_effect_function(model_name, data_uri, prompt):
            if prompt == "Describe this image":
                response = "The image shows a medium-sized dog sitting on a couch. The dog appears to be a breed with tan and white fur, likely a mix given the irregular patterns of its coat. It has a scrunched up expression on its face, possibly indicating curiosity or attentiveness towards something off-camera. There is a small animal, potentially another pet such as a cat, in front of the dog's paws. The background is blurred but seems to be an indoor setting with natural light filtering through."
            elif prompt == "Describe the location in this scene":
                response = "The image shows a dog sitting on a couch indoors. The room has a light wood floor and there's a glimpse of what appears to be an artwork or picture frame hanging on the wall in the background. The focus is on the dog, which suggests that it's either the main subject of the photograph or someone wanted to capture a candid moment with their pet."
            
            close_unlink_shm(data_uri)
            return {"response": f"{response}"}
        
        results = self.run_patched_job(component, job, side_effect_function)
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
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                GENERATE_FRAME_RATE_CAP='-1'
            ),
            media_properties={},
            feed_forward_track=None
        )  
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            if prompt == "Describe this image":
                response = "This is a photo of a dog with a black, white, and grey coat. The dog appears to be a Border Collie or similar breed known for its distinctive coloring. It's sitting on what looks like a concrete surface outdoors, possibly in a yard or on a patio. The dog has a focused gaze towards the camera, and its mouth is slightly open, suggesting it might be panting or perhaps reacting to the person taking the photo. In the background, there are elements of a fence and vegetation, indicating that this setting could be near a garden or a fenced area. The lighting suggests it's daytime."
            elif prompt == "Describe the location in this scene":
                response = "The image shows a dog sitting on what appears to be a stone or concrete floor. The dog is facing the camera with its mouth open, revealing its teeth and tongue, which could suggest it's panting or smiling. There is a fence in the background, indicating that this might be an outdoor area such as a garden, patio, or a residential backyard. Beyond the fence, there are some plants and trees, suggesting a natural environment. The lighting appears to be diffused, possibly from cloudy weather or shaded by nearby structures or foliage. There's no text visible in the image."

            close_unlink_shm(data_uri)
            return {"response": f"{response}"}
                        
        results = self.run_patched_job(component, job, side_effect_function)
        for result in results:    
            self.assertTrue("LOCATION" in result.detection_properties and "DESCRIPTION" in result.detection_properties)
            self.assertTrue(len(result.detection_properties["LOCATION"]) > 0 and len(result.detection_properties["DESCRIPTION"]) > 0)

    def test_json_response_image(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-json-response-image',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                ENABLE_JSON_PROMPT_FORMAT='True'
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            with open(os.path.join(os.path.dirname(__file__), 'data', 'outputs', f"{job.job_name}-output.txt")) as f:
                response = f.read()

            close_unlink_shm(data_uri)
            return response
        
        result = self.run_patched_job(component, job, side_effect_function)[0]
        self.assertTrue(len(result.detection_properties) > 1)

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
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                ENABLE_JSON_PROMPT_FORMAT='True',
                JSON_PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_json_prompts.json'),
                GENERATE_FRAME_RATE_CAP='-1'
            ),
            media_properties={},
            feed_forward_track=ff_track
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            with open(os.path.join(os.path.dirname(__file__), 'data', 'outputs', f"{job.job_name}-output.txt")) as f:
                response = f.read()

            close_unlink_shm(data_uri)
            return response

        result = list(self.run_patched_job(component, job, side_effect_function))[0]
        self.assertTrue(len(result.frame_locations[0].detection_properties) > 3)
    
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
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                GENERATE_FRAME_RATE_CAP='1.0'
            ),
            media_properties={
                'FPS': '2'
            },
            feed_forward_track=ff_track
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            if prompt == "Describe the color and breed of the dog.":
                response = "The dog in the image appears to be a Border Collie. The breed is characterized by its black and white color pattern, which you can see here with distinct patches of black fur against a mostly white background. Border Collies are known for their intelligent eyes and expressive faces, which they use to work livestock. They also have a double coat that is thick and wavy in texture. In this photo, the dog looks well-groomed and healthy."
            
            close_unlink_shm(data_uri)
            return {"response": f"{response}"}
        
        result = list(self.run_patched_job(component, job, side_effect_function))[0]

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
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME
            ),
            media_properties={
                'FPS': '2'
            },
            feed_forward_track=ff_track
        )
        component = GeminiComponent()

        def side_effect_function(model_name, data_uri, prompt):
            with open(os.path.join(os.path.dirname(__file__), 'data', 'outputs', f"{job.job_name}-output.txt")) as f:
                response = f.read()

            close_unlink_shm(data_uri)
            return response

        result = list(self.run_patched_job(component, job, side_effect_function))[0]

        for i, ff_location in result.frame_locations.items():
            if i % 2 == 0:
                self.assertTrue("ANNOTATED BY GEMINI" in ff_location.detection_properties)
            else:
                self.assertTrue("ANNOTATED BY GEMINI" not in ff_location.detection_properties)

    def test_unsure_results(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-unsure',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                JSON_PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_json_prompts.json'),
                ENABLE_JSON_PROMPT_FORMAT='True' 
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()

        expected_detection_properties = {
            'CLASSIFICATION': 'PERSON', 
            'GEMINI PERSON ACTION PERFORMED': 'Walking', 
            'GEMINI PERSON BACKGROUND DESCRIBE': 'The woman is walking indoors, possibly in a corridor or hall. The background is out of focus and does not provide any specific information.', 
            'GEMINI PERSON CLOTHING UPPER BODY CLOTHING': 'black top', 'GEMINI PERSON CLOTHING LOWER BODY CLOTHING': 'black pants', 
            'GEMINI PERSON ESTIMATED GENDER': 'female', 'GEMINI PERSON OBJECT IN HAND TYPE': 'bag', 'GEMINI PERSON OBJECT IN HAND COLOR': 'dark', 
            'GEMINI PERSON OBJECT IN HAND DESCRIBE': 'Person is carrying a large dark bag.', 
            'GEMINI PERSON SHOE TYPE': 'sneakers', 'GEMINI PERSON SHOE COLOR': 'white', 
            'GEMINI PERSON SHOE DESCRIBE': 'Woman is wearing white sneakers', 
            'GEMINI PERSON TYPE': 'civilian', 
            'GEMINI PERSON NEST LEVEL 1 LEVEL 1': 'valid', 
            'GEMINI PERSON NEST LEVEL 1 NEST LEVEL 2 LEVEL 2': 'valid', 
            'GEMINI PERSON NEST LEVEL 1 NEST LEVEL 2 NEST LEVEL 3 LEVEL 3': 'valid', 
            'GEMINI PERSON OTHER NOTABLE CHARACTERISTICS NORMAL': 'behavior', 
            'GEMINI PERSON OTHER NOTABLE CHARACTERISTICS BEHAVIOR': 'normal', 
            'ANNOTATED BY GEMINI': True
        }

        def side_effect_function(model_name, data_uri, prompt):
            with open(os.path.join(os.path.dirname(__file__), 'data', 'outputs', f"{job.job_name}-output.txt")) as f:
                response = f.read()

            close_unlink_shm(data_uri)
            return response

        result = self.run_patched_job(component, job, side_effect_function)[0]

        self.assertEqual(result.detection_properties, expected_detection_properties)

    def test_visible_results(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-visible',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                JSON_PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_json_prompts.json'),
                ENABLE_JSON_PROMPT_FORMAT='True' 
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()

        expected_detection_properties = {
            'CLASSIFICATION': 'PERSON',
            'GEMINI PERSON CLOTHING HEADWEAR': '[{"type": "hoodie", "color": "black", "location": "head", "description": "A black hood that is pulled up."}]',
            'GEMINI PERSON CLOTHING TOP LAYER TYPE': 'jacket',
            'GEMINI PERSON CLOTHING TOP LAYER COLOR': 'black',
            'GEMINI PERSON CLOTHING TOP LAYER LOCATION': 'torso',
            'GEMINI PERSON CLOTHING TOP LAYER DESCRIPTION': 'A dark colored jacket covering the torso.',
            'GEMINI PERSON CLOTHING LOWER LAYER COLOR': 'dark',
            'GEMINI PERSON CLOTHING LOWER LAYER LOCATION': 'pants',
            'GEMINI PERSON CLOTHING LOWER LAYER DESCRIPTION': 'Dark-colored pants that are only partially visible.',
            'GEMINI PERSON ESTIMATED AGE RANGE': 'adult',
            'GEMINI PERSON ESTIMATED GENDER': 'male',
            'GEMINI PERSON PERSON WEARING SHOE': 'True',
            'GEMINI PERSON SHOE TYPE': 'sneaker',
            'GEMINI PERSON SHOE COLOR': 'black', 
            'GEMINI PERSON SHOE DESCRIPTION': 'Black sneaker on the foot.', 
            'GEMINI PERSON HEAD FEATURES BALD': 'True', 
            'GEMINI PERSON HEAD FEATURES HEAD COVER TYPE': 'hoodie', 
            'GEMINI PERSON ACTION PERFORMED': 'walking', 
            'GEMINI PERSON BACKGROUND DESCRIBE': 'The person is in a building with an indoor surface visible behind them.', 
            'GEMINI PERSON BACKGROUND TYPE': 'indoors', 
            'ANNOTATED BY GEMINI': True
        }

        def side_effect_function(model_name, data_uri, prompt):
            with open(os.path.join(os.path.dirname(__file__), 'data', 'outputs', f"{job.job_name}-output.txt")) as f:
                response = f.read()

            close_unlink_shm(data_uri)
            return response

        result = self.run_patched_job(component, job, side_effect_function)[0]

        self.assertEqual(result.detection_properties, expected_detection_properties)

    def test_ignore_person_results(self):
        ff_loc = mpf.ImageLocation(0, 0, 347, 374, -1, dict(CLASSIFICATION="PERSON"))
        job = mpf.ImageJob(
            job_name='test-ignore-person',
            data_uri=self._get_test_file('person.jpg'),
            job_properties=dict(
                GEMINI_API_KEY=GEMINI_API_KEY,
                MODEL_NAME=MODEL_NAME,
                JSON_PROMPT_CONFIGURATION_PATH=self._get_test_file('custom_json_prompts.json'),
                ENABLE_JSON_PROMPT_FORMAT='True' 
            ),
            media_properties={},
            feed_forward_location=ff_loc
        )
        component = GeminiComponent()

        expected_detection_properties = {
            'CLASSIFICATION' : 'PERSON',
            'ANNOTATED BY GEMINI' : True
        }

        def side_effect_function(model, data_uri, prompt):
            with open(os.path.join(os.path.dirname(__file__), 'data', 'outputs', f"{job.job_name}-output.txt")) as f:
                response = f.read()
            close_unlink_shm(data_uri)
            return response

        result = self.run_patched_job(component, job, side_effect_function)[0]

        self.assertEqual(result.detection_properties, expected_detection_properties)
        
    def test_get_frames(self):
        component = GeminiComponent()
        self.assertEqual(component._get_frames_to_process([], 1), [])
        self.assertEqual(component._get_frames_to_process([1], 2), [1])
        self.assertEqual(component._get_frames_to_process([503], 2), [503])
        self.assertEqual(component._get_frames_to_process([503, 1_000], 5_000), [503])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5], 1), [0,1,2,3,4,5])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5], 2), [0,2,4, 5])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5], 3), [0,3,5])
        self.assertEqual(component._get_frames_to_process([0,1,2,3,4,5,900], 3), [0,3,5,900])
        self.assertEqual(component._get_frames_to_process([4,900,902,905,906,907,908,909,910,911,912,913], 5), [4,900,905,910,913])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918], 6), [910,916])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919], 6), [910,916,919])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919,920], 6), [910,916,920])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919,920,921], 6), [910,916,921])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,919,920,921,922], 6), [910,916,922])
        self.assertEqual(component._get_frames_to_process([910,911,912,913,914,915,916,917,918,5_000,5_001,10_000], 6), [910,916,5_000,10_000])

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)

def close_unlink_shm(data_uri):
    if isinstance(data_uri, tuple):
        _, shm = data_uri
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    unittest.main(verbosity=2)