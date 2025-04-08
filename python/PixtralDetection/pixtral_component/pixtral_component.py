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

import time
import os
import cv2
import base64
import json
import re
from vllm import LLM
from vllm.sampling_params import SamplingParams

import logging
from typing import Mapping, Iterable

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('PixtralComponent')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['CUDE_DEVICE_ID'] = '7'

IGNORE_WORDS = ('unsure', 'none', 'false', 'no', 'unclear', 'n/a', 'unspecified', 'unknown', 'unreadable', 'not visible', 'none visible')
IGNORE_PREFIXES = tuple([s + ' ' for s in IGNORE_WORDS])

class PixtralComponent:
    detection_type = 'CLASS'

    def __init__(self):
        # self.model_name = 'mistralai/Pixtral-12B-2409'
        self.model_name = '/root/.cache/huggingface/hub/models--mistralai--Pixtral-12B-2409/snapshots/c21b6fd59bfe3b1246861d2811d0d6ae53f78915'
        self.host_url = ''
        self.sampling_params = None
        self.llm = LLM(
            model=self.model_name,
            tokenizer_mode="mistral"
            # max_model_len=13472
        )
        self.class_prompts = dict()
        self.json_class_prompts = dict()
        self.frame_prompts = dict()
        self.json_frame_prompts = dict()

        self.json_limit = 3

    def get_detections_from_image(self, image_job: mpf.ImageJob) -> Iterable[mpf.ImageLocation]:
        logger.info('Received image job: %s', image_job.job_name)

        config = JobConfig(image_job.job_properties)
        image_reader = mpf_util.ImageReader(image_job)

        self.sampling_params = SamplingParams(max_tokens=8192, temperature=config.temperature)

        if image_job.feed_forward_location is None:
            detections = self._get_frame_detections(image_job, [image_reader.get_image(),], config)
        else:
            raise mpf.DetectionException(
                "Feed forward jobs not supported yet: ",
                mpf.DetectionError.DETECTION_FAILED
            )
        
        logger.info(f"Job complete. Found {len(detections)} detections.")
        return detections

    def _get_frame_detections(self, job, reader, config):
        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)

        tracks = []
        for idx, frame in enumerate(reader):
            height, width, _ = frame.shape
            detection_properties = { 'CLASSIFICATION': 'SCREENSHOT' }

            encoded = self._encode_image(frame, job.media_properties['MIME_TYPE'])
            for tag, prompt in self.json_frame_prompts.items():
                json_attempts, json_failed = 0, True
                while (json_attempts < self.json_limit) and (json_failed):  
                    json_attempts += 1
                    response = self._get_pixtral_response(prompt, encoded)
                    try:
                        logger.info(response)
                        if response.startswith('```json\n'):
                            response = response.split('```json\n')[1].split('```')[0]
                        response_json = json.loads(response)
                        self._update_detection_properties(detection_properties, response_json)
                        json_failed = False
                    except Exception as e:
                        logger.warning(f"Pixtral failed to produce valid JSON output: {e}.")
                        logger.warning(f"Failed {json_attempts} of {self.json_limit} attempts.")
                        continue

                if json_failed:
                    logger.warning("Using last full Pixtral response instead of parsed JSON output.")
                    detection_properties["FAILED TO PROCESS PIXTRAL RESPONSE"] = True
                    detection_properties["FULL PIXTRAL RESPONSE"] = response

            img_location = mpf.ImageLocation(0, 0, width, height, -1, detection_properties)
            tracks.append(img_location)

        return tracks

    def _update_detection_properties(self, detection_properties, response_json):
        key_list = self._get_keys(response_json) 
        key_vals = dict()
        keywords = []
        for key_str in key_list:
            split_key = [' '.join(x.split('_')) for x in ('pixtral' + key_str).split('||')]
            key, val = " ".join([s.upper() for s in split_key[:-1]]), split_key[-1]
            key_vals[key] = val

        tmp_key_vals = dict(key_vals)
        for key, val in key_vals.items():
            if 'VISIBLE' in key:
                tmp_key_vals.pop(key)
                if self._ignore(val):
                    keywords.append(key.split(' VISIBLE ')[1])
        key_vals = tmp_key_vals

        tmp_key_vals = dict(key_vals)
        for keyword in keywords:
            pattern = re.compile(fr'\b{keyword}\b')
            for key_to_remove in filter(pattern.search, key_vals):
                tmp_key_vals.pop(key_to_remove, None)
        key_vals = tmp_key_vals
        
        tmp_key_vals = dict(key_vals)
        for key, val in key_vals.items():
            if self._ignore(val):
                tmp_key_vals.pop(key)
        key_vals = tmp_key_vals
        
        detection_properties.update(key_vals)
        detection_properties['ANNOTATED BY PIXTRAL'] = True

    def _get_keys(self, response_json):
        if not response_json:
            yield f'||none'

        elif isinstance(response_json, (str, bool)):
            yield f'||{response_json}'

        elif isinstance(response_json, list):
            for elt in response_json:
                yield from self._get_keys(elt)

        elif isinstance(response_json, dict):
            if self._is_lowest_level(response_json):

                tmp_response_json = dict(response_json)
                for key, val in response_json.items():
                    if self._ignore(val):
                        tmp_response_json.pop(key)
                response_json = tmp_response_json

                if not response_json:
                    yield f'||none'
                else:
                    yield from (f'||{key}||{val}' for key, val in response_json.items())

            else:
                for key, value in response_json.items():
                    if self._ignore(key):
                        yield f'||none'
                    else:
                        yield from (f'||{key}{p}' for p in self._get_keys(value))
        
    @staticmethod
    def _is_lowest_level(response_json):
        for key, val in response_json.items():
            if not isinstance(val, (str, bool)):
                return False
        return True
    
    @staticmethod
    def _ignore(input):
        return not input or \
            input.strip().lower() in IGNORE_WORDS or \
            input.strip().lower().startswith(IGNORE_PREFIXES)

    def _encode_image(self, image, mime_type):
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        _, buffer = cv2.imencode('.png', image, encode_params)
        encoded_string = base64.b64encode(buffer).decode("utf-8")
        
        return f"data:{mime_type};base64,{encoded_string}"

    def _update_prompts(self, prompt_config_path, json_prompt_config_path):
        '''
        Updates self.class_prompts dictionary to have the following format

        {
            CLASS1: {TAG1: PROMPT1},
            CLASS2: {TAG2: PROMPT2, TAG3: PROMPT3},
            ...
        }

        and self.frame_prompts to be a dict of key, prompt string pairs.
        '''
        try:
            with open(prompt_config_path, 'r') as f:
                data = json.load(f)
                class_dicts, frame_dicts = data['classPrompts'], data['framePrompts']
                for class_dict in class_dicts:
                    classes, prompts = [cls.lower() for cls in class_dict['classes']], class_dict['prompts']
                    for cls in classes:
                        if cls not in self.class_prompts:
                            self.class_prompts[cls] = dict()
                        self.class_prompts[cls].update({ dct['detectionProperty']:dct['prompt'] for dct in prompts })

                for frame_dict in frame_dicts:
                    self.frame_prompts[frame_dict['detectionProperty']] = frame_dict['prompt']
            
            with open(json_prompt_config_path, 'r') as f:
                data = json.load(f)
                json_class_dicts, json_frame_dicts = data['classPrompts'], data['framePrompts']
                for class_dict in json_class_dicts:
                    classes, prompts = [cls.lower() for cls in class_dict['classes']], class_dict['prompts']
                    for cls in classes:
                        for idx, prompt in enumerate(prompts):
                            self.json_class_prompts[cls] = { f'JSON_{idx}':prompt }
                
                for idx, frame_dict in enumerate(json_frame_dicts):
                    self.json_frame_prompts[f'JSON_{idx}'] = frame_dict['prompt']
 
        except Exception as e:
            raise mpf.DetectionException(
                f"Invalid JSON structure for component: {e}",
                mpf.DetectionError.COULD_NOT_READ_DATAFILE
            )

    def _get_pixtral_response(self, prompt, encoded_image):
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": encoded_image}}]}]
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text
        except:
            raise mpf.DetectionException(
                "Could not communicate with Pixtral server: ",
                mpf.DetectionError.NETWORK_ERROR
            )
    
class JobConfig:
    def __init__(self, job_properties: Mapping[str, str]):
        self.prompt_config_path = self._get_prop(job_properties, "PROMPT_CONFIGURATION_PATH", "")
        if self.prompt_config_path == "":
            self.prompt_config_path = os.path.join(os.path.dirname(__file__), 'data', 'crypto_prompts.json')
        
        self.json_prompt_config_path = self._get_prop(job_properties, "JSON_PROMPT_CONFIGURATION_PATH", "")
        if self.json_prompt_config_path == "":
            self.json_prompt_config_path = os.path.join(os.path.dirname(__file__), 'data', 'crypto_json_prompts.json')
        
        self.temperature = self._get_prop(job_properties, "TEMPERATURE", 0.7)
        if self.temperature > 1.0 or self.temperature < 0.0:
            raise mpf.DetectionException(
                "Invalid value for TEMPERATURE. Make sure it is set to a value between 0 and 1, inclusive.",
                mpf.DetectionError.INVALID_PROPERTY
            )
	
    @staticmethod
    def _get_prop(job_properties, key, default_value, accept_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accept_values != []) and (prop not in accept_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptable values: {accept_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

class Timer:
    def __init__(self):
        self._seconds_elapsed = 0.0
        self._last_start_time = None

    def start(self):
        if self._last_start_time is None:
            self._last_start_time = time.perf_counter()

    def pause(self):
        if self._last_start_time is not None:
            self._seconds_elapsed += time.perf_counter() - self._last_start_time
            self._last_start_time = None

    def get_seconds_elapsed_from_last_pause(self) -> float:
        return self._seconds_elapsed

EXPORT_MPF_COMPONENT = PixtralComponent