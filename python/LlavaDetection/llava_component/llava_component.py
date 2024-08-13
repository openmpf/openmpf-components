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

import os
import cv2
import base64
import json
import ollama

import logging
from typing import Mapping, Iterable

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('LlavaComponent')

class LlavaComponent:
    detection_type = 'CLASS'

    def __init__(self):
        self.model = 'llava:34b'
        self.host_url = ''
        self.client = None
        self.class_prompts = dict()

    def get_detections_from_image(self, image_job: mpf.ImageJob) -> Iterable[mpf.ImageLocation]:
        logger.info('Received image job: %s', image_job.job_name)

        image_reader = mpf_util.ImageReader(image_job)
        kwargs = JobConfig(image_job.job_properties)
        return self._get_feed_forward_detections(image_job.feed_forward_location, image_reader, kwargs)
    
    def get_detections_from_video(self, video_job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        logger.info('Received video job: %s', video_job.job_name)

        video_capture = mpf_util.VideoCapture(video_job)
        kwargs = JobConfig(video_job.job_properties)
        return self._get_feed_forward_detections(video_job.feed_forward_track, video_capture, kwargs, is_video_job=True)

    def _get_feed_forward_detections(self, job_feed_forward, reader, kwargs, is_video_job=False):
        if job_feed_forward is None: 
            raise mpf.DetectionException(
                "Component can only process feed forward jobs, but no feed forward track provided.",
                mpf.DetectionError.UNSUPPORTED_DATA_TYPE
            )
    
        # Get job properties
        self._update_class_prompts(kwargs.prompt_config_path)

        try:
            if self.client is None or kwargs.ollama_client_host_url != self.host_url:
                self.host_url = kwargs.ollama_client_host_url
                self.client = ollama.Client(host=self.host_url)
        except:
            raise mpf.DetectionException(
                "Could not instantiate Ollama Client: ",
                mpf.DetectionError.NETWORK_ERROR
            )
        
        # Send prompts to ollama to generate responses
        if is_video_job:
            for frame, ff_location in zip(reader, job_feed_forward.frame_locations.values()):
                self._get_ollama_response(frame, ff_location)
        else:
            self._get_ollama_response(reader.get_image(), job_feed_forward)

        return [job_feed_forward]

    def _update_class_prompts(self, prompt_config_path):
        '''
        Updates self.class_prompts dictionary to have the following format

        {
            CLASS1: {TAG1: PROMPT1},
            CLASS2: {TAG2: PROMPT2, TAG3: PROMPT3},
            ...
        }
        '''
        try:
            with open(prompt_config_path, 'r') as f:
                data = json.load(f)
                for class_dict in data:
                    classes, prompts = [cls.lower() for cls in class_dict['classes']], class_dict['prompts']
                    for cls in classes:
                        if cls not in self.class_prompts:
                            self.class_prompts[cls] = dict()
                        self.class_prompts[cls].update({ dct['detectionProperty']:dct['prompt'] for dct in prompts })
        except:
            raise mpf.DetectionException(
                "Invalid JSON structure for component: ",
                mpf.DetectionError.COULD_NOT_READ_DATAFILE
            )

    def _get_ollama_response(self, image, ff_location):
        '''
        Communicates with ollama server to generate LLaVA responses and adds them to 
        the feed forward location's detection_properties dict.
        '''
        # Encode image to bytestring
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        _, buffer = cv2.imencode('.png', image, encode_params)
        b64_bytearr = base64.b64encode(buffer).decode("utf-8")

        classification = ff_location.detection_properties["CLASSIFICATION"]
        if classification.lower() in self.class_prompts:
            for tag, prompt in self.class_prompts[classification.lower()].items():
                try:
                    ff_location.detection_properties[tag] = self.client.generate(self.model, prompt, images=[b64_bytearr])['response']
                except:
                    raise mpf.DetectionException(
                        "Could not communicate with Ollama server: ",
                        mpf.DetectionError.NETWORK_ERROR
                    )

class JobConfig:
    def __init__(self, job_properties: Mapping[str, str]):
        self.prompt_config_path = self._get_prop(job_properties, "PROMPT_CONFIGURATION_PATH", "")
        if self.prompt_config_path == "":
            self.prompt_config_path = os.path.join(os.path.dirname(__file__), 'data', 'prompts.json')
        
        self.ollama_client_host_url = self._get_prop(job_properties, "OLLAMA_CLIENT_HOST_URL", "llava-detection-server")

        if not os.path.exists(self.prompt_config_path):
            raise mpf.DetectionException(
                "Invalid path provided for prompt config JSON file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )

    @staticmethod
    def _get_prop(job_properties, key, default_value, accept_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accept_values != []) and (prop not in accept_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptible values: {accept_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

EXPORT_MPF_COMPONENT = LlavaComponent