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
        # self.json_class_prompts = dict()
        self.frame_prompts = dict()

        # self.json_failed = True
        # self.json_attempts = 0
        # self.json_limit = 3

    def get_detections_from_image(self, image_job: mpf.ImageJob) -> Iterable[mpf.ImageLocation]:
        logger.info('Received image job: %s', image_job.job_name)

        self.video_process_timer = Timer()
        self.video_decode_timer = Timer()
        self.frame_count = 0

        config = JobConfig(image_job.job_properties)
        image_reader = mpf_util.ImageReader(image_job)

        if image_job.feed_forward_location is None:
            return self._get_frame_detections([image_reader.get_image(),], config)
        # elif config.prompt_json_format:
        #     return self._get_feed_forward_detections_json(image_job.feed_forward_location, image_reader, config)
        else:
            return self._get_feed_forward_detections(image_job.feed_forward_location, image_reader, config)

    def get_detections_from_video(self, video_job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        logger.info('Received video job: %s', video_job.job_name)

        self.video_process_timer = Timer()
        self.video_decode_timer = Timer()
        self.frame_count = 0

        config = JobConfig(video_job.job_properties)
        video_capture = mpf_util.VideoCapture(video_job)

        if video_job.feed_forward_track is None:
            tracks, video_process_timer, video_decode_timer, frame_count = self._get_frame_detections(video_capture, config, is_video_job=True)
        # elif config.prompt_json_format:
        #     tracks, video_process_timer, video_decode_timer, frame_count = self._get_feed_forward_detections_json(video_job.feed_forward_track, video_capture, config, is_video_job=True)
        else:
            tracks, video_process_timer, video_decode_timer, frame_count = self._get_feed_forward_detections(video_job.feed_forward_track, video_capture, config, is_video_job=True)

        decode_time = video_decode_timer.get_seconds_elapsed_from_last_pause()
        logger.info("Total frame load time: "
                    f"{decode_time:0.3f} seconds ({frame_count / decode_time:0.3f} frames/second)")

        process_time = video_process_timer.get_seconds_elapsed_from_last_pause()
        logger.info("Total detection and tracking time: "
                    f"{process_time:0.3f} seconds ({frame_count / process_time:0.3f} frames/second)")

        return tracks

    def _get_frame_detections(self, reader, config, is_video_job=False):
        self._update_prompts(config.prompt_config_path, config.prompt_json_config_path)
        self._check_client(config.ollama_server)

        tracks = []
        frame_count = 0
        video_decode_timer = Timer()
        video_process_timer = Timer()

        video_decode_timer.start()
        for idx, frame in enumerate(reader):
            video_decode_timer.pause()
            frame_count += 1

            height, width, _ = frame.shape
            detection_properties = dict()

            self._get_ollama_response(self.frame_prompts, frame, detection_properties, video_process_timer)

            img_location = mpf.ImageLocation(0, 0, width, height, -1, detection_properties)
            if is_video_job:
                tracks.append(mpf.VideoTrack(idx, idx, -1, { idx:img_location }, detection_properties))
            else:
                tracks.append(img_location)

            video_decode_timer.start()

        if is_video_job:
            for track in tracks:
                reader.reverse_transform(track)
            return tracks, video_process_timer, video_decode_timer, frame_count
        return tracks

    # def _get_feed_forward_detections_json(self, job_feed_forward, reader, config, is_video_job=False):
    #     self._update_prompts(config.prompt_config_path, config.prompt_json_config_path)
    #     self._check_client(config.ollama_server)

    #     classification = job_feed_forward.detection_properties["CLASSIFICATION"].lower()

    #     # Send prompts to ollama to generate responses
    #     frame_count = 0
    #     video_decode_timer = Timer()
    #     video_process_timer = Timer()

    #     prompts_to_use = self.json_class_prompts if config.prompt_json_format else self.class_prompts
    #     if is_video_job:
    #         video_decode_timer.start()
    #         for frame, ff_location in zip(reader, job_feed_forward.frame_locations.values()):
    #             video_decode_timer.pause()
    #             frame_count += 1

    #             encoded = self._encode_image(frame)
    #             for tag, prompt in prompts_to_use[classification].items():
    #                 # re-initialize json_attempts=0 and json_failed=True
    #                 while (self.json_attempts <= self.json_limit) and (self.json_failed):
    #                     self.json_attempts += 1
    #                     response = self._get_ollama_response_json(prompt, encoded, video_process_timer)

    #                     try:
    #                         response_json = json.loads(response)
    #                         # Read through json, create keys and add to detection properties
    #                         for key, val in response_json.items():
    #                             ff_location.detection_properties['LLAVA_' + key.upper()] = val

    #                         self.json_failed = False
    #                     except:
    #                         logger.warning(f"LLaVA failed to produce valid JSON output. Failed {self.json_attempts} of {self.json_limit} attempts.")
    #                         continue
    #                 # Check if json_failed=True. If so, add 'FAILED TO PROCESS LLAVA RESPONSE' and 'FULL LLAVA RESPONSE' to detection_properties
    #                 # logger.warning(f"Using last full LLaVA response instead of parsed JSON output.")

    #             video_decode_timer.start()
    #     else:
    #         encoded = self._encode_image(reader.get_image())
    #         for tag, prompt in prompts_to_use.items():
    #             while (self.json_attempts <= self.json_limit) and (self.json_failed):
    #                 self.json_attempts += 1
    #                 response = self._get_ollama_response_json(prompt, encoded, video_process_timer)
    #                 logger.info(response)
    #                 try:
    #                     response_json = json.loads(response)
    #                     for key, val in response_json.items():
    #                         job_feed_forward.detection_properties['LLAVA_' + key.upper()] = val
    #                     self.json_failed = False
    #                 except:
    #                     continue

    #     if is_video_job:
    #         return [job_feed_forward], video_process_timer, video_decode_timer, frame_count
    #     else:
    #       return [job_feed_forward]

    def _get_feed_forward_detections(self, job_feed_forward, reader, config, is_video_job=False):
        self._update_prompts(config.prompt_config_path, config.prompt_json_config_path)
        self._check_client(config.ollama_server)

        classification = job_feed_forward.detection_properties["CLASSIFICATION"].lower()
        # Send prompts to ollama to generate responses
        frame_count = 0
        video_decode_timer = Timer()
        video_process_timer = Timer()

        if is_video_job:
            video_decode_timer.start()
            for frame, ff_location in zip(reader, job_feed_forward.frame_locations.values()):
                video_decode_timer.pause()
                frame_count += 1

                if classification in self.class_prompts:
                    self._get_ollama_response(self.class_prompts[classification], frame, ff_location.detection_properties, video_process_timer)

                video_decode_timer.start()
            return [job_feed_forward], video_process_timer, video_decode_timer, frame_count
        else:
            if classification in self.class_prompts:
                self._get_ollama_response(self.class_prompts[classification], reader.get_image(), job_feed_forward.detection_properties, video_process_timer)
            return [job_feed_forward]

    def _check_client(self, host_url):
        try:
            if self.client is None or host_url != self.host_url:
                self.host_url = host_url
                self.client = ollama.Client(host=self.host_url)
        except:
            raise mpf.DetectionException(
                "Could not instantiate Ollama Client. Make sure OLLAMA_SERVER is set correctly: ",
                mpf.DetectionError.NETWORK_ERROR
            )

    def _encode_image(self, image):
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        _, buffer = cv2.imencode('.png', image, encode_params)
        return base64.b64encode(buffer).decode("utf-8")

    def _update_prompts(self, prompt_config_path, prompt_json_config_path):
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
            
            # with open(prompt_json_config_path, 'r') as f:
            #     data = json.load(f)
            #     json_class_dicts = data['classPrompts']
            #     for class_dict in json_class_dicts:
            #         classes, prompt = [cls.lower() for cls in class_dict['classes']], class_dict['prompt']
            #         for cls in classes:
            #             self.class_prompts[cls] = { 'JSON':prompt }
 
        except:
            raise mpf.DetectionException(
                "Invalid JSON structure for component: ",
                mpf.DetectionError.COULD_NOT_READ_DATAFILE
            )

    # def _get_ollama_response_json(self, prompt, encoded_image, video_process_timer):
    #     try:
    #         video_process_timer.start()
    #         response = self.client.generate(self.model, prompt, images=[encoded_image])['response']
    #         video_process_timer.pause()
    #         return response
    #     except:
    #         raise mpf.DetectionException(
    #             "Could not communicate with Ollama server: ",
    #             mpf.DetectionError.NETWORK_ERROR
    #         )
        
    def _get_ollama_response(self, prompt_dict, image, detection_properties, video_process_timer):
        try:
            encoded = self._encode_image(image)
            for tag, prompt in prompt_dict.items():
                video_process_timer.start()
                detection_properties[tag] = self.client.generate(self.model, prompt, images=[encoded])['response']
                video_process_timer.pause()
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
        
        self.prompt_json_config_path = self._get_prop(job_properties, "PROMPT_JSON_CONFIGURATION_PATH", "")
        if self.prompt_json_config_path == "":
            self.prompt_json_config_path = os.path.join(os.path.dirname(__file__), 'data', 'json_prompts.json')
        
        self.prompt_json_format = self._get_prop(job_properties, "PROMPT_JSON_FORMAT", False)

        self.ollama_server = self._get_prop(job_properties, "OLLAMA_SERVER", "llava-detection-server:11434")
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

EXPORT_MPF_COMPONENT = LlavaComponent
