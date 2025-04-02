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
import re
import math

import logging
from typing import Mapping, Iterable

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('LlavaComponent')

IGNORE_WORDS = ['unsure', 'none', 'false', 'no', 'unclear', 'n/a', '']

class LlavaComponent:
    detection_type = 'CLASS'

    def __init__(self):
        self.model = 'llava:34b'
        self.host_url = ''
        self.client = None
        self.class_prompts = dict()
        self.json_class_prompts = dict()
        self.frame_prompts = dict()

        self.json_limit = 3

    def get_detections_from_image(self, image_job: mpf.ImageJob) -> Iterable[mpf.ImageLocation]:
        logger.info('Received image job: %s', image_job.job_name)

        self.video_process_timer = Timer()
        self.video_decode_timer = Timer()
        self.frame_count = 0

        config = JobConfig(image_job.job_properties)
        image_reader = mpf_util.ImageReader(image_job)

        if image_job.feed_forward_location is None:
            detections = self._get_frame_detections(image_job, [image_reader.get_image(),], config)
        elif config.enable_json_prompt_format:
            detections = self._get_feed_forward_detections_json(image_job.feed_forward_location, image_reader, config)
        else:
            detections = self._get_feed_forward_detections(image_job.feed_forward_location, image_reader, config)
        
        logger.info(f"Job complete. Found {len(detections)} detections.")
        return detections

    def get_detections_from_video(self, video_job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        logger.info('Received video job: %s', video_job.job_name)

        self.video_process_timer = Timer()
        self.video_decode_timer = Timer()
        self.frame_count = 0

        config = JobConfig(video_job.job_properties, video_job.media_properties)
        video_capture = mpf_util.VideoCapture(video_job)

        if video_job.feed_forward_track is None:
            tracks = self._get_frame_detections(video_job, video_capture, config, is_video_job=True)
        elif config.enable_json_prompt_format:
            tracks = self._get_feed_forward_detections_json(video_job.feed_forward_track, video_capture, config, is_video_job=True)
        else:
            tracks = self._get_feed_forward_detections(video_job.feed_forward_track, video_capture, config, is_video_job=True)

        decode_time = self.video_decode_timer.get_seconds_elapsed_from_last_pause()
        if decode_time > 0.0:
            logger.info("Total frame load time: "
                        f"{decode_time:0.3f} seconds ({self.frame_count / decode_time:0.3f} frames/second)")

        process_time = self.video_process_timer.get_seconds_elapsed_from_last_pause()
        if process_time > 0.0:
            logger.info("Total detection and tracking time: "
                        f"{process_time:0.3f} seconds ({self.frame_count / process_time:0.3f} frames/second)")

        logger.info(f"Job complete. Found {len(tracks)} tracks.")
        return tracks

    def _get_frame_detections(self, job, reader, config, is_video_job=False):
        # Check if both frame_rate_cap and generate_frame_rate_cap are set > 0. If so, throw exception
        if (mpf_util.get_property(job.job_properties, 'FRAME_RATE_CAP', -1) > 0) and (config.frames_per_second_to_process > 0):
            raise mpf.DetectionException(
                "Cannot have FRAME_RATE_CAP and GENERATE_FRAME_RATE_CAP both set to values greater than zero on jobs without feed forward detections:",
                mpf.DetectionError.INVALID_PROPERTY
            )

        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)
        self._check_client(config.ollama_server)

        tracks = []
        self.frame_count = 0
        self.video_decode_timer = Timer()
        self.video_process_timer = Timer()

        self.video_decode_timer.start()
        for idx, frame in enumerate(reader):
            if (config.frames_per_second_to_process <= 0) or (idx % config.frames_per_second_to_process == 0):
                self.video_decode_timer.pause()
                self.frame_count += 1

                height, width, _ = frame.shape
                detection_properties = dict()

                self._get_ollama_response(self.frame_prompts, frame, detection_properties, self.video_process_timer)

                img_location = mpf.ImageLocation(0, 0, width, height, -1, detection_properties)
                if is_video_job:
                    tracks.append(mpf.VideoTrack(idx, idx, -1, { idx:img_location }, detection_properties))
                else:
                    tracks.append(img_location)

                self.video_decode_timer.start()

        if is_video_job:
            for track in tracks:
                reader.reverse_transform(track)

        return tracks

    def _get_feed_forward_detections_json(self, job_feed_forward, reader, config, is_video_job=False):
        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)
        self._check_client(config.ollama_server)

        classification = job_feed_forward.detection_properties["CLASSIFICATION"].lower()

        # Send prompts to ollama to generate responses
        self.frame_count = 0
        self.video_decode_timer = Timer()
        self.video_process_timer = Timer()

        prompts_to_use = self.json_class_prompts if config.enable_json_prompt_format else self.class_prompts
        if is_video_job:
            self.video_decode_timer.start()
            frame_indices = { i:frame for i, frame in zip(job_feed_forward.frame_locations.keys(), reader) }
            for idx in self._get_frames_to_process(list(frame_indices.keys()), config.frames_per_second_to_process):
                self.video_decode_timer.pause()
                frame = frame_indices[idx]
                ff_location = job_feed_forward.frame_locations[idx]
                self.frame_count += 1

                encoded = self._encode_image(frame)
                if classification in prompts_to_use:
                    for tag, prompt in prompts_to_use[classification].items():
                        # re-initialize json_attempts=0 and json_failed=True
                        json_attempts, json_failed = 0, True
                        while (json_attempts < self.json_limit) and (json_failed):
                            json_attempts += 1
                            response = self._get_ollama_response_json(prompt, encoded)
                            try:
                                response = response.split('```json\n')[1].split('```')[0]
                                response_json = json.loads(response)
                                self._update_detection_properties(ff_location.detection_properties, response_json)
                                json_failed = False
                            except Exception as e:
                                logger.warning(f"LLaVA failed to produce valid JSON output: {e}")
                                logger.warning(f"Failed {json_attempts} of {self.json_limit} attempts.")
                                continue
                        if json_failed:
                            logger.warning(f"Using last full LLaVA response instead of parsed JSON output.")
                            job_feed_forward.detection_properties['FAILED TO PROCESS LLAVA RESPONSE'] = True
                            job_feed_forward.detection_properties['FULL LLAVA RESPONSE'] = response

                    self.video_decode_timer.start()
        else:
            encoded = self._encode_image(reader.get_image())
            if classification in prompts_to_use:
                for tag, prompt in prompts_to_use[classification].items():
                    json_attempts, json_failed = 0, True
                    while (json_attempts < self.json_limit) and (json_failed):
                        json_attempts += 1
                        response = self._get_ollama_response_json(prompt, encoded)
                        try:
                            response = response.split('```json\n')[1].split('```')[0]
                            response_json = json.loads(response)
                            self._update_detection_properties(job_feed_forward.detection_properties, response_json)
                            json_failed = False
                        except Exception as e:
                            logger.warning(f"LLaVA failed to produce valid JSON output: {e}")
                            logger.warning(f"Failed {json_attempts} of {self.json_limit} attempts.")
                            continue
                    if json_failed:
                        logger.warning(f"Using last full LLaVA response instead of parsed JSON output.")
                        job_feed_forward.detection_properties['FAILED TO PROCESS LLAVA RESPONSE'] = True
                        job_feed_forward.detection_properties['FULL LLAVA RESPONSE'] = response

        return [job_feed_forward]

    def _update_detection_properties(self, detection_properties, response_json):
        key_list = self._get_keys(response_json)
        key_vals = dict()
        keywords = []
        for key_str in key_list:
            split_key = [' '.join(x.split('_')) for x in ('llava' + key_str).split('||')]
            key, val = " ".join([s.upper() for s in split_key[:-1]]), split_key[-1]
            key_vals[key] = val

        if ('LLAVA VISIBLE PERSON' not in key_vals) or (key_vals['LLAVA VISIBLE PERSON'].strip().lower() not in IGNORE_WORDS):
            ret_key_vals = dict(key_vals)
            for key, val in key_vals.items():
                if ('VISIBLE' in key) and (val.strip().lower() in IGNORE_WORDS):
                    keywords.append(key.split(' VISIBLE ')[1])
                    ret_key_vals.pop(key)

            # TODO: Test removal of features if "VISIBLE" feature is to be ignored
            for keyword in keywords:
                pattern = re.compile(fr'\b{keyword}\b')
                for key_to_remove in filter(pattern.search, key_vals):
                    ret_key_vals.pop(key_to_remove)
            
            for key, val in key_vals.items():
                if val.strip().lower() in IGNORE_WORDS:
                    ret_key_vals.pop(key)
            
            detection_properties.update(ret_key_vals)
            detection_properties['ANNOTATED BY LLAVA'] = True

    def _get_keys(self, response_json):
        if not response_json:
            yield f'||none'

        elif isinstance(response_json, str):
            yield f'||{response_json}'

        elif isinstance(response_json, list):
            yield f'||{json.dumps(response_json)}'

        elif isinstance(response_json, dict):
            if self._is_lowest_level(response_json):

                # TODO: Check every element and drop if ignorable
                return_response_json = dict(response_json)
                for key, val in response_json.items():
                    if val.strip().lower() in IGNORE_WORDS:
                        return_response_json.pop(key)

                if not return_response_json:
                    yield f'||none'
                else:
                    yield f'||{json.dumps(return_response_json)}'

            else:
                for key, value in response_json.items():
                    yield from (f'||{key}{p}' for p in self._get_keys(value))
        
    @staticmethod
    def _is_lowest_level(response_json):
        for key, val in response_json.items():
            if not isinstance(val, str):
                return False
        return True

    def _get_feed_forward_detections(self, job_feed_forward, reader, config, is_video_job=False):
        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)
        self._check_client(config.ollama_server)

        classification = job_feed_forward.detection_properties["CLASSIFICATION"].lower()
        frame_count = 0
        video_decode_timer = Timer()
        video_process_timer = Timer()

        if is_video_job:
            video_decode_timer.start()
            frame_indices = { i:frame for i, frame in zip(job_feed_forward.frame_locations.keys(), reader) }
            frames_to_process = self._get_frames_to_process(list(frame_indices.keys()), config.frames_per_second_to_process)
            for idx in frames_to_process:
                video_decode_timer.pause()
                frame = frame_indices[idx]
                ff_location = job_feed_forward.frame_locations[idx]
                frame_count += 1

                if classification in self.class_prompts:
                    self._get_ollama_response(self.class_prompts[classification], frame, ff_location.detection_properties, video_process_timer)

                video_decode_timer.start()
            return [job_feed_forward]
        else:
            if classification in self.class_prompts:
                self._get_ollama_response(self.class_prompts[classification], reader.get_image(), job_feed_forward.detection_properties, video_process_timer)
            return [job_feed_forward]

    def _check_client(self, host_url):
        try:
            if self.client is None or host_url != self.host_url:
                self.host_url = host_url
                self.client = ollama.Client(host=self.host_url)
        except Exception as e:
            raise mpf.DetectionException(
                f"Could not instantiate Ollama Client. Make sure OLLAMA_SERVER is set correctly: {e}",
                mpf.DetectionError.NETWORK_ERROR
            )

    def _encode_image(self, image):
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        _, buffer = cv2.imencode('.png', image, encode_params)
        return base64.b64encode(buffer).decode("utf-8")

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
                json_class_dicts = data['classPrompts']
                for class_dict in json_class_dicts:
                    classes, prompts = [cls.lower() for cls in class_dict['classes']], class_dict['prompts']
                    for cls in classes:
                        for idx, prompt in enumerate(prompts):
                            self.json_class_prompts[cls] = { f'JSON_{idx}':prompt }
 
        except Exception as e:
            raise mpf.DetectionException(
                f"Invalid JSON structure for component: {e}",
                mpf.DetectionError.COULD_NOT_READ_DATAFILE
            )

    def _get_ollama_response_json(self, prompt, encoded_image):
        try:
            self.video_process_timer.start()
            response = self.client.generate(self.model, prompt, images=[encoded_image])['response']
            logger.debug(f"Ollama response:\n{response}")
            self.video_process_timer.pause()
            return response
        except Exception as e:
            raise mpf.DetectionException(
                f"Could not communicate with Ollama server: {e}",
                mpf.DetectionError.NETWORK_ERROR
            )
        
    def _get_ollama_response(self, prompt_dict, image, detection_properties, video_process_timer):
        try:
            encoded = self._encode_image(image)
            for tag, prompt in prompt_dict.items():
                video_process_timer.start()
                response = self.client.generate(self.model, prompt, images=[encoded])['response']
                logger.debug(f"Ollama response:\n{response}")
                detection_properties[tag] = response
                video_process_timer.pause()
            detection_properties['ANNOTATED BY LLAVA'] = True
            
        except Exception as e:
            raise mpf.DetectionException(
                f"Could not communicate with Ollama server: {e}",
                mpf.DetectionError.NETWORK_ERROR
                )

    def _get_frames_to_process(self, frame_locations: list, skip: int) -> list:
        if not frame_locations:
            return []
        
        retval = []
        curr = frame_locations[0]
        retval.append(curr)
        want = curr + skip

        for i in range(1, len(frame_locations)):
            
            next = math.inf
            if i + 1 < len(frame_locations):
                next = frame_locations[i + 1]

            if next < want:
                continue

            curr = frame_locations[i]

            curr_delta = abs(want - curr)
            next_delta = abs(next - want)

            too_close_to_last = (curr - retval[-1]) <= (skip / 3)

            if curr_delta <= next_delta and not too_close_to_last:
                retval.append(curr)
                want = curr + skip
                continue

            if next != math.inf:
                retval.append(next)
                want = next + skip

        return retval
class JobConfig:
    def __init__(self, job_properties: Mapping[str, str], media_properties=None):
        self.prompt_config_path = self._get_prop(job_properties, "PROMPT_CONFIGURATION_PATH", "")
        if self.prompt_config_path == "":
            self.prompt_config_path = os.path.join(os.path.dirname(__file__), 'data', 'prompts.json')
        
        self.json_prompt_config_path = self._get_prop(job_properties, "JSON_PROMPT_CONFIGURATION_PATH", "")
        if self.json_prompt_config_path == "":
            self.json_prompt_config_path = os.path.join(os.path.dirname(__file__), 'data', 'json_prompts.json')
        
        self.enable_json_prompt_format = self._get_prop(job_properties, "ENABLE_JSON_PROMPT_FORMAT", False)

        self.ollama_server = self._get_prop(job_properties, "OLLAMA_SERVER", "llava-detection-server:11434")
        if not os.path.exists(self.prompt_config_path):
            raise mpf.DetectionException(
                "Invalid path provided for prompt config JSON file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )
 
        generate_frame_rate_cap = self._get_prop(job_properties, "GENERATE_FRAME_RATE_CAP", 1.0)
        if (media_properties != None) and (generate_frame_rate_cap > 0):
            # Check if fps exists. If not throw mpf.DetectionError.MISSING_PROPERTY exception
            try:
                self.frames_per_second_to_process = max(1, math.floor(float(media_properties['FPS']) / generate_frame_rate_cap))
            except:
                raise mpf.DetectionException(
                    "FPS not found for media: ",
                    mpf.DetectionError.MISSING_PROPERTY
                )
        else:
            self.frames_per_second_to_process = -1
	

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

EXPORT_MPF_COMPONENT = LlavaComponent
