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
import json
import math
import subprocess
import logging
from typing import Mapping, Iterable
from multiprocessing.shared_memory import SharedMemory
import json
import re

import numpy as np
from tenacity import retry, wait_random_exponential, stop_after_delay, retry_if_exception

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('GeminiComponent')

IGNORE_WORDS = ['unsure', 'none', 'false', 'no', 'unclear', 'n/a', 'unspecified', 'unknown', 'unreadable', 'not visible', 'none visible']
IGNORE_PREFIXES = tuple([s + ' ' for s in IGNORE_WORDS])

class GeminiComponent:
    detection_type = 'CLASS'

    def __init__(self):
        self.gemini_api_key = ''
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
            if config.enable_json_prompt_format:
                detections = self._get_frame_detections_json(image_job, [image_reader.get_image()], config)
            else:
                detections = self._get_frame_detections(image_job, [image_reader.get_image()], config)
        else:
            raise mpf.DetectionException(
                "Feed forward jobs not supported yet: ",
                mpf.DetectionError.DETECTION_FAILED
            )
        
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
            if config.enable_json_prompt_format:
                tracks = self._get_frame_detections_json(video_job, video_capture, config, is_video_job=True)
            else:
                tracks = self._get_frame_detections(video_job, video_capture, config, is_video_job=True)
        else:
            raise mpf.DetectionException(
                "Feed forward jobs not supported yet: ",
                mpf.DetectionError.DETECTION_FAILED
            )

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
        if (mpf_util.get_property(job.job_properties, 'FRAME_RATE_CAP', -1) > 0) and (config.frames_per_second_to_skip > 0):
            raise mpf.DetectionException(
                "Cannot have FRAME_RATE_CAP and GENERATE_FRAME_RATE_CAP both set to values greater than zero on jobs without feed forward detections:",
                mpf.DetectionError.INVALID_PROPERTY
            )

        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)
        self.gemini_api_key = config.gemini_api_key

        tracks = []
        self.frame_count = 0
        self.video_decode_timer = Timer()
        self.video_process_timer = Timer()

        self.video_decode_timer.start()

        for idx, frame in enumerate(reader):
            if (config.frames_per_second_to_skip <= 0) or (idx % config.frames_per_second_to_skip == 0):

                self.video_decode_timer.pause()
                self.frame_count += 1
                height, width, _ = frame.shape
                detection_properties = dict()
                encoded = self._encode_image(frame)
                self.video_process_timer.start()

                for tag, prompt in self.frame_prompts.items():
                    response = self._get_gemini_response(config.model_name, encoded, prompt)
                    detection_properties[tag] = response

                detection_properties['ANNOTATED BY GEMINI'] = True
                self.video_process_timer.pause()
                img_location = mpf.ImageLocation(0, 0, width, height, -1, detection_properties)

                if is_video_job:
                    tracks.append(mpf.VideoTrack(idx, idx, -1, {idx: img_location}, detection_properties))
                else:
                    tracks.append(img_location)

                self.video_decode_timer.start()

        if is_video_job:
            for track in tracks:
                reader.reverse_transform(track)

        return tracks

    def _get_frame_detections_json(self, job, reader, config, is_video_job=False):
        # Check if both frame_rate_cap and generate_frame_rate_cap are set > 0. If so, throw exception
        if (mpf_util.get_property(job.job_properties, 'FRAME_RATE_CAP', -1) > 0) and (config.frames_per_second_to_skip > 0):
            raise mpf.DetectionException(
                "Cannot have FRAME_RATE_CAP and GENERATE_FRAME_RATE_CAP both set to values greater than zero on jobs without feed forward detections:",
                mpf.DetectionError.INVALID_PROPERTY
            )

        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)
        self.gemini_api_key = config.gemini_api_key

        classification = config.classification.strip().lower()

        tracks = []
        self.frame_count = 0
        self.video_decode_timer = Timer()
        self.video_process_timer = Timer()
        self.video_decode_timer.start()

        for idx, frame in enumerate(reader):
            if (config.frames_per_second_to_skip <= 0) or (idx % config.frames_per_second_to_skip == 0):
                self.video_decode_timer.pause()
                self.frame_count += 1
                height, width, _ = frame.shape
                detection_properties = dict()
                encoded = self._encode_image(frame)
                self.video_process_timer.start()

                if classification in self.json_class_prompts:
                    for tag, prompt in self.json_class_prompts[classification].items():
                        json_attempts, json_failed = 0, True
                        while (json_attempts < self.json_limit) and (json_failed):
                            json_attempts += 1
                            response = self._get_gemini_response(config.model_name, encoded, prompt)
                            try:
                                response = response.split('```json\n')[1].split('```')[0]
                                response_json = json.loads(response)
                                self._update_detection_properties(detection_properties, response_json, classification)
                                json_failed = False
                            except Exception as e:
                                logger.warning(f"Gemini failed to produce valid JSON output: {e}")
                                logger.warning(f"Failed {json_attempts} of {self.json_limit} attempts.")
                                continue
                        if json_failed:
                            logger.warning(f"Using last full Gemini response instead of parsed JSON output.")
                            detection_properties['FAILED TO PROCESS GEMINI RESPONSE'] = True
                            detection_properties['FULL GEMINI RESPONSE'] = response

                self.video_process_timer.pause()
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

    def _encode_image(self, frame):
        shape = frame.shape
        dtype = frame.dtype
        shm = SharedMemory(create=True, size=frame.nbytes)
        np_shm = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        np_shm[:] = frame[:]
        return (shm.name, shape, str(dtype)), shm
        
    def _update_detection_properties(self, detection_properties, response_json, classification):

        is_person = (('CLASSIFICATION' in detection_properties) and (detection_properties['CLASSIFICATION'].lower() == 'person')) \
            or (classification == 'person')

        vehicle_classes = ['vehicle', 'car', 'truck', 'bus']
        is_vehicle = (('CLASSIFICATION' in detection_properties) and (detection_properties['CLASSIFICATION'].lower() in vehicle_classes)) \
            or (classification in vehicle_classes)

        key_list = self._get_keys(response_json, True) # TODO: flatten should be an algorithm property or specified in the prompts file
        key_vals = dict()
        keywords = []
        for key_str in key_list:
            split_key = [' '.join(x.split('_')) for x in ('gemini' + key_str).split('||')]
            key, val = " ".join([s.upper() for s in split_key[:-1]]), split_key[-1]
            key_vals[key] = val

        # TODO: Implement this generically to work with any class. Specify rollup class in prompt JSON file.
        ignore_person = is_person and ('GEMINI VISIBLE PERSON' in key_vals) and (self._ignore(key_vals['GEMINI VISIBLE PERSON']))
        ignore_vehicle = is_vehicle and ('GEMINI VISIBLE VEHICLE' in key_vals) and (self._ignore(key_vals['GEMINI VISIBLE VEHICLE']))

        if not ignore_person and not ignore_vehicle:
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
        
        detection_properties['CLASSIFICATION'] = classification.upper()
        detection_properties['ANNOTATED BY GEMINI'] = True
        logger.debug(f"{detection_properties=}")

    def _get_keys(self, response_json, flatten):
        if not response_json:
            yield f'||none'

        elif isinstance(response_json, (str, bool)):
            yield f'||{response_json}'

        elif isinstance(response_json, list):
            yield f'||{json.dumps(response_json)}'

        elif isinstance(response_json, dict):
            if self._is_lowest_level(response_json):

                tmp_response_json = dict(response_json)
                for key, val in response_json.items():
                    if self._ignore(val):
                        tmp_response_json.pop(key)
                response_json = tmp_response_json

                if not response_json:
                    yield f'||none'
                elif flatten:
                    yield from (f'||{key}||{val}' for key, val in response_json.items())
                else:
                    yield f'||{json.dumps(response_json)}'

            else:
                for key, value in response_json.items():
                    if self._ignore(key):
                        yield f'||none'
                    else:
                        yield from (f'||{key}{p}' for p in self._get_keys(value, flatten))

    @staticmethod
    def _is_lowest_level(response_json):
        for key, val in response_json.items():
            if not isinstance(val, str):
                return False
        return True

    @staticmethod
    def _ignore(input):
        return not input or \
            input.strip().lower() in IGNORE_WORDS or \
            input.strip().lower().startswith(IGNORE_PREFIXES)

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

    def _is_rate_limit_error(self, stderr):
        return "Caught a ResourceExhausted error (429 Too Many Requests)" in stderr

    @retry(
        # Each wait is between 4 and multiplier * 2^n seconds, where n is the number of retries. The max wait capped at 32 seconds.
        wait=wait_random_exponential(multiplier=2, max=32, min=4),
        # Stops retrying after the total time waiting >=60s, checks after each attempt
        stop=stop_after_delay(60),
        retry=retry_if_exception(lambda e: isinstance(e, mpf.DetectionException) and getattr(e, 'rate_limit', False))
    )

    def _get_gemini_response(self, model_name, data_uri, prompt):
    
        process = None
        shm = None
        try:
            if isinstance(data_uri, tuple):
                shm_info, shm = data_uri
                shm_name, shape, dtype = shm_info
                process = subprocess.Popen([
                    "/gemini-subprocess/venv/bin/python3",
                    "/gemini-subprocess/gemini-process-image.py",
                    '-m', model_name,
                    "--shm-name", shm_name,
                    "--shm-shape", json.dumps(shape),
                    "--shm-dtype", dtype,
                    "-p", prompt,
                    "-a", self.gemini_api_key],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE)
            else:
                raise mpf.DetectionException(
                    "Expected image data in shared memory format.",
                    mpf.DetectionError.DETECTION_FAILED
                )
            stdout, stderr = process.communicate()
        except Exception as e:
            if shm:
                shm.close()
                shm.unlink()
            raise mpf.DetectionException(
                f"Subprocess error: {e}",
                mpf.DetectionError.DETECTION_FAILED)

        if process.returncode == 0:
            response = stdout.decode()
            logger.info(response)
            return response
        
        stderr_decoded = stderr.decode()
        if self._is_rate_limit_error(stderr_decoded):
            logger.warning("Gemini rate limit hit (429). Retrying with backoff...")
            ex = mpf.DetectionException(
                f"Subprocess failed due to rate limiting: {stderr_decoded}",
                mpf.DetectionError.DETECTION_FAILED
            )
            ex.rate_limit = True
            raise ex
        raise mpf.DetectionException(
            f"Subprocess failed: {stderr_decoded}",
            mpf.DetectionError.DETECTION_FAILED
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

        if not os.path.exists(self.prompt_config_path):
            raise mpf.DetectionException(
                "Invalid path provided for prompt config file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )
        
        self.json_prompt_config_path = self._get_prop(job_properties, "JSON_PROMPT_CONFIGURATION_PATH", "")
        if self.json_prompt_config_path == "":
            self.json_prompt_config_path = os.path.join(os.path.dirname(__file__), 'data', 'json_prompts.json')
        
        self.enable_json_prompt_format = self._get_prop(job_properties, "ENABLE_JSON_PROMPT_FORMAT", False)
        
        self.gemini_api_key = self._get_prop(job_properties, "GEMINI_API_KEY", "")
        generate_frame_rate_cap = self._get_prop(job_properties, "GENERATE_FRAME_RATE_CAP", 1.0)
        self.classification = self._get_prop(job_properties, "CLASSIFICATION", "")
        self.model_name = self._get_prop(job_properties, "MODEL_NAME", "gemma-3-27b-it")

        if (media_properties != None) and (generate_frame_rate_cap > 0):
            # Check if fps exists. If not throw mpf.DetectionError.MISSING_PROPERTY exception
            try:
                self.frames_per_second_to_skip = max(1, math.floor(float(media_properties['FPS']) / generate_frame_rate_cap))
            except Exception as e:
                raise mpf.DetectionException(
                    f"FPS not found for media: {e}",
                    mpf.DetectionError.MISSING_PROPERTY
                )
        else:
            self.frames_per_second_to_skip = -1
	

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

EXPORT_MPF_COMPONENT = GeminiComponent