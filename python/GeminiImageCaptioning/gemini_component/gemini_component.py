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

import base64
import json
import logging
import math
import os
import re
import time

from typing import Mapping, Iterable

import cv2
from openai import OpenAI, RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_delay, retry_if_exception, before_sleep_log

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from google import genai
from google.genai import types
from google.genai.errors import ClientError

logger = logging.getLogger('GeminiComponent')

IGNORE_WORDS = ['unsure', 'none', 'false', 'no', 'unclear', 'n/a', 'unspecified', 'unknown', 'unreadable', 'not visible', 'none visible']
IGNORE_PREFIXES = tuple([s + ' ' for s in IGNORE_WORDS])

class GeminiComponent:
    detection_type = 'CLASS'

    def __init__(self):
        self.class_prompts = dict()
        self.json_class_prompts = dict()
        self.frame_prompts = dict()


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
            if config.enable_json_prompt_format:
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
            if config.enable_json_prompt_format:
                tracks = self._get_frame_detections_json(video_job, video_capture, config, is_video_job=True)
            else:
                tracks = self._get_frame_detections(video_job, video_capture, config, is_video_job=True)
        else:
            if config.enable_json_prompt_format:
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
        if (mpf_util.get_property(job.job_properties, 'FRAME_RATE_CAP', -1) > 0) and (config.frames_per_second_to_skip > 0):
            raise mpf.DetectionException(
                "Cannot have FRAME_RATE_CAP and GENERATE_FRAME_RATE_CAP both set to values greater than zero on jobs without feed forward detections:",
                mpf.DetectionError.INVALID_PROPERTY
            )

        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)

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
                self.video_process_timer.start()

                for tag, prompt in self.frame_prompts.items():
                    response = self._get_model_response(config, frame, prompt)
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
                self.video_process_timer.start()
                json_limit = config.generation_max_attempts

                if classification in self.json_class_prompts:
                    for tag, prompt in self.json_class_prompts[classification].items():
                        json_attempts, json_failed = 0, True
                        while (json_attempts < json_limit) and (json_failed):
                            json_attempts += 1
                            response = self._get_model_response(config, frame, prompt)
                            try:
                                response_json = self._parse_json_response(response)
                                self._update_detection_properties(detection_properties, response_json, classification)
                                json_failed = False
                            except Exception as e:
                                logger.warning(f"Gemini failed to produce valid JSON output: {e}")
                                logger.warning(f"Failed {json_attempts} of {json_limit} attempts.")
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
    
    def _get_feed_forward_detections(self, job_feed_forward, reader, config, is_video_job=False):
        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)

        classification = job_feed_forward.detection_properties["CLASSIFICATION"].lower()

        frame_count = 0
        self.video_decode_timer = Timer()
        self.video_process_timer = Timer()

        if is_video_job:
            self.video_decode_timer.start()
            frame_indices = {i: frame for i, frame in zip(job_feed_forward.frame_locations.keys(), reader)}
            frames_to_process = self._get_frames_to_process(list(frame_indices.keys()), config.frames_per_second_to_skip)
            for idx in frames_to_process:
                self.video_decode_timer.pause()
                frame = frame_indices[idx]
                ff_location = job_feed_forward.frame_locations[idx]
                frame_count += 1

                if classification in self.class_prompts:
                    detection_properties = ff_location.detection_properties

                    for tag, prompt in self.class_prompts[classification].items():
                        response = self._get_model_response(config, frame, prompt)
                        detection_properties[tag] = response
                    detection_properties['CLASSIFICATION'] = classification.upper()
                    detection_properties['ANNOTATED BY GEMINI'] = True

                self.video_decode_timer.start()
            return [job_feed_forward]
        else:
            if classification in self.class_prompts:
                detection_properties = job_feed_forward.detection_properties
                if hasattr(job_feed_forward, 'data_uri'):
                    image = job_feed_forward.data_uri
                else:
                    image = reader.get_image()

                for tag, prompt in self.class_prompts[classification].items():
                    response = self._get_model_response(config, image, prompt)
                    detection_properties[tag] = response
                detection_properties['CLASSIFICATION'] = classification.upper()
                detection_properties['ANNOTATED BY GEMINI'] = True
            return [job_feed_forward]

    def _get_feed_forward_detections_json(self, job_feed_forward, reader, config, is_video_job=False):
        self._update_prompts(config.prompt_config_path, config.json_prompt_config_path)
        json_limit = config.generation_max_attempts

        classification = job_feed_forward.detection_properties["CLASSIFICATION"].lower()
        self.frame_count = 0
        self.video_decode_timer = Timer()
        self.video_process_timer = Timer()
        prompts_to_use = self.json_class_prompts if config.enable_json_prompt_format else self.class_prompts

        if is_video_job:
            self.video_decode_timer.start()
            frame_indices = {i: frame for i, frame in zip(job_feed_forward.frame_locations.keys(), reader)}
            for idx in self._get_frames_to_process(list(frame_indices.keys()), config.frames_per_second_to_skip):
                self.video_decode_timer.pause()
                frame = frame_indices[idx]
                ff_location = job_feed_forward.frame_locations[idx]
                self.frame_count += 1

                if classification in prompts_to_use:
                    for tag, prompt in prompts_to_use[classification].items():
                        json_attempts, json_failed = 0, True

                        while (json_attempts < json_limit) and (json_failed):
                            json_attempts += 1
                            response = self._get_model_response(config, frame, prompt)
                            try:
                                response_json = self._parse_json_response(response)
                                self._update_detection_properties(ff_location.detection_properties, response_json, classification)
                                json_failed = False
                            except Exception as e:
                                logger.warning(f"Gemini failed to produce valid JSON output: {e}")
                                logger.warning(f"Failed {json_attempts} of {json_limit} attempts.")
                                continue
                        if json_failed:
                            logger.warning(f"Using last full Gemini response instead of parsed JSON output.")
                            ff_location.detection_properties['FAILED TO PROCESS GEMINI RESPONSE'] = True
                            ff_location.detection_properties['FULL GEMINI RESPONSE'] = response

                self.video_decode_timer.start()
                
            return [job_feed_forward]
        else:
            image = reader.get_image()
            if classification in prompts_to_use:
                for tag, prompt in prompts_to_use[classification].items():
                    json_attempts, json_failed = 0, True
                    while (json_attempts < json_limit) and (json_failed):
                        json_attempts += 1
                        response = self._get_model_response(config, image, prompt)
                        try:
                            response_json = self._parse_json_response(response)
                            self._update_detection_properties(job_feed_forward.detection_properties, response_json, classification)
                            json_failed = False
                        except Exception as e:
                            logger.warning(f"Gemini failed to produce valid JSON output: {e}")
                            logger.warning(f"Failed {json_attempts} of {json_limit} attempts.")
                            continue
                    if json_failed:
                        logger.warning(f"Using last full Gemini response instead of parsed JSON output.")
                        job_feed_forward.detection_properties['FAILED TO PROCESS GEMINI RESPONSE'] = True
                        job_feed_forward.detection_properties['FULL GEMINI RESPONSE'] = response
            return [job_feed_forward]
        
    def _resize_frame(self, frame, max_dim=4500):
        h, w = frame.shape[:2]
        scale = min(max_dim / w, max_dim / h)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized_frame
        else:
            return frame
 
    @staticmethod
    def _parse_json_response(response: str):
        response = response.strip()
        fenced_match = re.fullmatch(
            r"```(?:json)?\s*(.*?)\s*```", response, re.DOTALL | re.IGNORECASE)
        if fenced_match:
            response = fenced_match.group(1)
        return json.loads(response)

    def _update_detection_properties(self, detection_properties, response_json, classification):

        is_person = (('CLASSIFICATION' in detection_properties) and (detection_properties['CLASSIFICATION'].lower() == 'person')) \
            or (classification == 'person')

        vehicle_classes = ['vehicle', 'car', 'truck', 'bus', 'motorbike']
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

    @staticmethod
    def _get_openai_api_key(application_credentials: str) -> str:
        if application_credentials:
            environment_value = os.environ.get(application_credentials)
            if environment_value:
                return environment_value
            if os.path.exists(application_credentials):
                with open(application_credentials, "r") as api_key_file:
                    api_key = api_key_file.read().strip()
                    if api_key:
                        return api_key
        return "Empty"

    def _encode_frame(self, frame) -> bytes:
        resized_frame = self._resize_frame(frame)
        encoded, image_buffer = cv2.imencode(".jpg", resized_frame)
        if not encoded:
            raise mpf.DetectionException(
                "Failed to encode image as JPEG.",
                mpf.DetectionError.DETECTION_FAILED)
        return image_buffer.tobytes()

    @staticmethod
    def _rate_limit_exception(message: str) -> mpf.DetectionException:
        exception = mpf.DetectionException(
            message, mpf.DetectionError.DETECTION_FAILED)
        exception.rate_limit = True
        return exception

    @retry(
        wait=wait_random_exponential(multiplier=2, max=32, min=4),
        stop=stop_after_delay(60),
        retry=retry_if_exception(
            lambda error: isinstance(error, mpf.DetectionException)
            and getattr(error, "rate_limit", False)),
        before_sleep=before_sleep_log(logger, logging.WARNING))
    def _get_model_response(self, config, frame, prompt):
        image_bytes = self._encode_frame(frame)
        if config.api == "OpenAI":
            return self._get_openai_response(config, image_bytes, prompt)
        return self._get_google_response(config, image_bytes, prompt)

    def _get_openai_response(self, config, image_bytes: bytes, prompt: str) -> str:
        try:
            client = OpenAI(
                api_key=self._get_openai_api_key(config.application_credentials),
                base_url=config.base_url,
                timeout=config.openai_request_timeout_seconds,
                max_retries=config.openai_max_retries)
            image_data = base64.b64encode(image_bytes).decode("ascii")
            request_args = {
                "model": config.model_name,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }]
            }
            if config.openai_response_format_json_object and config.enable_json_prompt_format:
                request_args["response_format"] = {"type": "json_object"}
            if config.openai_max_tokens is not None:
                request_args["max_tokens"] = config.openai_max_tokens
            if config.openai_temperature is not None:
                request_args["temperature"] = config.openai_temperature

            response = client.chat.completions.create(**request_args)
            return response.choices[0].message.content or ""
        except RateLimitError as error:
            logger.warning("OpenAI rate limit hit. Retrying with backoff.")
            raise self._rate_limit_exception(
                "OpenAI API rate limit exceeded.") from error
        except mpf.DetectionException:
            raise
        except Exception as error:
            raise mpf.DetectionException(
                f"OpenAI API call failed: {error}",
                mpf.DetectionError.DETECTION_FAILED) from error

    def _get_google_response(self, config, image_bytes: bytes, prompt: str) -> str:
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.application_credentials
            client = genai.Client(
                project=config.project_id,
                location="global",
                vertexai=True)
            content_config = None
            if config.label_user and config.label_prefix and config.label_purpose:
                content_config = types.GenerateContentConfig(labels={
                    config.label_prefix + "user": config.label_user,
                    config.label_prefix + "purpose": config.label_purpose,
                    config.label_prefix + "modality": "image"
                })
            response = client.models.generate_content(
                model=config.model_name,
                contents=types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=prompt)
                    ]),
                config=content_config)
            return response.text or ""
        except ClientError as error:
            if getattr(error, "code", None) == 429:
                logger.warning("Google rate limit hit. Retrying with backoff.")
                raise self._rate_limit_exception(
                    "Google API rate limit exceeded.") from error
            raise mpf.DetectionException(
                f"Google API call failed: {error}",
                mpf.DetectionError.DETECTION_FAILED) from error
        except mpf.DetectionException:
            raise
        except Exception as error:
            raise mpf.DetectionException(
                f"Google API call failed: {error}",
                mpf.DetectionError.DETECTION_FAILED) from error

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
        self.api = self._get_prop(
            job_properties, "API", "OpenAI", ["OpenAI", "Google"])
        self.base_url = str(self._get_prop(
            job_properties, "OPENAI_BASE_URL", "")).strip() or None
        self.application_credentials = str(self._get_prop(
            job_properties, "APPLICATION_CREDENTIALS", "")).strip()
        if self.api == "Google":
            if not self.application_credentials or not os.path.exists(self.application_credentials):
                raise mpf.DetectionException(
                    "APPLICATION_CREDENTIALS must point to a Google credential file.",
                    mpf.DetectionError.COULD_NOT_OPEN_DATAFILE)
        elif (self.application_credentials
              and not os.path.exists(self.application_credentials)
              and not os.environ.get(self.application_credentials)):
            logger.warning(
                "APPLICATION_CREDENTIALS did not match an environment variable "
                "or file path; using the OpenAI client dummy key fallback.")

        self.project_id = self._get_prop(job_properties, "PROJECT_ID", "")
        self.label_prefix = self._get_prop(job_properties, "LABEL_PREFIX", "")
        self.label_user = self._get_prop(job_properties, "LABEL_USER", "")
        self.label_purpose = self._get_prop(job_properties, "LABEL_PURPOSE", "")

        self.openai_request_timeout_seconds = float(self._get_prop(
            job_properties, "OPENAI_REQUEST_TIMEOUT_SECONDS", "600"))
        self.openai_max_retries = int(self._get_prop(
            job_properties, "OPENAI_MAX_RETRIES",
            "2" if not self.base_url else "0"))
        response_format_default = "true" if not self.base_url else "false"
        response_format_value = self._get_prop(
            job_properties, "OPENAI_RESPONSE_FORMAT_JSON_OBJECT",
            response_format_default)
        self.openai_response_format_json_object = (
            str(response_format_value).strip().lower() == "true")
        max_tokens = str(self._get_prop(
            job_properties, "OPENAI_MAX_TOKENS", "")).strip()
        self.openai_max_tokens = int(max_tokens) if max_tokens else None
        temperature = str(self._get_prop(
            job_properties, "OPENAI_TEMPERATURE", "")).strip()
        self.openai_temperature = float(temperature) if temperature else None

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
        
        self.classification = self._get_prop(job_properties, "CLASSIFICATION", "")
        self.model_name = self._get_prop(job_properties, "MODEL_NAME", "gemma-3-27b-it")
        self.generation_max_attempts = self._get_prop(job_properties, "GENERATION_MAX_ATTEMPTS", 5)

        generate_frame_rate_cap = self._get_prop(job_properties, "GENERATE_FRAME_RATE_CAP", 1.0)
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
