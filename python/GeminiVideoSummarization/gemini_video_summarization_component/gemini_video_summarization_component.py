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

from __future__ import annotations # Postpones annotation eval
import os
import json
import logging
import base64
import cv2
from typing import Iterable, Mapping, Tuple, Union
from tenacity import retry, wait_random_exponential, stop_after_delay, retry_if_exception

import mpf_component_api as mpf
import mpf_component_util as mpf_util


from google import genai
from google.genai import types
from google.genai.types import Part
from google.cloud import storage
from google.genai.errors import ClientError

from openai import OpenAI, OpenAIError

import torch
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

logger = logging.getLogger('GeminiVideoSummarizationComponent')

class GeminiVideoSummarizationComponent:

    def __init__(self, model: Gemma4ForConditionalGeneration=None, processor: Gemma4Processor=None, API="OpenAI", device=None):
        self.model = model
        self.processor = processor
        self.device = device
        
        if self.model is not None:
            assert self.processor is not None, "If a model is provided, a tokenizer must also be provided."
        if self.model is not None and self.device is None:
            self.device = model.device
        
        self.api = API
        self.application_credentials = ''
        self.project_id = ''
        self.bucket_name = ''
        self.label_prefix = ''
        self.label_user = ''
        self.label_purpose = ''  

    def get_detections_from_video(self, job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        logger.info('Received video job: %s', job.job_name)

        if job.feed_forward_track:
            raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                'Component cannot process feed forward jobs.')

        if job.stop_frame < 0:
            raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                'Job stop frame must be >= 0.')
            
        config = JobConfig(job.job_properties, job.media_properties, model=True)
        
        tracks = []
            
        self.application_credentials = config.application_credentials
        self.project_id = config.project_id
        self.bucket_name = config.bucket_name
        self.label_prefix = config.label_prefix
        self.label_user = config.label_user
        self.label_purpose = config.label_purpose

        fps = config.process_fps
        enable_timeline=config.enable_timeline

        segment_start_time = job.start_frame / float(job.media_properties['FPS'])
        segment_stop_time = (job.stop_frame + 1) / float(job.media_properties['FPS'])
        
        prompt = _read_file(config.generation_prompt_path)

        model_name = config.model_name

        max_attempts = int(config.generation_max_attempts)
        timeline_check_target_threshold = int(config.timeline_check_target_threshold)
        
        error = None
        attempts = dict(
            base=0,
            timeline=0)

        while max(attempts.values()) < max_attempts:
            error= None
            if self.model is None: response = self._get_response(job, prompt, model_name, fps)
            else: response = self._local_get_response(job, prompt)
            
            logger.info(f'Gemini response received.: {response}')
            if '```json\n' in response and '```' in response:
                try:
                    response = response.split('```json\n')[1].split('```')[0]
                except IndexError:
                    # Fallback if splitting fails unexpectedly
                    error = "Invalid response format"
                    continue
            response_json, error = self._check_response(attempts, max_attempts, response)
            if error is not None:
                continue

            if enable_timeline == 1:
                event_timeline = response_json['video_event_timeline'] 
                error = self._check_timeline(
                    timeline_check_target_threshold, attempts, max_attempts, segment_start_time, segment_stop_time, event_timeline)
                if error is not None:
                    continue

            break

        if error:
            raise mpf.DetectionError.DETECTION_FAILED.exception(f'Failed to produce valid JSON file: {error}')
            
        tracks = self._create_tracks(job, response_json, enable_timeline)

        logger.info(f"Job complete. Found {len(tracks)} tracks.")
        return tracks

    def _is_rate_limit_error(self, stderr):
        return "Caught a ResourceExhausted error (429 Too Many Requests)" in stderr

    @retry(
        # Each wait is between 4 and multiplier * 2^n seconds, where n is the number of retries. The max wait capped at 32 seconds.
        wait=wait_random_exponential(multiplier=2, max=32, min=4),
        # Stops retrying after the total time waiting >=60s, checks after each attempt
        stop=stop_after_delay(60),
        # Retries if it detects an exception AND rate_limit is true
        # If e.rate_limit exists, getattr returns its actual value (True or False)
        # Else if e.rate_limit does NOT exist, getattr returns False
        retry=retry_if_exception(lambda e: isinstance(e, mpf.DetectionException) and getattr(e, 'rate_limit', False))
    )

    def _create_tracks(self, job: mpf.VideoJob, response_json: dict, enable_timeline) -> Iterable[mpf.VideoTrack]:
        logger.info('Creating tracks.')
        tracks = []

        segment_id = str(job.start_frame) + "-" + str(job.stop_frame)
        video_fps = float(job.media_properties['FPS'])
        segment_start_time = job.start_frame / video_fps
        
        frame_width = 0
        frame_height = 0
        if 'FRAME_WIDTH' in job.media_properties:
            frame_width = int(job.media_properties['FRAME_WIDTH'])
        if 'FRAME_HEIGHT' in job.media_properties:
            frame_height = int(job.media_properties['FRAME_HEIGHT'])

        if enable_timeline == 1:
            summary_track = self._create_segment_summary_track(job, response_json)
            tracks.append(summary_track)

            for event in response_json['video_event_timeline']:

                # get offset start/stop times in milliseconds
                event_start_time = self.convert_mm_ss_to_seconds(event["timestamp_start"], segment_start_time) * 1000
                event_stop_time = self.convert_mm_ss_to_seconds(event["timestamp_end"], segment_start_time) * 1000

                offset_start_frame = int((event_start_time * video_fps) / 1000)
                offset_stop_frame = int((event_stop_time * video_fps) / 1000) - 1

                detection_properties={
                    "SEGMENT ID": segment_id,
                    "TEXT": event['description']
                }

                # check offset_stop_frame
                if offset_stop_frame > job.stop_frame:
                    logger.debug(f'offset_stop_frame outside of acceptable range '
                              f'({offset_stop_frame} > {job.stop_frame}), setting offset_stop_frame to {job.stop_frame}')
                    offset_stop_frame = job.stop_frame
                elif offset_stop_frame < job.start_frame:
                    logger.debug(f'offset_stop_frame outside of acceptable range '
                              f'({offset_stop_frame} < {job.start_frame}), setting offset_stop_frame to {job.start_frame}')
                    offset_stop_frame = job.start_frame

                # check offset_start_frame
                if offset_start_frame > job.stop_frame:
                    logger.debug(f'offset_start_frame outside of acceptable range '
                              f'({offset_start_frame} > {job.stop_frame}), setting offset_start_frame to {job.stop_frame}')
                    offset_start_frame = job.stop_frame
                elif offset_start_frame < job.start_frame:
                    logger.debug(f'offset_start_frame outside of acceptable range '
                              f'({offset_start_frame} < {job.start_frame}), setting offset_start_frame to {job.start_frame}')
                    offset_start_frame = job.start_frame

                offset_middle_frame = int((offset_stop_frame - offset_start_frame) / 2) + offset_start_frame

                # check offset_middle_frame
                if offset_middle_frame > job.stop_frame:
                    logger.debug(f'offset_middle_frame outside of acceptable range '
                              f'({offset_middle_frame} > {job.stop_frame}), setting offset_middle_frame to {job.stop_frame}')
                    offset_middle_frame = job.stop_frame
                elif offset_middle_frame < job.start_frame:
                    logger.debug(f'offset_middle_frame outside of acceptable range '
                              f'({offset_middle_frame} < {job.start_frame}), setting offset_middle_frame to {job.start_frame}')
                    offset_middle_frame = job.start_frame

                track = mpf.VideoTrack(
                    offset_start_frame, 
                    offset_stop_frame, 
                    1.0,
                    # Add start and top frame locations to prevent the Workflow Manager from dropping / truncating track.
                    # Add middle frame for artifact extraction.
                    frame_locations = {
                        offset_start_frame:  mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0),
                        offset_middle_frame: mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0),
                        offset_stop_frame:   mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0)
                    },
                    detection_properties = detection_properties
                )

                track.frame_locations[offset_middle_frame].detection_properties["EXEMPLAR"] = "1"
                
                tracks.append(track)

        else: # no events timeline, create summary only
            tracks.append(self._create_segment_summary_track(job, response_json))
            
        logger.info('Processing complete. Video segment %s summarized in %d tracks.' % (segment_id, len(tracks)))
        return tracks

    def _create_segment_summary_track(self, job: mpf.VideoJob, response_json: dict) -> mpf.VideoTrack:
        start_frame = job.start_frame
        stop_frame = job.stop_frame
        
        segment_id = str(job.start_frame) + "-" + str(job.stop_frame)
        detection_properties={
            "SEGMENT ID": segment_id,
            "SEGMENT SUMMARY": "TRUE",
            "TEXT": response_json['video_summary']
        }
        frame_width = 0
        frame_height = 0
        if 'FRAME_WIDTH' in job.media_properties:
            frame_width = int(job.media_properties['FRAME_WIDTH'])
        if 'FRAME_HEIGHT' in job.media_properties:
            frame_height = int(job.media_properties['FRAME_HEIGHT'])

        middle_frame = int((stop_frame - start_frame) / 2) + start_frame

        track = mpf.VideoTrack(
            start_frame, 
            stop_frame, 
            1.0,
            # Add start and top frame locations to prevent the Workflow Manager from dropping / truncating track.
            # Add middle frame for artifact extraction.
            frame_locations = {
                start_frame:  mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0),
                middle_frame: mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0),
                stop_frame:   mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0)
            },
            detection_properties = detection_properties
        )

        track.frame_locations[middle_frame].detection_properties["EXEMPLAR"] = "1"

        return track

    def _check_response(self, attempts: dict, max_attempts: int, response: str
                        ) -> Tuple[Union[dict, None], Union[str, None]]:
        response_json = None

        if not response:
            error = 'Empty response.'
            logger.warning(error)
            logger.warning(f'Failed {attempts["base"] + 1} of {max_attempts} base attempts.')
            attempts['base'] += 1
            return None, error

        try:
            response_json = json.loads(response)
        except ValueError as ve:
            error = 'Response is not valid JSON.'
            logger.warning(error)
            logger.warning(str(ve))
            logger.warning(f'Failed {attempts["base"] + 1} of {max_attempts} base attempts.')
            attempts['base'] += 1
            return response_json, error
        
        return response_json, None


    def _check_timeline(self, threshold: float, attempts: dict, max_attempts: int,
                        segment_start_time: float, segment_stop_time: float, event_timeline: list
                        ) -> Union[str, None]:

        error = None

        if not event_timeline:
            error = 'No timeline events found in response.'
            logger.warning(error)
            logger.warning(f'Failed {attempts["timeline"] + 1} of {max_attempts} timeline attempts.')
            attempts['timeline'] += 1
            return error
        
        for event in event_timeline:

            try:
                timestamp_start = self.convert_mm_ss_to_seconds(event["timestamp_start"], segment_start_time)
                timestamp_end = self.convert_mm_ss_to_seconds(event["timestamp_end"], segment_start_time)

                if timestamp_start < 0:
                    error = (f'Timeline event start time of {timestamp_start} < 0.')
                    break
            
                if timestamp_end < 0:
                    error = (f'Timeline event end time of {timestamp_end} < 0.')
                    break

                if timestamp_end < timestamp_start:
                    error = (f'Timeline event end time is less than event start time. '
                            f'{timestamp_end} < {timestamp_start}.')
                    break
                
                if threshold != -1:

                    if (segment_start_time - timestamp_start) > threshold:
                        error = (f'Timeline event start time occurs too soon before segment start time. '
                                f'({segment_start_time} - {timestamp_start}) > {threshold}.')
                        break

                    if (timestamp_end - segment_stop_time) > threshold:
                        error = (f'Timeline event end time occurs too late after segment stop time. '
                                f'({timestamp_end} - {segment_stop_time}) > {threshold}.')
                        break
                    
            except Exception as e:
                error = (f'Timestamps could not be converted: {e}')
                break
        
        if threshold != -1:
            if not error:
                min_event_start = min(list(map(lambda d: float(self.convert_mm_ss_to_seconds(d.get('timestamp_start'), segment_start_time)),
                                            filter(lambda d: 'timestamp_start' in d, event_timeline))))

                if abs(segment_start_time - min_event_start) > threshold:
                    error = (f'Min timeline event start time not close enough to segment start time. '
                            f'abs({segment_start_time} - {min_event_start}) > {threshold}.')

            if not error:
                max_event_end = max(list(map(lambda d: float(self.convert_mm_ss_to_seconds(d.get('timestamp_end'), segment_start_time)),
                                            filter(lambda d: 'timestamp_end' in d, event_timeline))))

                if abs(max_event_end - segment_stop_time) > threshold:
                    error = (f'Max timeline event end time not close enough to segment stop time. '
                            f'abs({max_event_end} - {segment_stop_time}) > {threshold}.')
        if error:
            logger.warning(error)
            logger.warning(f'Failed {attempts["timeline"] + 1} of {max_attempts} timeline attempts.')
            attempts['timeline'] += 1
            return error

        return None

    def convert_mm_ss_to_seconds(self, timestamp_str, segment_start_time):
        try:
            minutes_str, seconds_str = timestamp_str.split(':')
            minutes = int(minutes_str)
            seconds = int(seconds_str)

            total_seconds = (minutes * 60) + seconds + segment_start_time
            return total_seconds
        except ValueError:
            raise ValueError("Invalid timestamp format.")
        except Exception as e:
            raise Exception(f"An unexpected error occurred: {e}")
        
    def _local_get_response(self, job: mpf.VideoJob, prompt: str) -> str:
        try:
            PROMPT = prompt

            # Video segment storage information
            FILE_PATH = job.data_uri
            
            # Processing metadata
            VIDEO_FPS = float(job.media_properties['FPS'])
            SEGMENT_START = job.start_frame / VIDEO_FPS
            SEGMENT_STOP = job.stop_frame / VIDEO_FPS

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": FILE_PATH},
                        {"type": "text", "text": PROMPT},
                        {"type": "metadata", "metadata": {
                                "start_offset": f"{SEGMENT_START:.2f}s",
                                "end_offset": f"{SEGMENT_STOP:.2f}s",
                                "fps": VIDEO_FPS
                        }}
                    ]
                }
            ]

            # 4. Apply the template and prepare model inputs
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(videos=FILE_PATH, text=text_prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=256)
                
            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            return response

        except ClientError as e:
            if hasattr(e, 'code') and e.code == 429:
                logger.warning("Gemini rate limit hit (429). Retrying with backoff...")
                ex = mpf.DetectionException(
                    "Gemini API rate limit (429 Too Many Requests)",
                    mpf.DetectionError.DETECTION_FAILED
                )
                ex.rate_limit = True
                raise ex
            raise
        except Exception as e:
            logger.error(f"Error in _local_get_gemini_response: {e}")
            raise mpf.DetectionException(
                f"{type(self.model)} failed to execute: {e}",
                mpf.DetectionError.DETECTION_FAILED
            )
            
    def _encode_frame_as_jpeg(self, frame, quality=85):
        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not success:
            raise ValueError("Failed to encode video frame as JPEG.")

        return base64.b64encode(buffer).decode("utf-8")
            
    def _openai_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float) -> str:
        if not model_name:
            raise mpf.DetectionException(
                "MODEL_NAME must be provided for OpenAI API requests.",
                mpf.DetectionError.INVALID_PROPERTY
            )

        api_key = _read_file(self.application_credentials).strip()
        
        # Video segment storage information
        FILE_PATH = job.data_uri
        FILE_NAME = os.path.basename(FILE_PATH)
        
        VIDEO_FPS = float(job.media_properties['FPS'])
        SEGMENT_START = job.start_frame / VIDEO_FPS
        SEGMENT_STOP = job.stop_frame / VIDEO_FPS
        
        # Generate OpenAI response
        client = OpenAI(api_key=api_key)
        reader = mpf_util.VideoCapture(job)
        
        content = [
            {
                "type": "text",
                "text": (
                    f"{prompt}\n\n"
                    f"The following frames are sampled from video segment "
                    f"{SEGMENT_START:.2f}s to {SEGMENT_STOP:.2f}s. "
                    f"Frame sample FPS: {fps}. "
                    "Frames are provided in chronological order."
                )
            }
        ]
        
        while True:
            segment_frame = reader.current_frame_position
            timestamp_seconds = SEGMENT_START + (reader.current_time_in_millis / 1000.0)

            success, frame = reader.read()
            
            if not success:
                break
            
            encoded_frame = self._encode_frame_as_jpeg(frame)
            content.append({
                "type": "text",
                "text": f"Frame {segment_frame}, timestamp {timestamp_seconds:.2f}s"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_frame}"
                }
            })

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            response_format={"type": "json_object"},
        )
        
        print(response.choices[0].message.content)
        return response.choices[0].message.content
        
    def _google_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float) -> str:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.application_credentials
        
        # Video segment storage information
        FILE_PATH = job.data_uri
        FILE_NAME = os.path.basename(FILE_PATH)
        STORAGE_PATH = self.label_user + "/" + FILE_NAME

        VIDEO_FPS = float(job.media_properties['FPS'])
        SEGMENT_START = job.start_frame / VIDEO_FPS
        SEGMENT_STOP = job.stop_frame / VIDEO_FPS
        
        # Uploads file to GCP bucket
        client = storage.Client(project=self.project_id)
        bucket = client.bucket(self.bucket_name)
        blob = bucket.blob(STORAGE_PATH)
        blob.upload_from_filename(FILE_PATH)

        file_uri = f"gs://{self.bucket_name}/{STORAGE_PATH}"

        # Generate Gemini response
        genai_client = genai.Client(
            project=self.project_id,
            location="global",
            vertexai=True
        )

        content_config = None
        if self.label_user and self.label_prefix and self.label_purpose:
            content_config = types.GenerateContentConfig(
                labels={
                    self.label_prefix + "user": self.label_user,
                    self.label_prefix + "purpose": self.label_purpose,
                    self.label_prefix + "modality": "video"
                }
            )

        response = genai_client.models.generate_content(
            model=model_name,
            contents=types.Content(
                role='user',
                parts=[
                    Part(
                        file_data=types.FileData(
                            file_uri=file_uri,
                            mime_type='video/mp4'
                        ),
                        video_metadata=types.VideoMetadata(
                            start_offset=f"{SEGMENT_START:.2f}s",
                            end_offset=f"{SEGMENT_STOP:.2f}s",
                            fps=fps
                        )
                    ),
                    Part(
                        text=prompt
                    )
                ],
            ),
            config=content_config
        )
        return response.text
    
    def _get_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float):
        try:
            if self.api == "OpenAI":
                return self._openai_response(job, prompt, model_name, fps)
            elif self.api == "Google":
                return self._google_response(job, prompt, model_name, fps)
            else:
                raise mpf.DetectionException(
                    f"Unsupported API specified: {self.api}",
                    mpf.DetectionError.INVALID_PROPERTY
                )
            
        except ClientError as e:
            if hasattr(e, 'code') and e.code == 429:
                logger.warning("Gemini rate limit hit (429). Retrying with backoff...")
                ex = mpf.DetectionException(
                    "Gemini API rate limit (429 Too Many Requests)",
                    mpf.DetectionError.DETECTION_FAILED
                )
                ex.rate_limit = True
                raise ex
            raise
        
        except Exception as e:
            logger.error(f"Error in _get_response: {e}")
            raise mpf.DetectionException(
                f"{self.api} API call failed: {e}",
                mpf.DetectionError.DETECTION_FAILED
            )

def _read_file(path: str) -> str:
    try:
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, path)
        with open(path, 'r') as file:
            return file.read()
    except Exception as e:
        raise mpf.DetectionError.COULD_NOT_READ_DATAFILE.exception(
            f"Could not read \"{path}\": {e}"
        ) from e

class JobConfig:
    def __init__(self, job_properties: Mapping[str, str], media_properties=None, model=False):
        self.generation_prompt_path = self._get_prop(job_properties, "GENERATION_PROMPT_PATH", "")
        self.enable_timeline = int(self._get_prop(job_properties, "ENABLE_TIMELINE", "1"))
        
        if self.generation_prompt_path == "" and self.enable_timeline == 1:
            self.generation_prompt_path= os.path.join(os.path.dirname(__file__), 'data', 'default_prompt.txt')
        elif self.generation_prompt_path == "" and self.enable_timeline == 0:
            self.generation_prompt_path= os.path.join(os.path.dirname(__file__), 'data', 'default_prompt_no_tl.txt')

        if not os.path.exists(self.generation_prompt_path):
            raise mpf.DetectionException(
                "Invalid path provided for prompt file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )

        self.application_credentials = self._get_prop(job_properties, "APPLICATION_CREDENTIALS", "")
        if not os.path.exists(self.application_credentials) and not model:
            raise mpf.DetectionException(
                "Invalid path provided for GCP credential file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )

        self.model_name = self._get_prop(job_properties, "MODEL_NAME", "")
        self.project_id = self._get_prop(job_properties, "PROJECT_ID", "")
        self.bucket_name = self._get_prop(job_properties, "BUCKET_NAME", "")
        self.label_prefix = self._get_prop(job_properties, "LABEL_PREFIX", "")
        self.label_user = self._get_prop(job_properties, "LABEL_USER", "")
        self.label_purpose = self._get_prop(job_properties, "LABEL_PURPOSE", "")
        self.generation_max_attempts = self._get_prop(job_properties, "GENERATION_MAX_ATTEMPTS", "5")
        self.timeline_check_target_threshold = self._get_prop(job_properties, "TIMELINE_CHECK_TARGET_THRESHOLD", "10")
        self.process_fps = self._get_prop(job_properties, "PROCESS_FPS", 1.0)

    @staticmethod
    def _get_prop(job_properties, key, default_value, accept_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accept_values != []) and (prop not in accept_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptable values: {accept_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

EXPORT_MPF_COMPONENT = GeminiVideoSummarizationComponent