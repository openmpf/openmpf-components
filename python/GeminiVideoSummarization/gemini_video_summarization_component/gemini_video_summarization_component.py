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
import itertools
import math
import shutil
import subprocess
import tempfile
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

from openai import OpenAI


logger = logging.getLogger('GeminiVideoSummarizationComponent')

MOTION_SCORE_WIDTH = 320

_MISSING_CHAT_TEMPLATE_ERRORS = (
    "chat_template is not set",
    "does not have a chat template",
)


class GeminiVideoSummarizationComponent:

    def __init__(self):
        self._last_preprocessed_frame_timestamps = []
        self._last_preprocessed_fps = None

    def get_detections_from_video(self, job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        logger.info('Received video job: %s', job.job_name)

        feed_forward_tracks = getattr(job, 'feed_forward_tracks', None) or []
        if feed_forward_tracks:
            raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                'Feed-forward tracks are not supported by this component.')
        if job.stop_frame < 0:
            raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                'Job stop frame must be >= 0.')

        has_local_model = getattr(self, "model", None) is not None
        config = JobConfig(
            job.job_properties,
            job.media_properties,
            model=has_local_model)

        tracks = []

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
            if has_local_model: response = self._local_get_response(job, prompt)
            else: response = self._get_response(job, prompt, model_name, fps, config)

            response = self._extract_json_object(response)
            response_json, error = self._check_response(attempts, max_attempts, response)
            if error is not None:
                continue

            if enable_timeline == 1:
                self._remap_response_timestamps_from_preprocessed_video(
                    response_json, segment_stop_time - segment_start_time)
                event_timeline = response_json['video_event_timeline']
                error = self._check_timeline(
                    timeline_check_target_threshold, attempts, max_attempts, segment_start_time, segment_stop_time, event_timeline)
                if error is not None:
                    continue

            break

        if error:
            raise mpf.DetectionError.DETECTION_FAILED.exception(f'Failed to produce valid JSON file: {error}')

        tracks = self._create_tracks(job, response_json, enable_timeline)
        response_json = json.dumps(response_json, indent=4)
        logger.info(f'Gemini response received.: {response_json}')
        logger.info(f"Job complete. Found {len(tracks)} tracks.")
        return tracks

    def _extract_json_object(self, response: str) -> str:
        if not response:
            return response

        response = response.strip()
        if response.startswith("```"):
            lines = response.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        json_start = response.find("{")
        json_end = response.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            return response[json_start:json_end + 1]

        return response

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

    def _remap_response_timestamps_from_preprocessed_video(self, response_json: dict, original_duration_seconds: float):
        frame_timestamps = self._last_preprocessed_frame_timestamps
        preprocessed_fps = self._last_preprocessed_fps
        if not frame_timestamps or not preprocessed_fps:
            return

        event_timeline = response_json.get('video_event_timeline')
        if not event_timeline:
            return

        raw_times = []
        try:
            for event in event_timeline:
                raw_times.append(self._parse_mm_ss_to_seconds(event['timestamp_start']))
                raw_times.append(self._parse_mm_ss_to_seconds(event['timestamp_end']))
        except Exception as error:
            logger.warning('Unable to parse model timestamps before remap: %s', error)
            return

        max_raw_time = max(raw_times) if raw_times else 0
        preprocessed_duration_seconds = len(frame_timestamps) / preprocessed_fps
        force_remap = max_raw_time > original_duration_seconds

        if force_remap:
            logger.info(
                'Forcing timestamp remap because model timestamp %.2fs exceeds original duration %.2fs. '
                'Preprocessed duration is %.2fs.',
                max_raw_time, original_duration_seconds, preprocessed_duration_seconds)
        else:
            logger.info(
                'Keeping model timestamps without remap because all timestamps fit original duration %.2fs. '
                'Preprocessed duration is %.2fs.',
                original_duration_seconds, preprocessed_duration_seconds)
            return

        first_original_timestamp = frame_timestamps[0]
        for event in event_timeline:
            try:
                original_start = self._map_preprocessed_seconds_to_original_segment_seconds(
                    self._parse_mm_ss_to_seconds(event['timestamp_start']),
                    frame_timestamps,
                    preprocessed_fps,
                    first_original_timestamp,
                    round_up=False)
                original_end = self._map_preprocessed_seconds_to_original_segment_seconds(
                    self._parse_mm_ss_to_seconds(event['timestamp_end']),
                    frame_timestamps,
                    preprocessed_fps,
                    first_original_timestamp,
                    round_up=True)
            except Exception as error:
                logger.warning('Unable to remap preprocessed timestamps for event %s: %s', event, error)
                continue

            event['PREPROCESSED_TIMESTAMP_START'] = event['timestamp_start']
            event['PREPROCESSED_TIMESTAMP_END'] = event['timestamp_end']
            event['timestamp_start'] = self._format_seconds_as_mm_ss(original_start, round_up=False)
            event['timestamp_end'] = self._format_seconds_as_mm_ss(max(original_start, original_end), round_up=True)
            logger.info(
                'Remapped event timestamps from preprocessed %s-%s to original %s-%s.',
                event['PREPROCESSED_TIMESTAMP_START'], event['PREPROCESSED_TIMESTAMP_END'],
                event['timestamp_start'], event['timestamp_end'])

    @staticmethod
    def _map_preprocessed_seconds_to_original_segment_seconds(
        preprocessed_seconds: float,
        frame_timestamps,
        preprocessed_fps: float,
        first_original_timestamp: float,
        round_up: bool
    ) -> float:
        raw_frame_index = preprocessed_seconds * preprocessed_fps
        if round_up:
            frame_index = math.ceil(raw_frame_index)
        else:
            frame_index = math.floor(raw_frame_index)
        frame_index = max(0, min(frame_index, len(frame_timestamps) - 1))
        return max(0.0, frame_timestamps[frame_index] - first_original_timestamp)

    @staticmethod
    def _parse_mm_ss_to_seconds(timestamp_str) -> int:
        minutes_str, seconds_str = str(timestamp_str).split(':')
        minutes = int(minutes_str)
        seconds = int(seconds_str)
        return (minutes * 60) + seconds

    @staticmethod
    def _format_seconds_as_mm_ss(seconds: float, round_up: bool) -> str:
        if round_up:
            seconds = math.ceil(seconds)
        else:
            seconds = math.floor(seconds)
        seconds = max(0, int(seconds))
        return f'{seconds // 60}:{seconds % 60:02d}'

    def _get_local_num_frames(
        self,
        job: mpf.VideoJob,
        available_frame_count: Union[int, None] = None
    ) -> Union[int, None]:
        video_processor = getattr(self.processor, "video_processor", None)
        requested_num_frames = getattr(video_processor, "num_frames", None)
        try:
            requested_num_frames = int(requested_num_frames)
        except (TypeError, ValueError):
            return None
        if requested_num_frames <= 0:
            return None

        frame_limits = [requested_num_frames]

        if available_frame_count is not None:
            try:
                available_frame_count = int(available_frame_count)
            except (TypeError, ValueError):
                available_frame_count = None
            if available_frame_count is not None and available_frame_count > 0:
                frame_limits.append(available_frame_count)
        else:
            segment_frame_count = job.stop_frame - job.start_frame + 1
            if segment_frame_count > 0:
                frame_limits.append(segment_frame_count)

            media_frame_count = self._get_media_frame_count(job)
            if media_frame_count is not None and media_frame_count > 0:
                frame_limits.append(media_frame_count)

        num_frames = max(1, min(frame_limits))
        return self._avoid_torchvision_endpoint_frame_index(available_frame_count, num_frames)

    @staticmethod
    def _avoid_torchvision_endpoint_frame_index(
        available_frame_count: Union[int, None],
        num_frames: int
    ) -> int:
        if available_frame_count is None or available_frame_count <= 0:
            return num_frames

        num_frames = min(num_frames, available_frame_count)
        while num_frames > 1:
            step = available_frame_count / num_frames
            indices = [int(index * step) for index in range(num_frames)]
            if len(indices) <= num_frames and max(indices) < available_frame_count:
                return num_frames
            num_frames -= 1

        return 1

    def _get_media_frame_count(self, job: mpf.VideoJob) -> Union[int, None]:
        frame_counts = []

        try:
            frame_count = int(float((job.media_properties or {})["FRAME_COUNT"]))
            if frame_count > 0:
                frame_counts.append(frame_count)
        except (KeyError, TypeError, ValueError):
            pass

        video = cv2.VideoCapture(job.data_uri)
        try:
            if video.isOpened():
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    frame_counts.append(frame_count)
        finally:
            video.release()

        return min(frame_counts) if frame_counts else None

    def _apply_local_chat_template(self, messages: list) -> str:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            if not any(error_text in str(e) for error_text in _MISSING_CHAT_TEMPLATE_ERRORS):
                raise
            logger.warning(
                "Processor tokenizer has no chat template; using Gemma 4 local prompt fallback."
            )
            return self._format_gemma4_chat_prompt(messages, add_generation_prompt=True)

    @staticmethod
    def _format_gemma4_chat_prompt(messages: list, add_generation_prompt: bool = True) -> str:
        prompt_parts = ["<bos>"]
        for message in messages:
            role = "model" if message.get("role") == "assistant" else message.get("role", "user")
            prompt_parts.append(f"<|turn>{role}\n")

            content = message.get("content", "")
            if isinstance(content, str):
                prompt_parts.append(content.strip())
            else:
                for item in content:
                    content_type = item.get("type")
                    if content_type == "text":
                        prompt_parts.append(item.get("text", "").strip())
                    elif content_type == "video":
                        prompt_parts.append("\n\n<|video|>\n\n")
                    elif content_type == "image":
                        prompt_parts.append("\n\n<|image|>\n\n")
                    elif content_type == "audio":
                        prompt_parts.append("<|audio|>")

            prompt_parts.append("<turn|>\n")

        if add_generation_prompt:
            prompt_parts.append("<|turn>model\n")

        return "".join(prompt_parts)

    def _local_get_response(self, job: mpf.VideoJob, prompt: str) -> str:
        preprocessed_video = None
        try:
            # Processing metadata
            VIDEO_FPS = float(job.media_properties['FPS'])
            SEGMENT_START = job.start_frame / VIDEO_FPS
            SEGMENT_STOP = (job.stop_frame + 1) / VIDEO_FPS
            preprocessed_video = self._preprocess_video_for_model(job)
            preprocessed_video_path = preprocessed_video['path']

            include_timestamp_instructions = int(mpf_util.get_property(
                job.job_properties, "ENABLE_TIMELINE", "1")) == 1
            preprocessing_prompt = self._build_preprocessed_timing_prompt(
                preprocessed_video,
                include_timestamp_instructions=include_timestamp_instructions)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": preprocessed_video_path},
                        {
                            "type": "text",
                            "text": f"{prompt}\n\n{preprocessing_prompt}"
                        }
                    ]
                }
            ]
            text_prompt = self._apply_local_chat_template(messages)
            processor_kwargs = {
                "videos": preprocessed_video_path,
                "text": text_prompt,
                "return_tensors": "pt"
            }
            local_num_frames = self._get_local_num_frames(
                job,
                len(preprocessed_video.get('frame_timestamps', []))
            )
            if local_num_frames is not None:
                processor_kwargs["num_frames"] = local_num_frames

            inputs = self.processor(**processor_kwargs).to(self.device)

            import torch
            with torch.no_grad():
                output_ids = self.model.generate(**inputs, max_new_tokens=1024)

            generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response

        except Exception as e:
            logger.error(f"Error in _local_get_response: {e}")
            raise mpf.DetectionException(
                f"{type(self.model)} failed to execute: {e}",
                mpf.DetectionError.DETECTION_FAILED
            )
        finally:
            self._cleanup_preprocessed_video(preprocessed_video)

    @staticmethod
    def _get_video_capture_job(job) -> mpf.VideoJob:
        if isinstance(job, mpf.VideoJob):
            return job
        return mpf.VideoJob(
            job.job_name,
            job.data_uri,
            job.start_frame,
            job.stop_frame,
            job.job_properties,
            job.media_properties,
        )

    @staticmethod
    def _track_has_speech_summarization_text_algorithm(track: mpf.VideoTrack) -> bool:
        properties = track.detection_properties or {}
        output_type = properties.get('FEED_FORWARD_OUTPUT_TYPE', '')
        algorithm = properties.get('FEED_FORWARD_ALGORITHM', '')
        if output_type.upper() == 'TEXT' and 'SPEECHSUMMARIZATION' in algorithm.upper():
            return True

        raw_output_json = properties.get('OPENMPF_OUTPUT_JSON')
        if raw_output_json:
            try:
                output_json = json.loads(raw_output_json)
                output_type = str(output_json.get('outputType', output_type))
                algorithm = str(output_json.get('algorithm', algorithm))
                if output_type.upper() == 'TEXT' and 'SPEECHSUMMARIZATION' in algorithm.upper():
                    return True
            except json.JSONDecodeError:
                logger.warning('Unable to parse OPENMPF_OUTPUT_JSON feed-forward property; using SDK track fields.')

        return False

    @classmethod
    def _get_feed_forward_speech_summary_tracks(cls, job) -> list[mpf.VideoTrack]:
        return [
            track
            for track in getattr(job, 'feed_forward_tracks', []) or []
            if cls._track_has_speech_summarization_text_algorithm(track)
        ]

    @classmethod
    def _has_speech_summarization_text_output(cls, job) -> bool:
        return bool(cls._get_feed_forward_speech_summary_tracks(job))

    @staticmethod
    def _parse_feed_forward_json(raw_json: str, property_name: str):
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning('Unable to parse %s feed-forward property; using SDK track fields.', property_name)
            return None

    @staticmethod
    def _get_openmpf_track_properties(track_json: dict) -> dict[str, str]:
        properties = track_json.get('trackProperties') or {}
        if not properties:
            exemplar = track_json.get('exemplar') or {}
            properties = exemplar.get('detectionProperties') or {}
        return {
            str(key): str(value)
            for key, value in properties.items()
            if value is not None
        }

    @classmethod
    def _iter_openmpf_tracks_with_voiced_segments(cls, output_json) -> Iterable[dict]:
        if not isinstance(output_json, dict):
            return

        if isinstance(output_json.get('tracks'), list):
            for track_json in output_json.get('tracks') or []:
                if not isinstance(track_json, dict):
                    continue
                properties = cls._get_openmpf_track_properties(track_json)
                if properties.get('VOICED_SEGMENTS'):
                    yield track_json

        for outputs in output_json.values():
            if not isinstance(outputs, list):
                continue
            for output in outputs:
                if isinstance(output, dict):
                    yield from cls._iter_openmpf_tracks_with_voiced_segments(output)

    @classmethod
    def _iter_feed_forward_speaker_tracks(cls, track: mpf.VideoTrack) -> Iterable[dict]:
        properties = track.detection_properties or {}
        raw_media_output_json = properties.get('OPENMPF_MEDIA_OUTPUT_JSON')
        raw_output_json = properties.get('OPENMPF_OUTPUT_JSON')
        raw_track_json = properties.get('OPENMPF_TRACK_JSON')

        for property_name, raw_json in (
                ('OPENMPF_MEDIA_OUTPUT_JSON', raw_media_output_json),
                ('OPENMPF_OUTPUT_JSON', raw_output_json),
                ('OPENMPF_TRACK_JSON', raw_track_json)):
            if not raw_json:
                continue
            output_json = cls._parse_feed_forward_json(raw_json, property_name)
            yield from cls._iter_openmpf_tracks_with_voiced_segments(output_json)

        if properties.get('VOICED_SEGMENTS'):
            yield {
                'id': properties.get('SPEAKER_ID'),
                'startOffsetFrame': track.start_frame,
                'stopOffsetFrame': track.stop_frame,
                'trackProperties': properties,
            }

    @staticmethod
    def _parse_voiced_segments(voiced_segments: str) -> list[tuple[int, int]]:
        segments = []
        for segment in str(voiced_segments).split(','):
            segment = segment.strip()
            if not segment or '-' not in segment:
                continue

            start_text, stop_text = segment.split('-', 1)
            try:
                start_ms = int(float(start_text.strip()))
                stop_ms = int(float(stop_text.strip()))
            except ValueError:
                logger.warning('Unable to parse VOICED_SEGMENTS entry: %s', segment)
                continue

            if stop_ms < start_ms:
                logger.warning('Skipping VOICED_SEGMENTS entry with stop before start: %s', segment)
                continue
            segments.append((start_ms, stop_ms))
        return segments

    @staticmethod
    def _format_timestamp(milliseconds: int, *, round_up: bool = False) -> str:
        seconds = milliseconds / 1000.0
        total_seconds = int(math.ceil(seconds) if round_up else math.floor(seconds))
        total_seconds = max(0, total_seconds)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f'{hours:02d}:{minutes:02d}:{seconds:02d}'
        return f'{minutes:02d}:{seconds:02d}'

    @staticmethod
    def _get_speaker_label(speaker_index: int) -> str:
        if speaker_index < 26:
            return f'Speaker {chr(ord("A") + speaker_index)}'
        return f'Speaker {speaker_index + 1}'

    @staticmethod
    def _get_speaker_context(properties: dict[str, str]) -> str:
        for key in ('SPEAKER_ROLE', 'ROLE', 'SPEAKER_CONTEXT', 'CONTEXT', 'SPEAKER_DESCRIPTION', 'DESCRIPTION'):
            value = properties.get(key)
            if value:
                return value.strip()
        return ''

    @classmethod
    def _build_feed_forward_speaker_timeline_prompt(cls, speech_tracks: list[mpf.VideoTrack]) -> str:
        speaker_ids = {}
        rows = []
        seen_tracks = set()

        for track in sorted(speech_tracks, key=lambda item: item.start_frame):
            for source_track in cls._iter_feed_forward_speaker_tracks(track):
                properties = cls._get_openmpf_track_properties(source_track)
                voiced_segments = properties.get('VOICED_SEGMENTS', '')
                speaker_id = properties.get('SPEAKER_ID') or source_track.get('id') or f'track-{len(seen_tracks)}'
                track_key = (source_track.get('id'), speaker_id, voiced_segments)
                if track_key in seen_tracks:
                    continue
                seen_tracks.add(track_key)

                if speaker_id not in speaker_ids:
                    speaker_ids[speaker_id] = cls._get_speaker_label(len(speaker_ids))
                speaker_label = speaker_ids[speaker_id]
                context = cls._get_speaker_context(properties)

                for start_ms, stop_ms in cls._parse_voiced_segments(voiced_segments):
                    rows.append({
                        'start_ms': start_ms,
                        'stop_ms': stop_ms,
                        'time_range': (
                            f'{cls._format_timestamp(start_ms)} - '
                            f'{cls._format_timestamp(stop_ms, round_up=True)}'
                        ),
                        'speaker_label': speaker_label,
                        'context': context,
                    })

        if not rows:
            return ''

        rows.sort(key=lambda item: (item['start_ms'], item['stop_ms'], item['speaker_label']))
        include_context = any(row['context'] for row in rows)
        if include_context:
            table_rows = [
                '| Time Range | Speaker Label | Context/Role (Optional) |',
                '| :--- | :--- | :--- |',
            ]
            table_rows.extend(
                f"| {row['time_range']} | {row['speaker_label']} | {row['context']} |"
                for row in rows
            )
        else:
            table_rows = [
                '| Time Range | Speaker Label |',
                '| :--- | :--- |',
            ]
            table_rows.extend(
                f"| {row['time_range']} | {row['speaker_label']} |"
                for row in rows
            )

        return '\n'.join(table_rows)

    def _get_preprocess_fps(self, job: mpf.VideoJob, source_fps: float) -> float:
        process_fps = mpf_util.get_property(job.job_properties, 'PROCESS_FPS', 1.0)
        try:
            process_fps = float(process_fps)
        except (TypeError, ValueError) as error:
            raise ValueError(f'PROCESS_FPS must be a positive number: {process_fps}') from error

        if process_fps <= 0:
            raise ValueError(f'PROCESS_FPS must be positive: {process_fps}')

        return min(source_fps, process_fps)

    def _iter_sampled_video_frames(
            self, job: mpf.VideoJob, target_fps: float, motion_profiling: bool = False):
        reader = mpf_util.VideoCapture(self._get_video_capture_job(job))
        sample_interval_seconds = 1.0 / target_fps
        next_sample_time = None
        sample_index = 0
        try:
            max_frames = max(0, job.stop_frame - job.start_frame + 1)
            frames_read = 0
            while frames_read < max_frames:
                timestamp_seconds = reader.current_time_in_millis / 1000.0
                success, frame = reader.read()
                if not success:
                    break

                if next_sample_time is None:
                    next_sample_time = timestamp_seconds

                if timestamp_seconds + 1e-6 >= next_sample_time:
                    yield sample_index, timestamp_seconds, frame
                    sample_index += 1
                    while next_sample_time <= timestamp_seconds + 1e-6:
                        next_sample_time += sample_interval_seconds

                frames_read += 1
        finally:
            reader.release()

    def _iter_motion_sampling_video_frames(self, job: mpf.VideoJob, target_fps: float, score_fps: float):
        yield from self._iter_read_motion_sampling_video_frames(job, target_fps, score_fps)

    def _iter_read_motion_sampling_video_frames(self, job: mpf.VideoJob, target_fps: float, score_fps: float):
        reader = mpf_util.VideoCapture(self._get_video_capture_job(job))
        sample_interval_seconds = 1.0 / target_fps
        score_interval_seconds = 1.0 / score_fps
        next_sample_time = None
        next_score_time = None
        try:
            max_frames = max(0, job.stop_frame - job.start_frame + 1)
            frames_read = 0
            while frames_read < max_frames:
                timestamp_seconds = reader.current_time_in_millis / 1000.0
                success, frame = reader.read()
                if not success:
                    break

                if next_sample_time is None:
                    next_sample_time = timestamp_seconds
                if next_score_time is None:
                    next_score_time = timestamp_seconds

                is_regular_sample = timestamp_seconds + 1e-6 >= next_sample_time
                is_score_sample = timestamp_seconds + 1e-6 >= next_score_time

                if is_regular_sample:
                    while next_sample_time <= timestamp_seconds + 1e-6:
                        next_sample_time += sample_interval_seconds
                if is_score_sample:
                    while next_score_time <= timestamp_seconds + 1e-6:
                        next_score_time += score_interval_seconds

                if is_regular_sample or is_score_sample:
                    yield frames_read, timestamp_seconds, frame, is_regular_sample, is_score_sample

                frames_read += 1
        finally:
            reader.release()

    def _get_configured_motion_emphasis_threshold(self, job: mpf.VideoJob):
        configured_threshold = mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_THRESHOLD', '5.0')
        if configured_threshold is None:
            return None
        configured_threshold = str(configured_threshold).strip()
        if configured_threshold == '' or configured_threshold.lower() == 'none':
            return None
        return float(configured_threshold)

    @staticmethod
    def _get_motion_emphasis_score_width(job: mpf.VideoJob) -> int:
        score_width = mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_SCORE_WIDTH', MOTION_SCORE_WIDTH)
        if score_width is None or str(score_width).strip() == '':
            return MOTION_SCORE_WIDTH
        try:
            score_width = int(float(score_width))
        except (TypeError, ValueError) as error:
            raise ValueError(f'MOTION_EMPHASIS_SCORE_WIDTH must be a non-negative integer: {score_width}') from error
        if score_width < 0:
            raise ValueError(f'MOTION_EMPHASIS_SCORE_WIDTH must be non-negative: {score_width}')
        return score_width

    @staticmethod
    def _get_motion_emphasis_score_fps(job: mpf.VideoJob, source_fps: float, target_fps: float) -> float:
        score_fps = mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_SCORE_FPS', 5.0)
        if score_fps is None or str(score_fps).strip() == '':
            return source_fps
        try:
            score_fps = float(score_fps)
        except (TypeError, ValueError) as error:
            raise ValueError(f'MOTION_EMPHASIS_SCORE_FPS must be a positive number: {score_fps}') from error
        if score_fps <= 0:
            raise ValueError(f'MOTION_EMPHASIS_SCORE_FPS must be positive: {score_fps}')
        return min(source_fps, max(target_fps, score_fps))

    @staticmethod
    def _get_motion_score_gray_frame(frame, score_width: int):
        if score_width > 0:
            height, width = frame.shape[:2]
            if width > score_width:
                score_height = max(1, round(height * (score_width / width)))
                frame = cv2.resize(
                    frame,
                    (score_width, score_height),
                    interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _get_motion_focus_max_ranges(job: mpf.VideoJob) -> int:
        max_ranges = mpf_util.get_property(job.job_properties, 'MOTION_FOCUS_MAX_RANGES', 40)
        try:
            max_ranges = int(float(max_ranges))
        except (TypeError, ValueError) as error:
            raise ValueError(f'MOTION_FOCUS_MAX_RANGES must be a non-negative integer: {max_ranges}') from error
        return max(0, max_ranges)

    def _build_motion_focus_ranges(
            self,
            job: mpf.VideoJob,
            high_motion_points: list[dict],
            source_fps: float,
            source_frame_count: int,
            target_fps: float,
            score_fps: float,
            neighbor_frames: int) -> list[dict]:
        if not high_motion_points:
            return []

        segment_duration_seconds = source_frame_count / source_fps
        score_padding_seconds = 0.5 / score_fps
        neighbor_padding_seconds = neighbor_frames / target_fps if target_fps > 0 else 0.0
        padding_seconds = max(score_padding_seconds, neighbor_padding_seconds)
        merge_gap_seconds = max(1.0 / score_fps, padding_seconds)

        ranges = []
        for point in sorted(high_motion_points, key=lambda item: item['timestamp_seconds']):
            timestamp_seconds = point['timestamp_seconds']
            start_seconds = max(0.0, timestamp_seconds - padding_seconds)
            end_seconds = min(segment_duration_seconds, timestamp_seconds + padding_seconds)
            if ranges and start_seconds <= ranges[-1]['end_seconds'] + merge_gap_seconds:
                current_range = ranges[-1]
                current_range['end_seconds'] = max(current_range['end_seconds'], end_seconds)
                current_range['peak_motion_score'] = max(
                    current_range['peak_motion_score'], point['motion_score'])
                current_range['motion_point_count'] += 1
            else:
                ranges.append({
                    'start_seconds': start_seconds,
                    'end_seconds': end_seconds,
                    'peak_motion_score': point['motion_score'],
                    'motion_point_count': 1,
                })

        max_ranges = self._get_motion_focus_max_ranges(job)
        if max_ranges == 0:
            return []
        if len(ranges) > max_ranges:
            ranges = sorted(ranges, key=lambda item: item['peak_motion_score'], reverse=True)[:max_ranges]
            ranges.sort(key=lambda item: item['start_seconds'])

        max_source_frame_index = max(0, source_frame_count - 1)
        sampled_frame_count = math.ceil(segment_duration_seconds * target_fps)
        max_preprocessed_frame_index = max(0, sampled_frame_count - 1)
        for motion_range in ranges:
            start_seconds = motion_range['start_seconds']
            end_seconds = motion_range['end_seconds']
            motion_range['source_start_frame'] = max(
                0, min(round(start_seconds * source_fps), max_source_frame_index))
            motion_range['source_end_frame'] = max(
                0, min(round(end_seconds * source_fps), max_source_frame_index))
            motion_range['preprocessed_start_frame'] = max(
                0, min(round(start_seconds * target_fps), max_preprocessed_frame_index))
            motion_range['preprocessed_end_frame'] = max(
                0, min(round(end_seconds * target_fps), max_preprocessed_frame_index))

        return ranges

    def _collect_preprocessed_frames(self, job: mpf.VideoJob) -> dict:
        source_fps = float(job.media_properties['FPS'])
        target_fps = self._get_preprocess_fps(job, source_fps)
        source_frame_count = max(0, job.stop_frame - job.start_frame + 1)
        configured_threshold = self._get_configured_motion_emphasis_threshold(job)
        score_width = self._get_motion_emphasis_score_width(job)
        score_fps = self._get_motion_emphasis_score_fps(job, source_fps, target_fps)

        if configured_threshold is None:
            threshold = self._get_motion_emphasis_threshold(
                job, self._collect_motion_scores(job, score_fps, score_width))
        else:
            threshold = configured_threshold

        preprocessed_frames = self._collect_sampled_and_high_motion_frames(
            job, target_fps, threshold, score_fps, score_width)

        sampled_frames = preprocessed_frames['sampled_frames']
        if not sampled_frames:
            raise ValueError('No frames were read from the video segment.')

        neighbor_frames = max(0, int(mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_NEIGHBOR_FRAMES', 0)))
        motion_focus_ranges = self._build_motion_focus_ranges(
            job,
            preprocessed_frames.get('high_motion_points', []),
            source_fps,
            source_frame_count,
            target_fps,
            score_fps,
            neighbor_frames)

        return {
            'sampled_frames': sampled_frames,
            'motion_focus_ranges': motion_focus_ranges,
            'high_motion_point_count': len(preprocessed_frames.get('high_motion_points', [])),
            'neighbor_frames': neighbor_frames,
            'threshold': threshold,
            'source_fps': source_fps,
            'target_fps': target_fps,
            'source_frame_count': source_frame_count,
            'motion_score_fps': score_fps,
            'motion_score_width': score_width,
            'regular_sampled_frame_count': preprocessed_frames['regular_sampled_frame_count'],
        }

    def _collect_motion_scores(self, job: mpf.VideoJob, score_fps: float, score_width: int) -> list[float]:
        motion_scores = []
        previous_gray = None
        for _, _, frame in self._iter_sampled_video_frames(job, score_fps, motion_profiling=True):
            gray_frame = self._get_motion_score_gray_frame(frame, score_width)
            if previous_gray is None:
                motion_scores.append(0.0)
            else:
                motion_scores.append(float(cv2.absdiff(gray_frame, previous_gray).mean()))
            previous_gray = gray_frame
        return motion_scores

    def _collect_sampled_and_high_motion_frames(
            self, job: mpf.VideoJob, target_fps: float, threshold: float, score_fps: float, score_width: int) -> dict:
        previous_gray = None
        sampled_frames = []
        high_motion_points = []
        regular_sampled_frame_count = 0

        for source_frame_index, timestamp_seconds, frame, is_regular_sample, is_score_sample in (
                self._iter_motion_sampling_video_frames(job, target_fps, score_fps)):
            motion_score = 0.0
            if is_score_sample:
                gray_frame = self._get_motion_score_gray_frame(frame, score_width)
                if previous_gray is not None:
                    motion_score = float(cv2.absdiff(gray_frame, previous_gray).mean())
                previous_gray = gray_frame

            if is_score_sample and motion_score >= threshold and motion_score > 0:
                high_motion_points.append({
                    'timestamp_seconds': timestamp_seconds,
                    'source_frame_index': source_frame_index,
                    'motion_score': motion_score,
                })

            if is_regular_sample:
                sampled_frames.append((timestamp_seconds, frame))
                regular_sampled_frame_count += 1

        return {
            'sampled_frames': sampled_frames,
            'high_motion_points': high_motion_points,
            'regular_sampled_frame_count': regular_sampled_frame_count,
        }

    @staticmethod
    def _iter_preprocessed_output_frames(preprocessed_frames: dict):
        for sample_index, (timestamp_seconds, frame) in enumerate(preprocessed_frames['sampled_frames']):
            yield sample_index, timestamp_seconds, frame

    def _write_preprocessed_video_with_ffmpeg(self, preprocessed_frames: dict, output_path: str) -> list[float] | None:
        ffmpeg_path = self._get_ffmpeg_path()
        if not ffmpeg_path:
            return None

        frame_iter = self._iter_preprocessed_output_frames(preprocessed_frames)
        try:
            _, first_timestamp_seconds, first_frame = next(frame_iter)
        except StopIteration:
            return []

        height, width = first_frame.shape[:2]
        command = [
            ffmpeg_path,
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
            '-f',
            'rawvideo',
            '-pix_fmt',
            'bgr24',
            '-s',
            f'{width}x{height}',
            '-r',
            f"{preprocessed_frames['target_fps']:.6f}",
            '-i',
            'pipe:0',
            '-an',
            '-c:v',
            'libx264',
            '-preset',
            'veryfast',
            '-crf',
            '23',
            '-pix_fmt',
            'yuv420p',
            '-movflags',
            '+faststart',
            output_path,
        ]

        output_timestamps = []
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        try:
            for _, timestamp_seconds, frame in itertools.chain(
                    [(0, first_timestamp_seconds, first_frame)], frame_iter):
                process.stdin.write(frame.tobytes())
                output_timestamps.append(timestamp_seconds)
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate()
        except Exception:
            process.kill()
            process.communicate()
            raise

        if process.returncode != 0:
            stderr_text = stderr.decode('utf-8', errors='replace').strip()
            raise ValueError(f'Unable to encode preprocessed video with ffmpeg: {stderr_text}')

        return output_timestamps

    def _write_preprocessed_video_with_opencv(self, preprocessed_frames: dict, output_path: str) -> list[float]:
        output_timestamps = []
        writer = None
        try:
            for _, timestamp_seconds, frame in self._iter_preprocessed_output_frames(preprocessed_frames):
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        preprocessed_frames['target_fps'],
                        (width, height))
                    if not writer.isOpened():
                        raise ValueError(f'Unable to open video writer for {output_path}.')

                writer.write(frame)
                output_timestamps.append(timestamp_seconds)
        finally:
            if writer is not None:
                writer.release()

        return output_timestamps

    @staticmethod
    def _get_sampled_frame_timestamps(preprocessed_frames: dict) -> list[float]:
        return [
            timestamp_seconds
            for timestamp_seconds, _ in preprocessed_frames['sampled_frames']
        ]

    @staticmethod
    def _is_full_video_job(job: mpf.VideoJob, source_frame_count: int) -> bool:
        try:
            media_frame_count = int(float((job.media_properties or {}).get('FRAME_COUNT', 0)))
        except (TypeError, ValueError):
            media_frame_count = 0
        return job.start_frame == 0 and source_frame_count > 0 and (
            media_frame_count <= 0 or job.stop_frame >= media_frame_count - 1)

    @staticmethod
    def _source_is_mp4(job: mpf.VideoJob) -> bool:
        mime_type = str((job.media_properties or {}).get('MIME_TYPE', '')).lower()
        return mime_type == 'video/mp4' or job.data_uri.lower().endswith('.mp4')

    @staticmethod
    def _target_matches_source_fps(preprocessed_frames: dict) -> bool:
        source_fps = float(preprocessed_frames['source_fps'])
        target_fps = float(preprocessed_frames['target_fps'])
        return abs(source_fps - target_fps) < 1e-6

    def _remux_preprocessed_video_with_ffmpeg(
            self, job: mpf.VideoJob, preprocessed_frames: dict, output_path: str) -> bool:
        ffmpeg_path = self._get_ffmpeg_path()
        if not ffmpeg_path:
            return False

        start_seconds, duration_seconds = self._get_source_segment_timing(
            job, preprocessed_frames['source_fps'], preprocessed_frames['source_frame_count'])
        command = [
            ffmpeg_path,
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
        ]
        if start_seconds > 0:
            command.extend(['-ss', f'{start_seconds:.6f}'])
        if duration_seconds > 0:
            command.extend(['-t', f'{duration_seconds:.6f}'])
        command.extend([
            '-i',
            job.data_uri,
            '-map',
            '0:v:0',
            '-an',
            '-c:v',
            'copy',
            '-movflags',
            '+faststart',
            '-f',
            'mp4',
            output_path,
        ])

        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return True
        except (OSError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, 'stderr', '') or ''
            logger.warning('Unable to remux source video without reencoding; falling back to encode: %s', stderr or error)
            return False

    def _get_preprocessed_video_passthrough(
            self, job: mpf.VideoJob, preprocessed_frames: dict) -> tuple[str, list[float], bool, str] | None:
        if not self._target_matches_source_fps(preprocessed_frames):
            return None

        output_timestamps = self._get_sampled_frame_timestamps(preprocessed_frames)
        if self._is_full_video_job(job, preprocessed_frames['source_frame_count']) and self._source_is_mp4(job):
            logger.info('Using source MP4 directly because process fps matches source fps: %s', job.data_uri)
            return job.data_uri, output_timestamps, False, 'source_mp4_direct'

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        if self._remux_preprocessed_video_with_ffmpeg(job, preprocessed_frames, temp_file.name):
            logger.info('Remuxed source video without reencoding because process fps matches source fps: %s', temp_file.name)
            return temp_file.name, output_timestamps, True, 'source_remux_no_reencode'

        try:
            os.remove(temp_file.name)
        except OSError:
            pass
        return None

    def _preprocess_video_for_model(self, job: mpf.VideoJob) -> dict:
        preprocessed_frames = self._collect_preprocessed_frames(job)
        passthrough_video = self._get_preprocessed_video_passthrough(job, preprocessed_frames)
        delete_video = True
        preprocess_mode = 'sampled_encode'
        if passthrough_video is not None:
            video_path, output_timestamps, delete_video, preprocess_mode = passthrough_video
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_file.close()
            video_path = temp_file.name

            try:
                output_timestamps = self._write_preprocessed_video_with_ffmpeg(
                    preprocessed_frames, video_path)
                if output_timestamps is None:
                    logger.info('ffmpeg was not found; writing preprocessed video with OpenCV mp4v fallback.')
                    output_timestamps = self._write_preprocessed_video_with_opencv(
                        preprocessed_frames, video_path)
            except Exception:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
                raise

        if not output_timestamps:
            if delete_video:
                try:
                    os.remove(video_path)
                except OSError:
                    pass
            raise ValueError('No sampled frames were written to the preprocessed video.')

        self._last_preprocessed_frame_timestamps = output_timestamps
        self._last_preprocessed_fps = preprocessed_frames['target_fps']

        logger.info(
            'Prepared model video with %d source frames, %d regular sampled frames, '
            '%d output frames, source fps %.3f, process fps %.3f, score fps %.3f, score width %d, '
            'threshold %.3f, high-motion points %d, motion focus ranges %d, mode %s, path %s.',
            preprocessed_frames['source_frame_count'], preprocessed_frames['regular_sampled_frame_count'],
            len(output_timestamps), preprocessed_frames['source_fps'], preprocessed_frames['target_fps'],
            preprocessed_frames['motion_score_fps'], preprocessed_frames['motion_score_width'],
            preprocessed_frames['threshold'], preprocessed_frames['high_motion_point_count'],
            len(preprocessed_frames['motion_focus_ranges']), preprocess_mode, video_path)
        keep_temp_media = self._is_enabled(mpf_util.get_property(job.job_properties, 'KEEP_TEMP_MEDIA', '0'))
        original_duration_seconds = preprocessed_frames['source_frame_count'] / preprocessed_frames['source_fps']
        if preprocess_mode in {'source_mp4_direct', 'source_remux_no_reencode'}:
            duration_seconds = original_duration_seconds
        else:
            duration_seconds = len(output_timestamps) / preprocessed_frames['target_fps']
        return {
            'path': video_path,
            'frame_timestamps': output_timestamps,
            'fps': preprocessed_frames['target_fps'],
            'duration_seconds': duration_seconds,
            'original_duration_seconds': original_duration_seconds,
            'original_fps': preprocessed_frames['source_fps'],
            'source_frame_count': preprocessed_frames['source_frame_count'],
            'motion_focus_ranges': preprocessed_frames['motion_focus_ranges'],
            'motion_focus_range_count': len(preprocessed_frames['motion_focus_ranges']),
            'high_motion_point_count': preprocessed_frames['high_motion_point_count'],
            'motion_threshold': preprocessed_frames['threshold'],
            'motion_score_fps': preprocessed_frames['motion_score_fps'],
            'motion_score_width': preprocessed_frames['motion_score_width'],
            'preprocess_mode': preprocess_mode,
            'delete_video': delete_video,
            'keep_temp_media': keep_temp_media
        }

    def _cleanup_preprocessed_video(self, preprocessed_video):
        if not preprocessed_video:
            return

        path = preprocessed_video.get('path')
        audio_path = preprocessed_video.get('audio_path')
        if preprocessed_video.get('keep_temp_media'):
            if path:
                logger.info('Keeping temporary preprocessed video: %s', path)
            if audio_path:
                logger.info('Keeping temporary preprocessed audio: %s', audio_path)
            return
        if path and preprocessed_video.get('delete_video', True):
            try:
                os.remove(path)
            except OSError:
                logger.warning('Unable to remove temporary preprocessed video: %s', path)
        if audio_path:
            try:
                os.remove(audio_path)
            except OSError:
                logger.warning('Unable to remove temporary preprocessed audio: %s', audio_path)

    @staticmethod
    def _get_source_segment_timing(
            job: mpf.VideoJob, source_fps: float, source_frame_count: int) -> tuple[float, float]:
        start_seconds = max(0.0, job.start_frame / source_fps)
        duration_seconds = source_frame_count / source_fps
        return start_seconds, duration_seconds

    def _build_ffmpeg_audio_input_args(
            self, job: mpf.VideoJob, source_fps: float, source_frame_count: int) -> list[str]:
        start_seconds, duration_seconds = self._get_source_segment_timing(job, source_fps, source_frame_count)
        audio_input_args = []
        if start_seconds > 0:
            audio_input_args.extend(['-ss', f'{start_seconds:.6f}'])
        if duration_seconds > 0:
            audio_input_args.extend(['-t', f'{duration_seconds:.6f}'])
        audio_input_args.extend(['-i', job.data_uri])
        return audio_input_args

    @staticmethod
    def _get_ffmpeg_path() -> str | None:
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path:
            return ffmpeg_path
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
        return imageio_ffmpeg.get_ffmpeg_exe()

    def _extract_source_audio_for_model(
            self, job: mpf.VideoJob, source_fps: float, source_frame_count: int) -> str | None:
        ffmpeg_path = self._get_ffmpeg_path()
        if not ffmpeg_path:
            logger.info('ffmpeg was not found; source audio will not be sent separately.')
            return None

        audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio_file.close()
        command = [
            ffmpeg_path,
            '-hide_banner',
            '-loglevel',
            'error',
            '-y',
            *self._build_ffmpeg_audio_input_args(job, source_fps, source_frame_count),
            '-vn',
            '-map',
            '0:a:0',
            '-acodec',
            'pcm_f32le',
            '-ac',
            '1',
            '-ar',
            '16000',
            '-f',
            'wav',
            audio_file.name,
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except (OSError, subprocess.CalledProcessError) as error:
            try:
                os.remove(audio_file.name)
            except OSError:
                pass
            stderr = getattr(error, 'stderr', '') or ''
            if 'matches no streams' in stderr or 'Stream map' in stderr:
                logger.info('Source video does not contain an audio stream.')
            else:
                logger.warning('Unable to extract source audio for model request: %s', error)
            return None

        logger.info('Extracted source audio for model request: %s', audio_file.name)
        return audio_file.name

    def _get_motion_emphasis_threshold(self, job: mpf.VideoJob, motion_scores) -> float:
        configured_threshold = mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_THRESHOLD', '5.0')
        if configured_threshold != '':
            return float(configured_threshold)

        percentile = float(mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_PERCENTILE', 90.0))
        sorted_scores = sorted(score for score in motion_scores if score > 0)
        if not sorted_scores:
            return float('inf')

        percentile = min(100.0, max(0.0, percentile))
        index = round((percentile / 100.0) * (len(sorted_scores) - 1))
        return sorted_scores[index]

    @staticmethod
    def _build_preprocessed_timing_prompt(
            preprocessed_video: dict, include_timestamp_instructions: bool = True) -> str:
        motion_focus_ranges = preprocessed_video.get('motion_focus_ranges') or []
        preprocess_mode = preprocessed_video.get('preprocess_mode', 'sampled_encode')
        uses_source_frame_rate = preprocess_mode in {'source_mp4_direct', 'source_remux_no_reencode'}
        if uses_source_frame_rate:
            prompt = (
                "The supplied video is the original segment at the source frame rate. It was not "
                "sampled down to 1 FPS or reencoded with motion-profile frames. It may have been "
                "remuxed into MP4 only for model compatibility.\n\n"
                "Video metadata:\n"
                f"- Original segment duration: {preprocessed_video['original_duration_seconds']:.2f} seconds\n"
                f"- Source FPS: {preprocessed_video['original_fps']:.3f}\n"
                f"- Supplied video FPS: {preprocessed_video['fps']:.3f}\n"
                f"- Supplied playback duration: {preprocessed_video['duration_seconds']:.2f} seconds\n"
                "- The video you see is the full source-frame-rate segment, not a 1 FPS sampled video.\n"
            )
        else:
            prompt = (
                "The supplied video has been sampled for model input while preserving playback timing "
                "with the original audio segment as closely as the sample rate allows. No frames were "
                "changed with extra motion-profile frames.\n\n"
                "Preprocessing metadata:\n"
                f"- Original segment duration: {preprocessed_video['original_duration_seconds']:.2f} seconds\n"
                f"- Original FPS: {preprocessed_video['original_fps']:.3f}\n"
                f"- Preprocessed video FPS: {preprocessed_video['fps']:.3f}\n"
                f"- Preprocessed playback duration: {preprocessed_video['duration_seconds']:.2f} seconds\n"
                "- The video you see is the sampled/preprocessed version.\n"
            )

        prompt += (
            "- First analyze the full video and audio together. Then reconsider any motion focus "
            "ranges listed below before writing the final summary.\n"
        )
        if motion_focus_ranges:
            prompt += "- Motion focus ranges use original segment time and frame indexes:\n"
            for range_index, motion_range in enumerate(motion_focus_ranges, start=1):
                frame_label = 'supplied video frames' if uses_source_frame_rate else 'sampled video frames'
                prompt += (
                    f"  {range_index}. original {motion_range['start_seconds']:.2f}-"
                    f"{motion_range['end_seconds']:.2f}s; source frames "
                    f"{motion_range['source_start_frame']}-{motion_range['source_end_frame']}; "
                    f"{frame_label} {motion_range['preprocessed_start_frame']}-"
                    f"{motion_range['preprocessed_end_frame']}; peak motion score "
                    f"{motion_range['peak_motion_score']:.3f}\n"
                )
        else:
            prompt += "- Motion profiler did not identify extra focus ranges above the configured threshold.\n"

        if include_timestamp_instructions:
            if uses_source_frame_rate:
                return (
                    f"{prompt}"
                    "- Return timestamps relative to the supplied video's playback time, which matches "
                    "the original segment time."
                )
            return (
                f"{prompt}"
                "- Return timestamps relative to the supplied preprocessed video's playback time.\n"
                "- The component will convert your preprocessed-video timestamps back to "
                "original-video timestamps."
            )
        return (
            f"{prompt}"
            "- Do not include timestamps or frame indexes in the response."
        )

    @staticmethod
    def _encode_file_as_data_url(path: str, mime_type: str) -> str:
        with open(path, 'rb') as data_file:
            encoded_data = base64.b64encode(data_file.read()).decode('utf-8')
        return f'data:{mime_type};base64,{encoded_data}'

    def _encode_video_as_data_url(self, video_path: str) -> str:
        return self._encode_file_as_data_url(video_path, 'video/mp4')

    def _encode_audio_as_data_url(self, audio_path: str) -> str:
        return self._encode_file_as_data_url(audio_path, 'audio/wav')

    @staticmethod
    def _model_supports_audio_input(model_name: str) -> bool:
        normalized_model_name = (model_name or '').lower()
        return any(
            audio_model in normalized_model_name
            for audio_model in ('gemma-4-e2b', 'gemma-4-e4b', 'gemma-4-12b')
        )

    @staticmethod
    def _is_enabled(value) -> bool:
        return str(value).strip().lower() not in ('0', 'false', 'no', 'off')

    @staticmethod
    def _get_openai_api_key(application_credentials: str) -> str:
        if application_credentials:
            env_value = os.environ.get(application_credentials)
            if env_value:
                return env_value
            if os.path.exists(application_credentials):
                with open(application_credentials, 'r') as api_key_file:
                    api_key = api_key_file.read().strip()
                    if api_key:
                        return api_key
        return "Empty"

    def _openai_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float, config=None) -> str:
        if not model_name:
            raise mpf.DetectionException(
                "MODEL_NAME must be provided for OpenAI API requests.",
                mpf.DetectionError.INVALID_PROPERTY
            )
        config = config or JobConfig(
            job.job_properties,
            job.media_properties,
            model=getattr(self, "model", None) is not None)
        api_key = self._get_openai_api_key(config.application_credentials)

        preprocessed_video = None
        
        try:
            preprocessed_video = self._preprocess_video_for_model(job)
            logger.info(
                'Sending one model video to OpenAI/vLLM: %d frame timestamps, fps %.3f, duration %.2fs, mode %s.',
                len(preprocessed_video.get('frame_timestamps', [])),
                preprocessed_video.get('fps', fps),
                preprocessed_video['duration_seconds'],
                preprocessed_video.get('preprocess_mode', 'sampled_encode'))

            timeout_seconds = float(mpf_util.get_property(
                job.job_properties, "OPENAI_REQUEST_TIMEOUT_SECONDS", "600"))
            max_retries = int(mpf_util.get_property(
                job.job_properties, "OPENAI_MAX_RETRIES", "2" if not config.base_url else "0"))
            
            client = OpenAI(
                api_key=api_key,
                base_url=config.base_url or None,
                timeout=timeout_seconds,
                max_retries=max_retries)
            
            include_timestamp_instructions = int(mpf_util.get_property(
                job.job_properties, "ENABLE_TIMELINE", "1")) == 1
            preprocessing_prompt = self._build_preprocessed_timing_prompt(
                preprocessed_video,
                include_timestamp_instructions=include_timestamp_instructions)
            
            request_content = [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": self._encode_video_as_data_url(preprocessed_video['path'])
                    }
                },
                {
                    "type": "text",
                    "text": f"{prompt}\n\n{preprocessing_prompt}"
                }
            ]
            
            enable_audio = self._is_enabled(mpf_util.get_property(job.job_properties, "ENABLE_AUDIO", "1"))
            if enable_audio and self._model_supports_audio_input(model_name):
                audio_path = self._extract_source_audio_for_model(
                    job,
                    preprocessed_video['original_fps'],
                    preprocessed_video['source_frame_count'])
                
                if audio_path:
                    preprocessed_video['audio_path'] = audio_path
                    request_content[1]["text"] += (
                        "\n\nAudio note: This request includes a separate audio clip extracted from "
                        "the original video segment. Use that audio when describing conversations, "
                        "spoken language, translated speech, and other sound-based context. Do not "
                        "state that no audio was provided."
                    )
                    request_content.append({
                        "type": "audio_url",
                        "audio_url": {
                            "url": self._encode_audio_as_data_url(audio_path)
                        }
                    })
                    
                    logger.info('Sending original segment audio with OpenAI/vLLM request.')
                else:
                    logger.info('No source audio was available to send with the OpenAI/vLLM request.')
                    
            elif enable_audio:
                logger.info('Model %s does not support audio input; sending video frames only.', model_name)
                
            feed_forward_speech_tracks = self._get_feed_forward_speech_summary_tracks(job)
            if feed_forward_speech_tracks:
                speaker_timeline = self._build_feed_forward_speaker_timeline_prompt(feed_forward_speech_tracks)
                if speaker_timeline:
                    request_content[1]["text"] += (
                        "\n\nThis request includes a speaker timeline. "
                        "Use it only to keep speaker identities consistent while analyzing the provided video and audio. "
                        "It intentionally does not include transcript or translation text; use the audio itself for spoken content. "
                        "Do not use internal speaker IDs in the final summary; use the generated speaker labels or natural descriptions."
                        f"\n\nSpeaker timeline:\n{speaker_timeline}"
                    )
                    logger.info('Feed-forward speech speaker timeline found, sending to Gemma')
                else:
                    logger.info('Feed-forward speech summarization TEXT tracks found, but no VOICED_SEGMENTS were available.')

            request_args = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": request_content
                    }
                ],
            }
            default_json_response_format = "true" if not config.base_url else "false"
            use_json_response_format = mpf_util.get_property(
                job.job_properties,
                "OPENAI_RESPONSE_FORMAT_JSON_OBJECT",
                default_json_response_format)
            if str(use_json_response_format).lower() == "true":
                request_args["response_format"] = {"type": "json_object"}

            max_tokens = str(mpf_util.get_property(
                job.job_properties, "OPENAI_MAX_TOKENS", "")).strip()
            if max_tokens:
                request_args["max_tokens"] = int(max_tokens)

            temperature = str(mpf_util.get_property(
                job.job_properties, "OPENAI_TEMPERATURE", "")).strip()
            if temperature:
                request_args["temperature"] = float(temperature)

            logger.info(
                "OpenAI/vLLM request options: response_format_json=%s, max_tokens=%s, "
                "temperature=%s, timeout=%.1fs, max_retries=%d",
                "enabled" if "response_format" in request_args else "disabled",
                request_args.get("max_tokens", "default"),
                request_args.get("temperature", "default"),
                timeout_seconds,
                max_retries)

            response = client.chat.completions.create(**request_args)

            return response.choices[0].message.content
        finally:
            self._cleanup_preprocessed_video(preprocessed_video)

    def _google_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float, config=None) -> str:
        config = config or JobConfig(
            job.job_properties,
            job.media_properties,
            model=getattr(self, "model", None) is not None)
        preprocessed_video = None
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.application_credentials
            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                raise mpf.DetectionException(
                    f"Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is not set.",
                    mpf.DetectionError.INVALID_PROPERTY
                )
            preprocessed_video = self._preprocess_video_for_model(job)

            # Video segment storage information
            FILE_NAME = os.path.basename(preprocessed_video['path'])
            STORAGE_PATH = config.label_user + "/" + FILE_NAME

            # Uploads file to GCP bucket
            client = storage.Client(project=config.project_id)
            bucket = client.bucket(config.bucket_name)
            blob = bucket.blob(STORAGE_PATH)
            blob.upload_from_filename(preprocessed_video['path'])

            file_uri = f"gs://{config.bucket_name}/{STORAGE_PATH}"

            # Generate Gemini response
            genai_client = genai.Client(
                project=config.project_id,
                location="global",
                enterprise=True
            )

            content_config = None
            if config.label_user and config.label_prefix and config.label_purpose:
                content_config = types.GenerateContentConfig(
                    labels={
                        config.label_prefix + "user": config.label_user,
                        config.label_prefix + "purpose": config.label_purpose,
                        config.label_prefix + "modality": "video"
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
                                start_offset="0.00s",
                                end_offset=f"{preprocessed_video['duration_seconds']:.2f}s",
                                fps=preprocessed_video.get('fps', fps)
                            )
                        ),
                        Part(
                            text=(
                                f"{prompt}\n\n"
                                f"{self._build_preprocessed_timing_prompt(
                                    preprocessed_video,
                                    include_timestamp_instructions=int(mpf_util.get_property(
                                        job.job_properties, 'ENABLE_TIMELINE', '1')) == 1)}"
                            )
                        )
                    ],
                ),
                config=content_config
            )

            return response.text
        finally:
            self._cleanup_preprocessed_video(preprocessed_video)

    def _get_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float, config=None):
        config = config or JobConfig(
            job.job_properties,
            job.media_properties,
            model=getattr(self, "model", None) is not None)
        try:
            if config.api == "OpenAI":
                return self._openai_response(job, prompt, model_name, fps, config)
            elif config.api == "Google":
                return self._google_response(job, prompt, model_name, fps, config)
            else:
                raise mpf.DetectionException(
                    f"Unsupported API specified: {config.api}",
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
                f"{config.api} API call failed: {e}",
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
    def __init__(
            self,
            job_properties: Mapping[str, str],
            media_properties=None,
            model=False):
        self.api = self._get_prop(job_properties, "API", "OpenAI", ["OpenAI", "Google"])
        self.base_url = self._get_prop(job_properties, "OPENAI_BASE_URL", "")
        self.base_url = str(self.base_url).strip() or None

        self.generation_prompt_path = self._get_prop(job_properties, "GENERATION_PROMPT_PATH", "")
        self.enable_timeline = int(self._get_prop(job_properties, "ENABLE_TIMELINE", "1"))

        if self.generation_prompt_path == "" and self.enable_timeline == 1:
            self.generation_prompt_path= os.path.join(os.path.dirname(__file__), 'data', 'default_prompt.txt')
        elif self.generation_prompt_path == "" and self.enable_timeline == 0:
            self.generation_prompt_path= os.path.join(os.path.dirname(__file__), 'data', 'default_prompt_no_tl.txt')

        self.generation_prompt_path = self._resolve_existing_path(self.generation_prompt_path)
        if not self.generation_prompt_path:
            raise mpf.DetectionException(
                "Invalid path provided for prompt file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )

        self.application_credentials = self._get_prop(job_properties, "APPLICATION_CREDENTIALS", "")
        if self.api == "Google":
            self.application_credentials = self._resolve_existing_path(self.application_credentials)
            if not self.application_credentials and not model:
                raise mpf.DetectionException(
                    "Invalid path provided for GCP credential file: ",
                    mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
                )
        if (
                self.api == "OpenAI"
                and self.application_credentials
                and not os.path.exists(self.application_credentials)
                and not os.environ.get(self.application_credentials)):
            logger.warning(
                "APPLICATION_CREDENTIALS did not match an environment variable or file path; "
                "using the OpenAI client dummy key fallback.")

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
    def _resolve_existing_path(path: str) -> str:
        if not path:
            return ""
        if os.path.exists(path):
            return path
        if not os.path.isabs(path):
            component_relative_path = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(component_relative_path):
                return component_relative_path
        return ""

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
