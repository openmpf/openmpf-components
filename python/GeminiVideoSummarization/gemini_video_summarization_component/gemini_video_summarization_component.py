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

import torch
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

logger = logging.getLogger('GeminiVideoSummarizationComponent')

_MISSING_CHAT_TEMPLATE_ERRORS = (
    "chat_template is not set",
    "does not have a chat template",
)


class GeminiVideoSummarizationComponent:

    def __init__(self, model: Gemma4ForConditionalGeneration=None, processor: Gemma4Processor=None, API="OpenAI", base_url=None, device=None):
        self.model = model
        self.processor = processor
        self.device = device
        self.base_url = base_url
        
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
            
        config = JobConfig(job.job_properties, job.media_properties, model=self.model is not None)
        
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
            response = self._extract_json_object(response)
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
            indices = torch.arange(
                0,
                available_frame_count,
                available_frame_count / num_frames
            ).int()
            if len(indices) <= num_frames and int(indices.max()) < available_frame_count:
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
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": preprocessed_video_path},
                        {
                            "type": "text",
                            "text": (
                                f"{prompt}\n\n"
                                "The supplied video has been preprocessed to make brief, high-motion events easier to see. "
                                "Frames with large motion may be duplicated, so repeated frames do not mean the event lasted longer. "
                                f"Describe timestamps relative to the original {SEGMENT_STOP - SEGMENT_START:.2f}-second segment, "
                                "not the preprocessed video's playback duration."
                            )
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

    def _preprocess_video_for_model(self, job: mpf.VideoJob) -> dict:
        reader = mpf_util.VideoCapture(job)
        frames = []
        gray_frames = []
        frame_timestamps = []
        try:
            max_frames = max(0, job.stop_frame - job.start_frame + 1)
            frames_read = 0
            while frames_read < max_frames:
                timestamp_seconds = reader.current_time_in_millis / 1000.0
                success, frame = reader.read()
                if not success:
                    break
                frames.append(frame)
                gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                frame_timestamps.append(timestamp_seconds)
                frames_read += 1
        finally:
            reader.release()

        if not frames:
            raise ValueError('No frames were read from the video segment.')

        motion_scores = [0.0]
        for current_gray, previous_gray in zip(gray_frames[1:], gray_frames[:-1]):
            motion_scores.append(float(cv2.absdiff(current_gray, previous_gray).mean()))

        threshold = self._get_motion_emphasis_threshold(job, motion_scores)
        duplicate_count = max(0, int(mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_DUPLICATES', 8)))
        neighbor_frames = max(0, int(mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_NEIGHBOR_FRAMES', 1)))

        high_motion_indexes = {
            index
            for index, score in enumerate(motion_scores)
            if score >= threshold and score > 0
        }
        emphasized_indexes = set()
        for index in high_motion_indexes:
            start_index = max(0, index - neighbor_frames)
            stop_index = min(len(frames) - 1, index + neighbor_frames)
            emphasized_indexes.update(range(start_index, stop_index + 1))

        output_frames = []
        output_timestamps = []
        for index, frame in enumerate(frames):
            output_frames.append(frame)
            output_timestamps.append(frame_timestamps[index])
            if index in emphasized_indexes:
                for _ in range(duplicate_count):
                    output_frames.append(frame.copy())
                    output_timestamps.append(frame_timestamps[index])

        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()

        height, width = frames[0].shape[:2]
        fps = float(job.media_properties['FPS'])
        writer = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        try:
            if not writer.isOpened():
                raise ValueError(f'Unable to open video writer for {temp_file.name}.')
            for frame in output_frames:
                writer.write(frame)
        finally:
            writer.release()

        logger.info(
            'Created motion-emphasized video with %d original frames, %d output frames, threshold %.3f, emphasized frames %d.',
            len(frames), len(output_frames), threshold, len(emphasized_indexes))
        return {
            'path': temp_file.name,
            'frame_timestamps': output_timestamps,
            'duration_seconds': len(output_frames) / fps,
            'original_duration_seconds': (job.stop_frame - job.start_frame + 1) / fps
        }

    def _cleanup_preprocessed_video(self, preprocessed_video):
        if not preprocessed_video:
            return

        path = preprocessed_video.get('path')
        if path:
            try:
                os.remove(path)
            except OSError:
                logger.warning('Unable to remove temporary preprocessed video: %s', path)

    def _get_motion_emphasis_threshold(self, job: mpf.VideoJob, motion_scores) -> float:
        configured_threshold = mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_THRESHOLD', '')
        if configured_threshold != '':
            return float(configured_threshold)

        percentile = float(mpf_util.get_property(job.job_properties, 'MOTION_EMPHASIS_PERCENTILE', 90.0))
        sorted_scores = sorted(score for score in motion_scores if score > 0)
        if not sorted_scores:
            return float('inf')

        percentile = min(100.0, max(0.0, percentile))
        index = round((percentile / 100.0) * (len(sorted_scores) - 1))
        return sorted_scores[index]
            
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

        preprocessed_video = None
        video = None
        try:
            api_key = os.environ.get(self.application_credentials)
            if not api_key:
                raise mpf.DetectionException(
                    f"Environment variable '{self.application_credentials}' is not set.",
                    mpf.DetectionError.INVALID_PROPERTY
                )
            preprocessed_video = self._preprocess_video_for_model(job)
            frame_timestamps = preprocessed_video['frame_timestamps']
            
            # Generate OpenAI response
            client = OpenAI(api_key=api_key, base_url=self.base_url)
            video = cv2.VideoCapture(preprocessed_video['path'])
            if not video.isOpened():
                raise ValueError(f"Unable to open preprocessed video: {preprocessed_video['path']}")
            
            content = [
                {
                    "type": "text",
                    "text": (
                        f"{prompt}\n\n"
                        "The following frames come from a preprocessed version of the video segment. "
                        "Frames with large motion may be duplicated to make brief events easier to see. "
                        "Each frame is labeled with its timestamp in the original video segment; use those labels for event times."
                    )
                }
            ]

            frame_index = 0
            while True:
                success, frame = video.read()
                if not success:
                    break

                timestamp_seconds = frame_timestamps[min(frame_index, len(frame_timestamps) - 1)]
                encoded_frame = self._encode_frame_as_jpeg(frame)
                content.append({
                    "type": "text",
                    "text": f"Frame {frame_index}, original timestamp {timestamp_seconds:.2f}s"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_frame}"
                    }
                })
                frame_index += 1

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
            
            return response.choices[0].message.content
        finally:
            if video is not None:
                video.release()
            self._cleanup_preprocessed_video(preprocessed_video)
        
    def _google_response(self, job: mpf.VideoJob, prompt: str, model_name: str, fps: float) -> str:
        preprocessed_video = None
        try:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.application_credentials
            if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
                raise mpf.DetectionException(
                    f"Environment variable 'GOOGLE_APPLICATION_CREDENTIALS' is not set.",
                    mpf.DetectionError.INVALID_PROPERTY
                )
            preprocessed_video = self._preprocess_video_for_model(job)
            
            # Video segment storage information
            FILE_NAME = os.path.basename(preprocessed_video['path'])
            STORAGE_PATH = self.label_user + "/" + FILE_NAME
            
            # Uploads file to GCP bucket
            client = storage.Client(project=self.project_id)
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(STORAGE_PATH)
            blob.upload_from_filename(preprocessed_video['path'])

            file_uri = f"gs://{self.bucket_name}/{STORAGE_PATH}"

            # Generate Gemini response
            genai_client = genai.Client(
                project=self.project_id,
                location="global",
                enterprise=True
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
                                start_offset="0.00s",
                                end_offset=f"{preprocessed_video['duration_seconds']:.2f}s",
                                fps=fps
                            )
                        ),
                        Part(
                            text=(
                                f"{prompt}\n\n"
                                "The supplied video has been preprocessed to make brief, high-motion events easier to see. "
                                "Frames with large motion may be duplicated, so repeated frames do not mean the event lasted longer. "
                                f"Describe timestamps relative to the original {preprocessed_video['original_duration_seconds']:.2f}-second segment, "
                                "not the preprocessed video's playback duration."
                            )
                        )
                    ],
                ),
                config=content_config
            )

            return response.text
        finally:
            self._cleanup_preprocessed_video(preprocessed_video)
    
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