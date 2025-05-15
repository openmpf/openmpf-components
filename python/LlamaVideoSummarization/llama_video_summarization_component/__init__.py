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

import importlib.resources
import json
import logging
import os
import pickle
import socket
import subprocess

from jsonschema import validate, ValidationError
from typing import Iterable, Mapping, Tuple

import mpf_component_api as mpf
import mpf_component_util as mpf_util

log = logging.getLogger('LlamaVideoSummarizationComponent')

class LlamaVideoSummarizationComponent:

    def __init__(self):
        self.child_process = ChildProcess(['/llama/venv/bin/python3', '/llama/summarize_video.py', str(log.getEffectiveLevel())])


    def get_detections_from_video(self, job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        try:
            log.info('Received video job.')

            if job.feed_forward_track:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    'Component cannot process feed forward jobs.')
            
            segment_start_time = job.start_frame / float(job.media_properties['FPS'])
            segment_stop_time = job.stop_frame / float(job.media_properties['FPS'])

            job_config = _parse_properties(job.job_properties, segment_start_time)
            job_config['video_path'] = job.data_uri
            job_config['segment_start_time'] = segment_start_time
            job_config['segment_stop_time'] = segment_stop_time

            response_json = self._get_response_from_subprocess(job_config)

            log.info('Processing complete.')

            tracks = self._create_tracks(job, response_json)

            return tracks

        except Exception:
            log.exception('Failed to complete job due to the following exception:')
            raise


    def _get_response_from_subprocess(self, job_config: dict) -> dict:
        schema_str = job_config['generation_json_schema']
        schema_json = json.loads(schema_str)

        attempts = dict(
            base=0,
            timeline=0)

        max_attempts = job_config['generation_max_attempts']
        timeline_check_target_threshold = job_config['timeline_check_target_threshold']
        segment_start_time = job_config['segment_start_time']
        segment_stop_time = job_config['segment_stop_time']

        response_json = {}
        error = None
        while max(attempts.values()) < max_attempts:
            response = self.child_process.send_job_get_response(job_config)
            response_json, error = self._check_response(attempts, max_attempts, schema_json, response)
            if error is not None:
                continue

            event_timeline = response_json['video_event_timeline']

            if timeline_check_target_threshold != -1:
                error = self._check_timeline(
                    timeline_check_target_threshold, attempts, max_attempts, segment_start_time, segment_stop_time, event_timeline)
                if error is not None:
                    continue

            break

        if error:
            raise mpf.DetectionError.DETECTION_FAILED.exception(f'Subprocess failed: {error}')

        return response_json


    def _check_response(self, attempts: dict, max_attempts: int, schema_json: dict, response: str) -> Tuple[dict, str]:
        response_json = None

        if not response:
            error = 'Empty response.'
            log.warning(error)
            log.warning(f'Failed {attempts["base"] + 1} of {max_attempts} base attempts.')
            attempts['base'] += 1
            return None, error

        try:
            response_json = json.loads(response)
        except ValueError as ve:
            error = 'Response is not valid JSON.'
            log.warning(error)
            log.warning(str(ve))
            log.warning(f'Failed {attempts["base"] + 1} of {max_attempts} base attempts.')
            attempts['base'] += 1
            return response_json, error

        try:
            validate(response_json, schema_json)
        except ValidationError as ve:
            error = 'Response JSON is not in the desired format.'
            log.warning(error)
            log.warning(str(ve))
            log.warning(f'Failed {attempts["base"] + 1} of {max_attempts} base attempts.')
            attempts['base'] += 1
            return response_json, error
        
        return response_json, None

    def _check_timeline(self, threshold: float, attempts: dict, max_attempts: int,
                        segment_start_time: float, segment_stop_time: float, event_timeline: list) -> str:

        error = None
        for event in event_timeline:
            timestamp_start = _get_timestamp_value(event["timestamp_start"])
            timestamp_end = _get_timestamp_value(event["timestamp_end"])

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
            
            if (segment_start_time - timestamp_start) > threshold:
                error = (f'Timeline event start time occurs too soon before segment start time. '
                         f'({segment_start_time} - {timestamp_start}) > {threshold}.')
                break

            if (timestamp_end - segment_stop_time) > threshold:
                error = (f'Timeline event end time occurs too late after segment stop time. '
                         f'({timestamp_end} - {segment_stop_time}) > {threshold}.')
                break
        
        if not error:
            min_event_start = min(list(map(lambda d: _get_timestamp_value(d.get('timestamp_start')),
                                           filter(lambda d: 'timestamp_start' in d, event_timeline))))

            if abs(segment_start_time - min_event_start) > threshold:
                error = (f'Min timeline event start time not close enough to segment start time. '
                         f'abs({segment_start_time} - {min_event_start}) > {threshold}.')

        if not error:
            max_event_end = max(list(map(lambda d: _get_timestamp_value(d.get('timestamp_end')),
                                         filter(lambda d: 'timestamp_end' in d, event_timeline))))

            if abs(max_event_end - segment_stop_time) > threshold:
                error = (f'Max timeline event end time not close enough to segment stop time. '
                         f'abs({max_event_end} - {segment_stop_time}) > {threshold}.')

        if error:
            log.warning(error)
            log.warning(f'Failed {attempts["timeline"] + 1} of {max_attempts} timeline attempts.')
            attempts['timeline'] += 1
            return error

        return None


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


    def _create_tracks(self, job: mpf.VideoJob, response_json: dict) -> Iterable[mpf.VideoTrack]:
        tracks = []
        frame_width = 0
        frame_height = 0
        if 'FRAME_WIDTH' in job.media_properties:
            frame_width = int(job.media_properties['FRAME_WIDTH'])
        if 'FRAME_HEIGHT' in job.media_properties:
            frame_height = int(job.media_properties['FRAME_HEIGHT'])

        if response_json['video_event_timeline']:
            summary_track = self._create_segment_summary_track(job, response_json)
            tracks.append(summary_track)

            segment_id = summary_track.detection_properties['SEGMENT ID']
            video_fps = float(job.media_properties['FPS'])

            for event in response_json['video_event_timeline']:
                # get offset start/stop times in milliseconds
                event_start_time = int(_get_timestamp_value(event['timestamp_start']) * 1000)
                event_stop_time = int(_get_timestamp_value(event['timestamp_end']) * 1000)

                offset_start_frame = int((event_start_time * video_fps) / 1000)
                offset_stop_frame = int((event_stop_time * video_fps) / 1000) - 1

                detection_properties={
                    "SEGMENT ID": segment_id,
                    "TEXT": event['description']
                }
                
                offset_middle_frame = int((offset_stop_frame - offset_start_frame) / 2) + offset_start_frame

                # check offset_stop_frame
                if offset_stop_frame > job.stop_frame:
                    log.debug(f'offset_stop_frame outside of acceptable range '
                              f'({offset_stop_frame} > {job.stop_frame}), setting offset_stop_frame to {job.stop_frame}')
                    offset_stop_frame = job.stop_frame
                elif offset_stop_frame < job.start_frame:
                    log.debug(f'offset_stop_frame outside of acceptable range '
                              f'({offset_stop_frame} < {job.start_frame}), setting offset_stop_frame to {job.start_frame}')
                    offset_stop_frame = job.start_frame

                # check offset_middle_frame
                if offset_middle_frame > job.stop_frame:
                    log.debug(f'offset_middle_frame outside of acceptable range '
                              f'({offset_middle_frame} > {job.stop_frame}), setting offset_middle_frame to {job.stop_frame}')
                    offset_middle_frame = job.stop_frame
                elif offset_middle_frame < job.start_frame:
                    log.debug(f'offset_middle_frame outside of acceptable range '
                              f'({offset_middle_frame} < {job.start_frame}), setting offset_middle_frame to {job.start_frame}')
                    offset_middle_frame = job.start_frame

                # check offset_start_frame
                if offset_start_frame > job.stop_frame:
                    log.debug(f'offset_start_frame outside of acceptable range '
                              f'({offset_start_frame} > {job.stop_frame}), setting offset_start_frame to {job.stop_frame}')
                    offset_start_frame = job.stop_frame
                elif offset_start_frame < job.start_frame:
                    log.debug(f'offset_start_frame outside of acceptable range '
                              f'({offset_start_frame} < {job.start_frame}), setting offset_start_frame to {job.start_frame}')
                    offset_start_frame = job.start_frame

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
            
        log.info('Processing complete. Video segment %s summarized in %d tracks.' % (segment_id, len(tracks)))
        return tracks

def _get_timestamp_value(seconds: any) -> float:
    if isinstance(seconds, str):
        secval = float(seconds.replace('s', ''))
    else:
        secval = float(seconds)
    return secval

def _parse_properties(props: Mapping[str, str], segment_start_time: float) -> dict:
    process_fps = mpf_util.get_property(
        props, 'PROCESS_FPS', 1)
    max_frames = mpf_util.get_property(
        props, 'MAX_FRAMES', 180)
    max_new_tokens = mpf_util.get_property(
        props, 'MAX_NEW_TOKENS', 1024)

    generation_prompt_path = mpf_util.get_property(
        props, 'GENERATION_PROMPT_PATH', 'default_prompt.txt')
    generation_json_schema_path = mpf_util.get_property(
        props, 'GENERATION_JSON_SCHEMA_PATH', 'default_schema.json')
    system_prompt_path = mpf_util.get_property(
        props, 'SYSTEM_PROMPT_PATH', '')
    generation_max_attempts = mpf_util.get_property(
        props, 'GENERATION_MAX_ATTEMPTS', 5)
    timeline_check_target_threshold = mpf_util.get_property(
        props, 'TIMELINE_CHECK_TARGET_THRESHOLD', 10)
    timeline_check_threshold_acceptable = mpf_util.get_property(
        props, 'TIMELINE_CHECK_THRESHOLD_ACCEPTABLE', 30)

    generation_prompt = _read_file(generation_prompt_path) % (segment_start_time)

    generation_json_schema = _read_file(generation_json_schema_path)

    system_prompt = generation_prompt
    if system_prompt_path:
        system_prompt = _read_file(system_prompt_path)

    return dict(
        process_fps = process_fps,
        max_frames = max_frames,
        max_new_tokens = max_new_tokens,
        generation_prompt = generation_prompt,
        generation_json_schema = generation_json_schema,
        system_prompt = system_prompt,
        generation_max_attempts = generation_max_attempts,
        timeline_check_target_threshold = timeline_check_target_threshold
    )


def _read_file(path: str) -> str:
    try:
        expanded_path = os.path.expandvars(path)
        if os.path.dirname(expanded_path):
            with open(expanded_path, "r") as f:
                return f.read()
        return importlib.resources.read_text(__name__, expanded_path).strip()
    except:
        raise mpf.DetectionError.COULD_NOT_READ_DATAFILE.exception(f"Could not read \"{path}\".")


class ChildProcess:

    def __init__(self, start_cmd: Tuple[str, ...]):
        parent_socket, child_socket = socket.socketpair()
        with parent_socket, child_socket:
            env = {**os.environ, 'MPF_SOCKET_FD': str(child_socket.fileno())}
            self._proc = subprocess.Popen(
                    start_cmd,
                    pass_fds=(child_socket.fileno(),),
                    env=env)
            self._socket = parent_socket.makefile('rwb')

    def __del__(self):
        print("Terminating subprocess...")
        self._socket.close()
        self._proc.terminate()
        self._proc.wait()
        print("Subprocess terminated")

    def send_job_get_response(self, config: dict):
        job_bytes = pickle.dumps(config)
        self._socket.write(len(job_bytes).to_bytes(4, 'little'))
        self._socket.write(job_bytes)
        self._socket.flush()

        response_length = int.from_bytes(self._socket.read(4), 'little')
        if response_length == 0:
            error_length = int.from_bytes(self._socket.read(4), 'little')
            error = pickle.loads(self._socket.read(error_length))
            raise mpf.DetectionError.DETECTION_FAILED.exception(f'Subprocess failed: {error}')

        return pickle.loads(self._socket.read(response_length))