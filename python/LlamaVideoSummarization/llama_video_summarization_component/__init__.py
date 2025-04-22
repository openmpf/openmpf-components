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
            
            actual_segment_length = (job.stop_frame - job.start_frame) / float(job.media_properties['FPS'])

            job_config = _parse_properties(job.job_properties, actual_segment_length)
            job_config['video_path'] = job.data_uri
            job_config['max_length'] = actual_segment_length

            response_json = self._get_detections_from_subprocess(job_config)

            log.info('Processing complete.')

            tracks = _create_tracks(job, response_json)

            return tracks

        except Exception:
            log.exception('Failed to complete job due to the following exception:')
            raise

    def _get_detections_from_subprocess(self, job_config: dict) -> dict:
        schema_str = job_config['generation_json_schema']
        schema_json = json.loads(schema_str)

        max_attempts = job_config['generation_max_attempts']
        attempts = dict(timeline=0,
                segment_length=0,
                other=0)

        timeline_check_threshold = job_config['timeline_check_threshold']

        segment_length_check_threshold = job_config['segment_length_check_threshold']

        last_error = None
        while max(attempts.values()) <= max_attempts:
            response = self.child_process.send_job_get_response(job_config)

            if not response:
                last_error = f'Empty response.'
                log.warning(last_error, f'Failed {attempts["other"] + 1} of {max_attempts} attempts.')
                attempts['other'] += 1
                continue

            try:
                response_json = json.loads(response)
            except ValueError:
                last_error = f'Response is not valid JSON.'
                log.warning(last_error, f'Failed {attempts["other"] + 1} of {max_attempts} attempts.')
                attempts['other'] += 1
                continue

            try:
                validate(response_json, schema_json)
            except ValidationError:
                last_error = 'Response JSON is not in the desired format.'
                log.warning(last_error, f'Failed {attempts["other"] + 1} of {max_attempts} attempts.')
                attempts['other'] += 1
                continue

            if segment_length_check_threshold != -1:
                video_length = float(response_json['video_length'])
                max_length = float(job_config['max_length'])
                if abs(video_length - max_length) > segment_length_check_threshold:
                    last_error = (f'Video length doesn\'t match expected segment length. '
                        f'abs({video_length} - {max_length}) > {segment_length_check_threshold}.')
                    log.warning(last_error, f'Failed {attempts["segment_length"] + 1} of {max_attempts} attempts.')
                    attempts['segment_length'] += 1
                    continue

            if timeline_check_threshold != -1:
                video_length = float(response_json['video_length'])
                last_event_end = float(response_json['video_event_timeline'][-1]['timestamp_end'])
                if abs(video_length - last_event_end) > timeline_check_threshold:
                    last_error = (f'Video length doesn\'t correspond with last event timestamp end. '
                        f'abs({video_length} - {last_event_end}) > {timeline_check_threshold}.')
                    log.warning(last_error, f'Failed {attempts["timeline"] + 1} of {max_attempts} attempts.')
                    attempts['timeline'] += 1
                    continue

            last_error = None
            break

        if last_error:
            raise mpf.DetectionError.DETECTION_FAILED.exception(f'Subprocess failed: {last_error}')

        return response_json

def _create_segment_summary_track(job: mpf.VideoJob, response_json: dict) -> mpf.VideoTrack:
    start_frame = job.start_frame
    video_fps = float(job.media_properties['FPS'])
    video_secs = float(response_json['video_length'])
    calculated_stop_frame = job.start_frame + int(video_secs * video_fps)
    stop_frame = job.stop_frame - 1
    
    #segment_id = str(start_frame) + "-" + str(stop_frame)
    segment_id = str(job.start_frame) + "-" + str(job.stop_frame)
    detection_properties={
        "SEGMENT ID": segment_id,
        "SEGMENT LENGTH": response_json['video_length'],
        "SEGMENT SUMMARY": "TRUE",
        "TEXT": response_json['video_summary']
    }
    frame_width = int(job.media_properties['FRAME_WIDTH'])
    frame_height = int(job.media_properties['FRAME_HEIGHT'])
    middle_frame = int((stop_frame - start_frame) / 2) + start_frame

    track = mpf.VideoTrack(start_frame, stop_frame, 1.0,\
    # add dummy locations to prevent the Workflow Manager from dropping / truncating track
    frame_locations={
        middle_frame: mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0)
    },
    detection_properties=detection_properties)
    return track

def _create_tracks(job: mpf.VideoJob, response_json: dict) -> Iterable[mpf.VideoTrack]:
    tracks = []
    frame_width = int(job.media_properties['FRAME_WIDTH'])
    frame_height = int(job.media_properties['FRAME_HEIGHT'])
    if response_json['video_event_timeline']:
        summary_track = _create_segment_summary_track(job, response_json)
        tracks.append(summary_track)
        segment_id = summary_track.detection_properties['SEGMENT ID']

        # get FPS
        video_fps = float(job.media_properties['FPS'])
        segment_length = response_json['video_length']
        segment_start_time = int((job.start_frame / video_fps)*1000)
        for event in response_json['video_event_timeline']:
            # get offset start/stop times
            event_start_time = int(event['timestamp_start']*1000)
            event_stop_time = int(event['timestamp_end']*1000)
            # calculate event duration
            video_secs = float(event_stop_time - event_start_time)/1000

            # calculate offset start/stop frames
            event_start_frame = int((event_start_time * video_fps)/1000)
            offset_start_frame = job.start_frame + event_start_frame
            offset_stop_frame = (int(video_secs * video_fps) - 1) + offset_start_frame

            detection_properties={
                "SEGMENT ID": segment_id,
                "TEXT": event['description'],
                "TIMESTAMP START": event["timestamp_start"], # debug only, sanity check track start/stop times
                "TIMESTAMP END": event["timestamp_end"], # debug only, sanity check track start/stop times
            }
            offset_middle_frame = int((offset_stop_frame - offset_start_frame) / 2) + offset_start_frame
            track = mpf.VideoTrack(offset_start_frame, offset_stop_frame, 1.0,\
            # add dummy locations to prevent the Workflow Manager from dropping / truncating track
            frame_locations={
                offset_middle_frame: mpf.ImageLocation(0, 0, frame_width, frame_height, 1.0)
            },
            detection_properties=detection_properties)
            track.start_time = event_start_time
            track.stop_time = event_stop_time
            track.start_offset_time = event_start_time + segment_start_time
            track.stop_offset_time = event_stop_time + segment_start_time
            tracks.append(track)

    else: # no events timeline, create summary only
        tracks.append(_create_segment_summary_track(job, response_json))
    
    return tracks

def _parse_properties(props: Mapping[str, str], actual_segment_length) -> dict:
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
        props, 'GENERATION_MAX_ATTEMPTS', 3)
    timeline_check_threshold = mpf_util.get_property(
        props, 'TIMELINE_CHECK_THRESHOLD', 20)
    
    segment_length_check_threshold = mpf_util.get_property(
        props, 'SEGMENT_LENGTH_CHECK_THRESHOLD', 30)

    generation_prompt = _read_file(generation_prompt_path) % (actual_segment_length, actual_segment_length)
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
        timeline_check_threshold = timeline_check_threshold,
        segment_length_check_threshold = segment_length_check_threshold
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