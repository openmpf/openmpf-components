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

import logging
import os
import pickle
import socket
import subprocess

from typing import Iterable, Mapping, Tuple

import mpf_component_api as mpf
import mpf_component_util as mpf_util

log = logging.getLogger('LlamaVideoSummarizationComponent')

DEFAULT_PROMPT = \
'''
Follow these instructions:

You are a helpful assistant. Provide information in following JSON format:

{
"video_summary": "A general short summary of the activity in the video",
"video_length": "Full length of the video in seconds",
"video_event_timeline": [
    {
        "timestamp": "event A start time in seconds - event A end time in seconds",
        "description": "Description of event A"
    },
    {
        "timestamp": "event B start time in seconds - event B end time in seconds",
        "description": "Description of event B"
    },
    {
        "timestamp": "event C start time in seconds - event C end time in seconds",
        "description": "Description of event C"
    }
],
}
'''

class LlamaVideoSummarizationComponent:

    def __init__(self):
        self.child_process = ChildProcess(['/llama/venv/bin/python3', '/llama/summarize_video.py'])

    def get_detections_from_video(self, job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        try:
            log.info('Received video job.')

            if job.feed_forward_track:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    'Component cannot process feed forward jobs.')
            
            job_config = JobConfig(job.job_properties)

            self.child_process.send_job(job_config)

            # TODO: Create subprocess and feed in pickled dict

            # detector = LanguageDetector(job.job_properties)
            # detector.add_language_detections(ff_track.detection_properties)
            # for ff_location in ff_track.frame_locations.values():
            #     detector.add_language_detections(ff_location.detection_properties)

            log.info('Processing complete.')

            track = mpf.VideoTrack(0, 0, detection_properties={'TEXT': 'HELLO'})

            return (track,)
        except Exception:
            log.exception('Failed to complete job due to the following exception:')
            raise


class JobConfig:

    def __init__(self, props: Mapping[str, str]):
        self.process_fps = mpf_util.get_property(
            props, 'PROCESS_FPS', 1)
        self.max_frames = mpf_util.get_property(
            props, 'MAX_FRAMES', 180)
        self.max_new_tokens = mpf_util.get_property(
            props, 'MAX_NEW_TOKENS', 1024)

        generation_prompt_path = os.path.expandvars(mpf_util.get_property(
            props, 'GENERATION_PROMPT_PATH', ''))
        system_prompt_path = os.path.expandvars(mpf_util.get_property(
            props, 'SYSTEM_PROMPT_PATH', ''))

        self.generation_prompt = DEFAULT_PROMPT
        if generation_prompt_path:
            with open(generation_prompt_path,"r") as f:
                self.generation_prompt = f.read()

        self.system_prompt = self.generation_prompt
        if system_prompt_path:
            with open(system_prompt_path,"r") as f:
                self.system_prompt = f.read()


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

    def send_job(self, job: JobConfig): # TODO: Annotate return type
        job = "REQUEST" # DEBUG
        job_bytes = pickle.dumps(job)
        self._socket.write(len(job_bytes).to_bytes(4, 'little'))
        self._socket.write(job_bytes)
        self._socket.flush()

        response_length = int.from_bytes(self._socket.read(4), 'little')
        response = pickle.loads(self._socket.read(response_length))
        print(f'{response=}')

        return response