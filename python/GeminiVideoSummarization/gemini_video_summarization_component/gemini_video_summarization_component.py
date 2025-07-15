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
import json
import re
import importlib.resources

from tenacity import retry, wait_random_exponential, stop_after_delay, retry_if_exception

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('GeminiVideoSummarizationComponent')

IGNORE_WORDS = ['unsure', 'none', 'false', 'no', 'unclear', 'n/a', 'unspecified', 'unknown', 'unreadable', 'not visible', 'none visible']
IGNORE_PREFIXES = tuple([s + ' ' for s in IGNORE_WORDS])

class GeminiVideoSummarizationComponent:

    def __init__(self):
        
        self.google_application_credentials = ''
        self.project_id = ''
        self.bucket_name = ''
        self.label_prefix = ''
        self.label_user = ''
        self.label_purpose = ''

    def get_detections_from_video(self, video_job: mpf.VideoJob) -> Iterable[mpf.VideoTrack]:
        logger.info('Received video job: %s', video_job.job_name)

        tracks = []

        config = JobConfig(video_job.job_properties, video_job.media_properties)

        self.google_application_credentials = config.google_application_credentials
        self.project_id = config.project_id
        self.bucket_name = config.bucket_name
        self.label_prefix = config.label_prefix
        self.label_user = config.label_user
        self.label_purpose = config.label_purpose

        video_capture = mpf_util.VideoCapture(video_job)

        prompt = _read_file(config.prompt_config_path)

        if video_job.feed_forward_track is None:
            response = self._get_gemini_response(config.model_name, video_job.data_uri, prompt)
            tracks.append(mpf.VideoTrack(        
                start_frame=0,
                stop_frame=video_capture.frame_count - 1,
                detection_properties={"TEXT": response}
                )
            )
        else:
            raise mpf.DetectionException(
                f"Feed forward jobs not supported by this component: {e}",
                mpf.DetectionError.DETECTION_FAILED)

        logger.info(f"Job complete. Found {len(tracks)} tracks.")
        return tracks

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
        try:
            process = subprocess.Popen([
                "/gemini-subprocess/venv/bin/python3",
                "/gemini-subprocess/gemini-process-video.py",
                "-m", model_name,
                "-d", data_uri,
                "-p", prompt,
                "-c", self.google_application_credentials,
                "-i", self.project_id,
                "-b", self.bucket_name,
                "-l", self.label_prefix,
                "-u", self.label_user,
                "-r", self.label_purpose
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
        except Exception as e:
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

def _read_file(path: str) -> str:
    try:
        if not os.path.isabs(path):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base_dir, path)
        with open(path, 'r') as file:
            return file.read()
    except:
        raise mpf.DetectionError.COULD_NOT_READ_DATAFILE.exception(f"Could not read \"{path}\".")

class JobConfig:
    def __init__(self, job_properties: Mapping[str, str], media_properties=None):
        self.prompt_config_path = self._get_prop(job_properties, "PROMPT_CONFIGURATION_PATH", "default_prompt.txt") 
        self.model_name = self._get_prop(job_properties, "MODEL_NAME", "gemini-2.5-pro")
        self.google_application_credentials = self._get_prop(job_properties, "GOOGLE_APPLICATION_CREDENTIALS", "")
        self.project_id = self._get_prop(job_properties, "PROJECT_ID", "")
        self.bucket_name = self._get_prop(job_properties, "BUCKET_NAME", "")
        self.label_prefix = self._get_prop(job_properties, "LABEL_PREFIX", "")
        self.label_user = self._get_prop(job_properties, "LABEL_USER", "")
        self.label_purpose = self._get_prop(job_properties, "LABEL_PURPOSE", "")

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