#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
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

import os
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional, Mapping

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from .acs_speech_processor import SpeakerInfo, parse_segments_str
from .azure_connect import AcsServerInfo
from .azure_utils import ISO6393_TO_BCP47


class TriggerMismatch(Exception):
    """ Exception raised when the feed-forward track does not meet the trigger.

        :param trigger_key: The trigger key defined by TRIGGER property
        :param expected: The expected value defined by TRIGGER property
        :param trigger_val: The actual value of the trigger property (if present)
    """
    def __init__(self, trigger_key, expected, trigger_val=None):
        self.trigger_key = trigger_key
        self.expected = expected
        self.trigger_val = trigger_val

    def __str__(self):
        if self.trigger_val is None:
            return f"Trigger property {self.trigger_key} not present in feed-forward detection properties"
        return f"Expected {self.trigger_key} to be {self.expected}, got {self.trigger_val}"


class JobConfig(object):
    """
    Attributes:
        job_name (str): Job name, used for logging
        target_file (Path): File location of audio or video input
        start_time (int): Start time of the audio (in milliseconds)
        stop_time (Optional[int]): Stop time of the audio (in milliseconds)
        is_feed_forward_job (bool): Whether the job has a feed-forward track
        speaker (Optional[Speaker]): The speaker information contained in the
            feed-forward track
        server_info (AcsServerInfo): Information for the ACS server
        blob_access_time (int): Timespan for which access signatures to blob
            storage containers will be valid, in minutes.
        expiry (int): Timespan after which to automatically delete
            transcriptions, in minutes.
        language (str): The language/locale to use for transcription.
        diarize (bool): Whether to split audio into speaker turns. Supports
            two-speaker conversation.
        cleanup (bool): Whether to delete recording blobs after processing.
    """
    def __init__(self, job: Union[mpf.AudioJob, mpf.VideoJob]):
        # General job data
        self.job_name: str
        self.target_file: Path
        self.start_time: int
        self.stop_time: Optional[int]
        self._add_job_data(job)

        # Properties related to dynamic speech pipelines
        self.is_feed_forward_job: bool
        self.speaker: Optional[SpeakerInfo] = None
        self._add_dynamic_speech_properties(job)

        # Job properties
        self.server_info: AcsServerInfo
        self.blob_access_time: int
        self.expiry: int
        self.language: str
        self.diarize: bool
        self.cleanup: bool
        self._add_job_properties(job.job_properties)

    @staticmethod
    def _get_job_property_or_env_value(
                property_name: str,
                job_properties: Mapping[str, str]
            ) -> str:
        property_value = job_properties.get(property_name)
        if property_value:
            return property_value

        env_value = os.getenv(property_name)
        if env_value:
            return env_value

        raise mpf.DetectionException(
            f'The "{property_name}" property must be provided as a '
            'job property or environment variable.',
            mpf.DetectionError.MISSING_PROPERTY)

    def _add_job_data(self, job: Union[mpf.AudioJob, mpf.VideoJob]):
        self.target_file = Path(job.data_uri)
        self.job_name = job.job_name
        if isinstance(job, mpf.VideoJob):
            start_frame = job.start_frame
            stop_frame = job.stop_frame
            if stop_frame < 0:
                stop_frame = None

            # Convert frame locations to timestamps
            media_frame_count = int(job.media_properties.get('FRAME_COUNT', -1))
            media_duration = float(job.media_properties.get('DURATION', -1))
            fpms = float(job.media_properties['FPS']) / 1000.0
            self.start_time = int(start_frame / fpms)

            # The WFM will pass a job stop frame equal to FRAME_COUNT-1 for the
            #  last video segment. We want to use the detected DURATION in such
            #  cases instead to ensure we process the entire audio track.
            #  Only use the job stop frame if it differs from FRAME_COUNT-1.
            if stop_frame is not None and stop_frame < media_frame_count - 1:
                self.stop_time = int(stop_frame / fpms)
            elif media_duration > 0:
                self.stop_time = int(media_duration)
            elif media_frame_count > 0:
                self.stop_time = int(media_frame_count / fpms)
            else:
                self.stop_time = None
        else:
            self.start_time = job.start_time
            self.stop_time = job.stop_time
            if self.stop_time < 0:
                self.stop_time = None

    def _add_dynamic_speech_properties(self, job: Union[mpf.AudioJob, mpf.VideoJob]):
        self.is_feed_forward_job = (job.feed_forward_track is not None)
        if not self.is_feed_forward_job:
            return

        # Check TRIGGER is met
        trigger = mpf_util.get_property(
            properties=job.job_properties,
            key='TRIGGER',
            default_value='',
            prop_type=str
        )
        t_key, t_val = trigger.strip().split('=') if trigger else (None, None)

        ff_det_props = job.feed_forward_track.detection_properties

        # If trigger key is in feed-forward properties, only run
        #  transcription if the value matches trigger_val
        if t_key not in ff_det_props:
            raise TriggerMismatch(t_key, t_val)
        if ff_det_props[t_key] != t_val:
            raise TriggerMismatch(t_key, t_val, ff_det_props[t_key])

        speaker_id = mpf_util.get_property(
            properties=ff_det_props,
            key='SPEAKER_ID',
            default_value='0',
            prop_type=str
        )
        # If speaker ID was overwritten, use long speaker ID
        if speaker_id == '0':
            speaker_id = mpf_util.get_property(
                properties=ff_det_props,
                key='LONG_SPEAKER_ID',
                default_value='0',
                prop_type=str
            )

        gender = mpf_util.get_property(
            properties=ff_det_props,
            key='GENDER',
            default_value='',
            prop_type=str
        )

        gender_score = mpf_util.get_property(
            properties=ff_det_props,
            key='GENDER_CONFIDENCE',
            default_value=-1.0,
            prop_type=float
        )

        languages = mpf_util.get_property(
            properties=ff_det_props,
            key='SPEAKER_LANGUAGES',
            default_value='',
            prop_type=str
        )

        language_scores = mpf_util.get_property(
            properties=ff_det_props,
            key='SPEAKER_LANGUAGE_CONFIDENCES',
            default_value='',
            prop_type=str
        )

        languages = [lab.strip().lower() for lab in languages.split(',')]
        language_scores = [float(s.strip()) for s in language_scores.split(',')]
        language_scores = defaultdict(
            lambda: -1.0,
            zip(languages, language_scores)
        )

        segments_str = mpf_util.get_property(
            properties=ff_det_props,
            key='VOICED_SEGMENTS',
            default_value='',
            prop_type=str
        )
        speech_segs = parse_segments_str(
            segments_string=segments_str,
            media_start_time=self.start_time,
            media_stop_time=self.stop_time
        )

        language_iso = mpf_util.get_property(
            properties=ff_det_props,
            key='LANGUAGE',
            default_value='',
            prop_type=str
        )
        language_iso = language_iso.strip().upper()
        languages = ISO6393_TO_BCP47.get(language_iso, None)
        if languages is None:
            raise mpf.DetectionException(
                f"ISO 639-3 code '{language_iso}' provided in feed-forward track"
                f" does not correspond to a BCP-47 language code supported by "
                f" Azure Speech-to-Text.",
                mpf.DetectionError.INVALID_PROPERTY
            )

        self.speaker = SpeakerInfo(
            speaker_id=speaker_id,
            language=languages[0],
            language_scores=language_scores,
            gender=gender,
            gender_score=gender_score,
            speech_segs=speech_segs
        )

    def _add_job_properties(self, job_properties: Mapping[str, str]):
        acs_url = self._get_job_property_or_env_value('ACS_URL', job_properties)
        acs_subscription_key = self._get_job_property_or_env_value('ACS_SUBSCRIPTION_KEY', job_properties)
        acs_blob_container_url = self._get_job_property_or_env_value('ACS_BLOB_CONTAINER_URL', job_properties)
        acs_blob_service_key = self._get_job_property_or_env_value('ACS_BLOB_SERVICE_KEY', job_properties)
        http_retry = mpf_util.HttpRetry.from_properties(job_properties)

        http_max_attempts = mpf_util.get_property(
            properties=job_properties,
            key='COMPONENT_HTTP_RETRY_MAX_ATTEMPTS',
            default_value=10,
            prop_type=int
        )

        self.server_info = AcsServerInfo(
            url=acs_url,
            subscription_key=acs_subscription_key,
            blob_container_url=acs_blob_container_url,
            blob_service_key=acs_blob_service_key,
            http_retry=http_retry,
            http_max_attempts=http_max_attempts
        )

        self.blob_access_time = mpf_util.get_property(
            properties=job_properties,
            key='BLOB_ACCESS_TIME',
            default_value=120,
            prop_type=int
        )

        self.expiry = mpf_util.get_property(
            properties=job_properties,
            key='TRANSCRIPTION_EXPIRATION',
            default_value=120,
            prop_type=int
        )

        self.language = mpf_util.get_property(
            properties=job_properties,
            key='LANGUAGE',
            default_value='en-US',
            prop_type=str
        )

        self.diarize = mpf_util.get_property(
            properties=job_properties,
            key='DIARIZE',
            default_value=True,
            prop_type=bool
        )
        if self.is_feed_forward_job:
            self.diarize = False
            self.language = self.speaker.language

        self.cleanup = mpf_util.get_property(
            properties=job_properties,
            key='CLEANUP',
            default_value=True,
            prop_type=bool
        )
