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

import logging
import os
from typing import Union, Mapping

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from .azure_connect import AcsServerInfo
from .azure_utils import ISO6393_TO_BCP47


logger = logging.getLogger('AcsSpeechComponent')


class AzureJobConfig(mpf_util.DynamicSpeechJobConfig):
    """
    :ivar server_info: Information for the ACS server
    :ivar blob_access_time: Timespan for which access signatures to blob
        storage containers will be valid, in minutes.
    :ivar expiry: Timespan after which to automatically delete transcriptions,
        in minutes.
    :ivar language: The language/locale to use for transcription.
    :ivar diarize: Whether to split audio into speaker turns. Supports
        two-speaker conversation.
    :ivar cleanup: Whether to delete recording blobs after processing.
    """
    def __init__(self, job: Union[mpf.AudioJob, mpf.VideoJob]):
        # Job properties
        self.server_info: AcsServerInfo
        self.blob_access_time: int
        self.expiry: int
        self.language: str
        self.diarize: bool
        self.cleanup: bool
        super().__init__(job)

        if self.is_triggered_job:
            self.diarize = False

    def _add_feed_forward_properties(self, job: Union[mpf.AudioJob, mpf.VideoJob]):
        super()._add_feed_forward_properties(job)

        language_iso = self.speaker.language
        languages = ISO6393_TO_BCP47.get(language_iso, None)
        if languages is None:
            logger.warning(
                f"ISO 639-3 code '{language_iso}' provided in feed-forward "
                f"track does not correspond to a BCP-47 language code.",
            )
            languages = [language_iso]

        self.speaker = mpf_util.SpeakerInfo(
            speaker_id=self.speaker.speaker_id,
            language=languages[0],
            language_scores=self.speaker.language_scores,
            gender=self.speaker.gender,
            gender_score=self.speaker.gender_score,
            speech_segs=self.speaker.speech_segs
        )

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

        self.cleanup = mpf_util.get_property(
            properties=job_properties,
            key='CLEANUP',
            default_value=True,
            prop_type=bool
        )
