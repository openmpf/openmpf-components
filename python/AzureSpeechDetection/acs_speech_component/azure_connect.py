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

import json
import logging
import time
from urllib import request, parse
from datetime import datetime, timedelta
from math import ceil
from typing import NamedTuple

from azure.storage.blob import (
    ResourceTypes, AccountSasPermissions,
    generate_account_sas, ContainerClient
)

import mpf_component_api as mpf
import mpf_component_util as mpf_util


logger = logging.getLogger('AcsSpeechComponent')

# Target the latest GA Speech-to-text REST API.
DEFAULT_ACS_API_VERSION = "2025-10-15"

SUPPORTED_ACS_API_VERSIONS = frozenset({
    "2024-11-15",
    "2025-10-15",
})

class AcsServerInfo(NamedTuple):
    url: str
    api_version: str
    subscription_key: str
    blob_container_url: str
    blob_service_key: str
    http_retry: mpf_util.HttpRetry
    http_max_attempts: int
    use_sas_auth: bool


class AzureConnection(object):
    def __init__(self):
        self.url = None
        self.api_version = DEFAULT_ACS_API_VERSION
        self.subscription_key = None
        self.blob_container_url = None
        self.blob_service_key = None
        self.acs_headers = {}
        self.container_client = None
        self.http_retry = None
        self.http_max_attempts = None
        self.use_sas_auth = False

        self.supported_locales = set()
        self.submit_locales = set()
        self.transcribe_locales = set()

        self._base_url = None
        self._transcriptions_submit_url = None
        self._locales_url = None


    @staticmethod
    def _normalize_locale_for_lookup(locale: str) -> str:
        if not locale:
            return locale

        sep = '-' if '-' in locale else ('_' if '_' in locale else None)
        if sep is None:
            return locale

        parts = locale.split(sep)
        out = [parts[0].lower()]
        for p in parts[1:]:
            if len(p) == 2:
                out.append(p.upper())      # region
            elif len(p) == 4:
                out.append(p.title())      # script
            else:
                out.append(p.lower())      # variant / dialect
        return sep.join(out)

    @classmethod
    def _expand_locale_set(cls, locales):
        expanded = set()
        for loc in locales:
            expanded.add(loc)
            expanded.add(cls._normalize_locale_for_lookup(loc))
        return expanded

    def update_acs(self, server_info: AcsServerInfo):
        """
        Normalize the Speech-to-text endpoint and construct
        transcriptions / locales URLs using the configured API version.

        Expected ACS_URL formats (examples):
        - https://<region>.api.cognitive.microsoft.com/speechtotext
        - https://<region>.api.cognitive.microsoft.com/speechtotext/transcriptions

        Both will be normalized back to the /speechtotext base.
        """
        raw_url = server_info.url
        self.api_version = server_info.api_version or DEFAULT_ACS_API_VERSION

        parsed = parse.urlparse(raw_url)
        path = parsed.path.rstrip('/')

        # Find the /speechtotext root in whatever path we were given.
        idx = path.find('/speechtotext')
        if idx != -1:
            path = path[: idx + len('/speechtotext')]

        base = parsed._replace(path=path.rstrip('/'), query='', params='', fragment='')
        base_url = parse.urlunparse(base).rstrip('/')

        self._base_url = base_url
        self._transcriptions_submit_url = (
            f"{base_url}/transcriptions:submit?api-version={self.api_version}"
        )
        self._locales_url = (
            f"{base_url}/transcriptions/locales?api-version={self.api_version}"
        )

        # Keep self.url as the base endpoint to avoid accidental query-string concatenation.
        self.url = self._base_url
        self.subscription_key = server_info.subscription_key
        self.acs_headers = {
            'Ocp-Apim-Subscription-Key': server_info.subscription_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        self.http_retry = server_info.http_retry
        self.use_sas_auth = server_info.use_sas_auth

        logger.info('Retrieving valid transcription locales')
        req = request.Request(
            url=self._locales_url,
            headers=self.acs_headers,
            method='GET'
        )
        with self.http_retry.urlopen(req) as response:
            raw_locales = json.load(response)

        # New 2024/2025 locales endpoint returns a dict like:
        #   { "Submit": [...], "Transcribe": [...] }
        # Normalize to sets via either format (V3, 2025).
        if isinstance(raw_locales, dict):
            raw_submit = set(raw_locales.get('Submit', []))
            raw_transcribe = set(raw_locales.get('Transcribe', []))

            self.submit_locales = self._expand_locale_set(raw_submit)
            self.transcribe_locales = self._expand_locale_set(raw_transcribe)
            self.supported_locales = self.submit_locales

            logger.info('Supported locales (Submit): %s', sorted(raw_submit))
            logger.info('Supported locales (Transcribe): %s', sorted(raw_transcribe))
        else:
            raw_supported = set(raw_locales)
            self.supported_locales = self._expand_locale_set(raw_supported)
            self.submit_locales = self.supported_locales
            self.transcribe_locales = self.supported_locales

            logger.info('Supported locales: %s', sorted(raw_supported))

        if (self.blob_container_url != server_info.blob_container_url
                or self.blob_service_key != server_info.blob_service_key
                or self.http_max_attempts != server_info.http_max_attempts):
            logger.debug('Updating ACS connection')
            self.blob_container_url = server_info.blob_container_url
            self.blob_service_key = server_info.blob_service_key
            self.http_max_attempts = server_info.http_max_attempts
            self.container_client = ContainerClient.from_container_url(
                server_info.blob_container_url,
                server_info.blob_service_key,
                retry_total=server_info.http_max_attempts,
                retry_connect=server_info.http_max_attempts,
                retry_read=server_info.http_max_attempts,
                retry_status=server_info.http_max_attempts
            )
        else:
            logger.debug('ACS arguments unchanged')

    def get_blob_client(self, recording_id):
        return self.container_client.get_blob_client(recording_id)

    def generate_account_sas(self, time_limit: timedelta) -> str:
        # Shared access signature (SAS) is required for the ACS Speech
        # service to access the container.
        return generate_account_sas(
            self.container_client.account_name,
            account_key=self.container_client.credential.account_key,
            resource_types=ResourceTypes(object=True),
            permission=AccountSasPermissions(read=True),
            expiry=(datetime.utcnow() + time_limit)
        )

    def upload_file_to_blob(self, audio_bytes, recording_id, blob_access_time):
        try:
            blob_client = self.get_blob_client(recording_id)
            blob_client.upload_blob(audio_bytes)
        except Exception as e:
            if 'blob already exists' in str(e):
                logger.info('Blob exists for file {}'.format(recording_id))
            else:
                logger.error('Uploading file to blob failed for file {}'.format(recording_id))
                raise mpf.DetectionError.NETWORK_ERROR.exception(
                    'Uploading file to blob failed due to: ' + str(e)) from e

        time_limit = timedelta(minutes=blob_access_time)
        blob_url = f'{self.container_client.url}/{recording_id}'
        if self.use_sas_auth:
            blob_url += '?' + self.generate_account_sas(time_limit)
        return blob_url

    def submit_batch_transcription(self, recording_url, job_name,
                                   diarize, language, expiry):
        if language not in self.supported_locales:
            raise ValueError(
                f"Provided language ({language}) not supported. "
                "Refer to component README for list of supported "
                "locales for Azure Speech."
            )

        logger.info('Submitting batch transcription...')

        # API 2025-10-15 expects TTL as integer hours.
        # Docs specify minimum 6 hours and recommend 48 hours.
        ttl_hours = max(6, int(ceil(expiry / 60.0)))

        properties = dict(
            wordLevelTimestampsEnabled=True,
            profanityFilterMode='None',
            timeToLiveHours=ttl_hours,
            diarization=dict(enabled=bool(diarize), maxSpeakers=2),
        )

        data = dict(
            contentUrls=[recording_url],
            displayName=job_name,
            description=job_name,
            locale=language,
            properties=properties
        )

        payload = json.dumps(data).encode()
        req = request.Request(
            url=self._transcriptions_submit_url,
            data=payload,
            headers=self.acs_headers,
            method='POST'
        )
        try:
            with self.http_retry.urlopen(req) as response:
                body = json.load(response)
                location_header = response.getheader('Location')
                self_link = body.get('self')

                logger.info('Submit Location header: %s', location_header)
                logger.info('Submit self link: %s', self_link)

                # Prefer the canonical transcription resource URL from the body.
                location = self_link or location_header
                if not location:
                    raise mpf.DetectionException(
                        'Submit transcription succeeded but no polling URL was returned.',
                        mpf.DetectionError.DETECTION_FAILED
                    )

                # Check: the poll URL must be a transcription resource,
                # not the submit action endpoint.
                if '/transcriptions:submit/' in location:
                    raise mpf.DetectionException(
                        f'Invalid polling URL returned from submit: {location}',
                        mpf.DetectionError.DETECTION_FAILED
                    )

                return location

        except Exception as e:
            raise mpf.DetectionException(
                'Failed to post job. Error message: {:s}'.format(str(e)),
                mpf.DetectionError.DETECTION_FAILED
            )

    def poll_for_result(self, location):
        req = request.Request(
            url=location,
            headers=self.acs_headers,
            method='GET'
        )
        logger.info('Polling for transcription success')
        while True:
            with self.http_retry.urlopen(req) as response:
                result = json.load(response)
                status = result['status']
                if status == 'Succeeded':
                    logger.debug('Transcription succeeded')
                    break
                elif status == 'Failed':
                    logger.debug('Transcription failed')
                    break
                else:
                    retry_after = int(response.info()['Retry-After'])
                    logger.info(
                        'Status is {}. Retry in {} seconds.'.format(
                            status,
                            retry_after
                        )
                    )
            time.sleep(retry_after)
        return result

    def get_transcription(self, result):
        results_uri = result['links']['files']
        req = request.Request(
            url=results_uri,
            headers=self.acs_headers,
            method='GET'
        )

        logger.info(f'Retrieving transcription file location from {results_uri}')
        with self.http_retry.urlopen(req) as response:
            files = json.load(response)

        transcript_uri = next(
            v['links']['contentUrl'] for v in files['values']
            if v['kind'] == 'Transcription'
        )
        logger.info(f'Retrieving transcription result from {transcript_uri}')
        with self.http_retry.urlopen(transcript_uri) as response:
            return json.load(response)

    def delete_transcription(self, location):
        logger.info('Deleting transcription...')
        req = request.Request(
            url=location,
            headers=self.acs_headers,
            method='DELETE'
        )
        try:
            with self.http_retry.urlopen(req):
                pass
        except mpf.DetectionException:
            # If the transcription task doesn't exist, ignore.
            logger.warning(
                f'Transcription task deletion failed. This transcription '
                f'should be deleted manually: {location}'
            )

    def delete_blob(self, recording_id):
        logger.info('Deleting blob...')
        self.container_client.delete_blob(recording_id)