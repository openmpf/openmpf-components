#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
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
import time
from urllib import request
from datetime import datetime, timedelta
from dateutil import relativedelta
from typing import (
    Dict, Optional, Sequence, List,
    Iterable, Set, Tuple, Mapping, NamedTuple
)

from azure.storage.blob import (
    ResourceTypes, AccountSasPermissions,
    generate_account_sas, ContainerClient
)

import mpf_component_api as mpf
import mpf_component_util as mpf_util


def minutes_to_iso8601(mins):
    """ Convert minutes to ISO 8601 duration, the format expected by
        Azure for timeToLive. Use dateutil to construct this string,
        to avoid issues with daylight savings, leap years, etc.
    """
    now = datetime.now()
    expiration = now + timedelta(minutes=mins)

    delta = relativedelta.relativedelta(expiration, now)
    date = f"{delta.years}Y{delta.months}M{delta.days}D"
    time = f"{delta.hours}H{delta.minutes}M{delta.seconds}S"

    return f"P{date}T{time}"


class AcsServerInfo(NamedTuple):
    url: str
    subscription_key: str
    blob_container_url: str
    blob_service_key: str
    http_retry: int
    http_max_attempts: int


class AzureConnection(object):
    def __init__(self, logger):
        self.logger = logger
        self.url = None
        self.subscription_key = None
        self.blob_container_url = None
        self.blob_service_key = None
        self.acs_headers = {}
        self.container_client = None
        self.http_retry = None
        self.http_max_attempts = None

    def update_acs(self, server_info: AcsServerInfo):
        self.url = server_info.url
        self.subscription_key = server_info.subscription_key
        self.acs_headers = {
            'Ocp-Apim-Subscription-Key': server_info.subscription_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        self.http_retry = server_info.http_retry

        self.logger.info('Retrieving valid transcription locales')
        req = request.Request(
            url=self.url + '/locales',
            headers=self.acs_headers,
            method='GET'
        )
        with self.http_retry.urlopen(req) as response:
            self.supported_locales = json.load(response)

        if (self.blob_container_url != server_info.blob_container_url
                or self.blob_service_key != server_info.blob_service_key
                or self.http_max_attempts != server_info.http_max_attempts):
            self.logger.debug('Updating ACS connection')
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
            self.logger.debug('ACS arguments unchanged')

    def get_blob_client(self, recording_id):
        return self.container_client.get_blob_client(recording_id)

    def generate_account_sas(self, time_limit):
        # Shared access signature (SAS) is required for the ACS Speech
        #  service to access the container
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
                self.logger.info('Blob exists for file {}'.format(recording_id))
            else:
                self.logger.error('Uploading file to blob failed for file {}'.format(recording_id))
                raise mpf.DetectionError.NETWORK_ERROR.exception(
                    'Uploading file to blob failed due to: ' + str(e)) from e

        time_limit = timedelta(minutes=blob_access_time)
        return '{url:s}/{recording_id:s}?{sas_url:s}'.format(
            url=self.container_client.url,
            recording_id=recording_id,
            sas_url=self.generate_account_sas(time_limit)
        )

    def submit_batch_transcription(self, recording_url, job_name,
                                   diarize, language, expiry):
        if language not in self.supported_locales:
            raise ValueError(f"Provided language ({language}) not supported."
                             " Refer to component README for list of supported"
                             " locales for Azure Speech.")

        self.logger.info('Submitting batch transcription...')
        data = dict(
            contentUrls=[recording_url],
            displayName=job_name,
            description=job_name,
            locale=language,
            properties=dict(
                wordLevelTimestampsEnabled='true',
                profanityFilterMode='None',
                diarizationEnabled=str(diarize).lower(),
                timeToLive=minutes_to_iso8601(expiry)
            )
        )

        data = json.dumps(data).encode()
        req = request.Request(
            url=self.url,
            data=data,
            headers=self.acs_headers,
            method='POST'
        )
        try:
            with self.http_retry.urlopen(req) as response:
                return response.getheader('Location')
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
        self.logger.info('Polling for transcription success')
        while True:
            with self.http_retry.urlopen(req) as response:
                result = json.load(response)
                status = result['status']
                if status == 'Succeeded':
                    self.logger.debug('Transcription succeeded')
                    break
                elif status == 'Failed':
                    self.logger.debug('Transcription failed')
                    break
                else:
                    retry_after = int(response.info()['Retry-After'])
                    self.logger.info(
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

        self.logger.info(f'Retrieving transcription file location from {results_uri}')
        with self.http_retry.urlopen(req) as response:
            files = json.load(response)

        transcript_uri = next(
            v['links']['contentUrl'] for v in files['values']
            if v['kind'] == 'Transcription'
        )
        self.logger.info(f'Retrieving transcription result from {transcript_uri}')
        with self.http_retry.urlopen(transcript_uri) as response:
            return json.load(response)

    def delete_transcription(self, location):
        self.logger.info('Deleting transcription...')
        req = request.Request(
            url=location,
            headers=self.acs_headers,
            method='DELETE'
        )
        try:
            with self.http_retry.urlopen(req):
                pass
        except mpf.DetectionException:
            # If the transcription task doesn't exist, ignore
            #  This is a temporary solution, to be fixed with v3.0
            self.logger.warning(
                f'Transcription task deletion failed. This transcription '
                f'should be deleted manually: {location}')

    def delete_blob(self, recording_id):
        self.logger.info('Deleting blob...')
        self.container_client.delete_blob(recording_id)
