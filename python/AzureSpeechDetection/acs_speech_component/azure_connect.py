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
from urllib.error import HTTPError
from datetime import datetime, timedelta
from azure.storage.blob import ResourceTypes, AccountSasPermissions, generate_account_sas, ContainerClient

import mpf_component_api as mpf
import mpf_component_util as mpf_util


class AzureConnection(object):
    def __init__(self, logger):
        self.logger = logger
        self.url = None
        self.subscription_key = None
        self.blob_container_url = None
        self.blob_service_key = None

    def update_acs(self, url, subscription_key, blob_container_url,
                   blob_service_key):
        self.url = url
        self.subscription_key = subscription_key
        self.blob_container_url = blob_container_url
        self.blob_service_key = blob_service_key
        self.acs_headers = {
            'Ocp-Apim-Subscription-Key': subscription_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }
        self.container_client = ContainerClient.from_container_url(
            blob_container_url,
            blob_service_key
        )

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

    def upload_file_to_blob(self, filepath, recording_id, blob_access_time,
                            start_time=0, stop_time=None):
        try:
            blob_client = self.get_blob_client(recording_id)
            audio_bytes = mpf_util.transcode_to_wav(
                filepath,
                start_time,
                stop_time
            )
            blob_client.upload_blob(audio_bytes)
        except Exception as e:
            if 'blob already exists' in str(e):
                self.logger.info('Blob exists for file {}'.format(recording_id))
            else:
                self.logger.error('Uploading file to blob failed for file {}'.format(recording_id))
                raise

        time_limit = timedelta(minutes=blob_access_time)
        return '{url:s}/{recording_id:s}?{sas_url:s}'.format(
            url=self.container_client.url,
            recording_id=recording_id,
            sas_url=self.generate_account_sas(time_limit)
        )

    def submit_batch_transcription(self, recording_url, job_name,
                                   diarize, language):
        self.logger.info('Submitting batch transcription...')
        data = dict(
            recordingsUrl=recording_url,
            name=job_name,
            description=job_name,
            locale=language,
            properties=dict(
                AddWordLevelTimestamps='True',
                AddDiarization=str(diarize),
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
            response = request.urlopen(req)
            return response.getheader('Location')
        except HTTPError as e:
            response_content = e.read()
            raise mpf.DetectionException(
                'Failed to post job. Got HTTP status {} and message: {}'.format(
                    e.code,
                    response_content
                ),
                mpf.DetectionError.DETECTION_FAILED
            )
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
            try:
                response = request.urlopen(req)
            except HTTPError as e:
                raise mpf.DetectionException(
                    'Polling failed with status {} and message: {}'.format(
                        e.code,
                        e.read()
                    ),
                    mpf.DetectionError.DETECTION_FAILED
                )
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
        results_uri = result['resultsUrls']['channel_0']
        try:
            self.logger.debug('Retrieving transcription result')
            response = request.urlopen(results_uri)
            return json.load(response)
        except HTTPError as e:
            error_str = e.read()
            raise mpf.DetectionException(
                'Request failed with HTTP status {} and messages: {}'.format(
                    e.code,
                    error_str
                ),
                mpf.DetectionError.DETECTION_FAILED
            )

    def delete_transcription(self, location):
        self.logger.info('Deleting transcription...')
        req = request.Request(
            url=location,
            headers=self.acs_headers,
            method='DELETE'
        )
        try:
            request.urlopen(req)
        except HTTPError:
            # If the transcription task doesn't exist, ignore
            #  This is a temporary solution, to be fixed with v3.0
            self.logger.warning(
                f'Transcription task deletion failed. This transcription '
                f'should be deleted manually: {location}')

    def delete_blob(self, recording_id):
        self.logger.info('Deleting blob...')
        self.container_client.delete_blob(recording_id)

    def get_transcriptions(self):
        self.logger.info('Retrieving all transcriptions...')
        req = request.Request(
            url=self.url,
            headers=self.acs_headers,
            method='GET'
        )
        response = request.urlopen(req)
        transcriptions = json.load(response)
        return transcriptions

    def delete_all_transcriptions(self):
        self.logger.info('Deleting all transcriptions...')
        for trans in self.get_transcriptions():
            req = request.Request(
                url=self.url + '/' + trans['id'],
                headers=self.acs_headers,
                method='DELETE'
            )
            request.urlopen(req)
