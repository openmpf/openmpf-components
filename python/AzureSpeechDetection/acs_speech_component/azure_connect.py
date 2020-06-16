#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2019 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2019 The MITRE Corporation                                      #
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

from __future__ import division, print_function

import os
import json
import time
from urllib2 import Request, urlopen, HTTPError
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ResourceTypes, AccountSasPermissions, generate_account_sas
from tempfile import NamedTemporaryFile

import mpf_component_api as mpf
import mpf_component_util as mpf_util


class RequestWithMethod(Request):
  def __init__(self, *args, **kwargs):
    self._method = kwargs.pop('method', None)
    Request.__init__(self, *args, **kwargs)

  def get_method(self):
    return self._method if self._method else super(RequestWithMethod, self).get_method()


class AzureConnection(object):
    def __init__(self, logger):
        self.logger = logger
        self.url = None
        self.account_name = None
        self.subscription_key = None
        self.speech_key = None
        self.container_name = None
        self.endpoint_suffix = None

    def _update_acs(self, url, account_name, subscription_key, speech_key,
                 container_name, endpoint_suffix):
        self.url = url
        self.account_name = account_name
        self.subscription_key = subscription_key
        self.speech_key = speech_key
        self.container_name = container_name
        self.endpoint_suffix = endpoint_suffix

        connect_str = (
            'DefaultEndpointsProtocol=https;'
            'AccountName={:s};'
            'AccountKey={:s};'
            'EndpointSuffix={:s}'
        ).format(
            account_name,
            subscription_key,
            endpoint_suffix
        )
        self.bsc = BlobServiceClient.from_connection_string(connect_str)
        self.container_client = self.bsc.get_container_client(container_name)
        self.expiry = datetime.utcnow()
        self.acs_headers = {
            'Ocp-Apim-Subscription-Key': speech_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json',
        }

    def get_blob_client(self, recording_id):
        return self.container_client.get_blob_client(recording_id)

    def generate_account_sas(self):
        return generate_account_sas(
            self.bsc.account_name,
            account_key=self.bsc.credential.account_key,
            resource_types=ResourceTypes(object=True),
            permission=AccountSasPermissions(read=True),
            expiry=self.expiry
        )

    def upload_file_to_blob(self, filepath, recording_id,
                            start_time=0, stop_time=None):
        try:
            blob_client = self.get_blob_client(recording_id)
            with NamedTemporaryFile(delete=True) as tmp:
                mpf_util.rip_audio(
                    filepath,
                    tmp.name,
                    start_time,
                    stop_time
                )
                tmp.seek(0)
                blob_client.upload_blob(tmp.read())
        except Exception as e:
            if 'blob already exists' in str(e):
                self.logger.info("Blob exists for file {}".format(recording_id))
            else:
                self.logger.error("Uploading file to blob failed for file {}".format(recording_id))
                raise

        if datetime.utcnow() + timedelta(minutes=5) > self.expiry:
            self.expiry = datetime.utcnow() + timedelta(hours=1)
            self.sas_url = self.generate_account_sas()
        return '{url:s}{container:s}/{recording_id:s}?{sas_url:s}'.format(
            url=self.bsc.url,
            container=self.container_name,
            recording_id=recording_id,
            sas_url=self.sas_url
        )

    def submit_batch_transcription(self, recording_url, job_name,
                                   diarize, language):
        self.logger.info("Submitting batch transcription...")
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
        req = RequestWithMethod(
            url=self.url,
            data=data,
            headers=self.acs_headers,
            method='POST'
        )
        try:
            response = urlopen(req)
            return response.info().getheader('Location')
        except HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
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
        req = RequestWithMethod(
            url=location,
            headers=self.acs_headers,
            method='GET'
        )
        self.logger.info("Polling for transcription success")
        while True:
            try:
                response = urlopen(req)
            except HTTPError as e:
                response_content = e.read().decode('utf-8', errors='replace')
                raise mpf.DetectionException(
                    "Polling failed with status {} and message: {}".format(
                        e.code,
                        e.read().decode('utf-8', errors='replace')
                    ),
                    mpf.DetectionError.DETECTION_FAILED
                )
            result = json.load(response)
            status = result['status']
            if status == "Succeeded":
                self.logger.debug("Transcription succeeded")
                break
            elif status == "Failed":
                self.logger.debug("Transcription failed")
                break
            else:
                retry_after = int(response.info()['Retry-After'])
                self.logger.debug(
                    "Status is {}. Retry in {} seconds.".format(
                        status,
                        retry_after
                    )
                )
                time.sleep(retry_after)

        return result

    def get_transcription(self, result):
        results_uri = result['resultsUrls']['channel_0']
        try:
            self.logger.debug("Retrieving transcription result")
            response = urlopen(results_uri)
            return json.load(response)
        except HTTPError as e:
            error_str = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionException(
                "Request failed with HTTP status {} and messages: {}".format(
                    e.code,
                    error_str
                ),
                mpf.DetectionError.DETECTION_FAILED
            )

    def delete_transcription(self, location):
        self.logger.info("Deleting transcription...")
        req = RequestWithMethod(
            url=location,
            headers=self.acs_headers,
            method='DELETE'
        )
        urlopen(req)

    def delete_blob(self, recording_id):
        self.logger.info("Deleting blob...")
        self.container_client.delete_blob(recording_id)

    def get_transcriptions(self):
        self.logger.info("Retrieving all transcriptions...")
        req = RequestWithMethod(
            url=self.url,
            headers=self.acs_headers,
            method='GET'
        )
        response = urlopen(req)
        transcriptions = json.load(response)
        return transcriptions

    def delete_all_transcriptions(self):
        self.logger.info("Deleting all transcriptions...")
        for trans in self.get_transcriptions():
            req = RequestWithMethod(
                url=self.url + '/' + trans['id'],
                headers=self.acs_headers,
                method='DELETE'
            )
            response = urlopen(req)
