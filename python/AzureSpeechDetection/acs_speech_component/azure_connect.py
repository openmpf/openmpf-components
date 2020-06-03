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

import os
import json
import time
from urllib import request, parse
from urllib.error import HTTPError
from datetime import datetime, timedelta
from azure.storage.blob import BlobServiceClient, ResourceTypes, AccountSasPermissions, generate_account_sas

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

    def upload_file_to_blob(self, filepath):
        filename = os.path.split(filepath)[-1]
        try:
            blob_client = self.container_client.get_blob_client(filename)
            with open(filepath, "rb") as fin:
                result = blob_client.upload_blob(fin)
        except Exception as e:
            if 'blob already exists' in str(e):
                self.logger.info(f"Blob exists for file {filename}")
            else:
                self.logger.error(f"Uploading file to blob failed for file {filename}")
                raise

        if datetime.utcnow() + timedelta(minutes=5) > self.expiry:
            self.expiry = datetime.utcnow() + timedelta(hours=1)
            self.sas_url = generate_account_sas(
                self.bsc.account_name,
                account_key=self.bsc.credential.account_key,
                resource_types=ResourceTypes(object=True),
                permission=AccountSasPermissions(read=True),
                expiry=self.expiry
            )
        return filename, '{url:s}{container:s}/{filename:s}?{sas_url:s}'.format(
            url=self.bsc.url,
            container=self.container_name,
            filename=filename,
            sas_url=self.sas_url
        )

    def submit_batch_transcription(self, recordings_url, filename, language):
        self.logger.info("Submitting batch transcription...")
        data = dict(
            recordingsUrl=recordings_url,
            name=filename,
            description=filename,
            locale=language,
            properties=dict(
                # PunctuationMode='DictatedAndAutomatic',
                AddWordLevelTimestamps='True',
                AddDiarization='True',
                # TranscriptionResultsContainerUrl="<results container>"
            )
        )

        data = json.dumps(data).encode()
        req = request.Request(
            url=self.url,
            data=data,
            headers=self.acs_headers,
        )
        try:
            response = request.urlopen(req)
            return response.getheader('Location')
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
                'Failed to post job. Error message: {:s}'.format(e),
                mpf.DetectionError.DETECTION_FAILED
            )


    def get_batch_transcription(self, location):
        self.logger.info("Retrieving batch transcription...")
        req = request.Request(
            url=location,
            headers=self.acs_headers,
            method='GET'
        )
        result = None
        while True:
            try:
                response = request.urlopen(req)
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
                raise mpf.DetectionException(
                    f"Transcription failed: {result['status_message']}",
                    mpf.DetectionError.DETECTION_FAILED
                )
            else:
                retry_after = int(response.info()['Retry-After'])
                self.logger.debug(
                    "Status is {}. Retry in {} seconds.".format(
                        status,
                        retry_after
                    )
                )
                time.sleep(retry_after)

        results_uri = result['resultsUrls']['channel_0']
        try:
            self.logger.debug("Retrieving transcription result")
            response = request.urlopen(results_uri)
            return json.load(response)['AudioFileResults'][0]['SegmentResults']
        except HTTPError as e:
            error_str = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionException(
                f"Request failed with HTTP status {e.code} and messages: {error_str}",
                mpf.DetectionError.DETECTION_FAILED
            )

    def delete_transcription(self, location):
        self.logger.info("Deleting transcription...")
        req = request.Request(
            url=location,
            headers=self.acs_headers,
            method='DELETE'
        )
        request.urlopen(req)

    def get_transcriptions(self):
        self.logger.info("Retrieving all transcriptions...")
        req = request.Request(
            url=self.url,
            headers=self.acs_headers,
            method='GET'
        )
        response = request.urlopen(req)#, cafile='/etc/pki/ca-trust/source/anchors/mitre_ba_root.pem')
        transcriptions = json.load(response)
        return transcriptions

    def delete_all_transcriptions(self):
        self.logger.info("Deleting all transcriptions...")
        for trans in self.get_transcriptions():
            req = request.Request(
                url=self.url + '/' + trans['id'],
                headers=self.acs_headers,
                method='DELETE'
            )
            response = request.urlopen(req)


    def run_transcription(self, filepath, language):
        recordings_url = self.upload_file_to_blob(filepath)
        location = self.submit_batch_transcription(
            recordings_url=recordings_url,
            filename=os.path.split(filepath)[-1],
            language=language,
        )
        return self.get_batch_transcription(location)
