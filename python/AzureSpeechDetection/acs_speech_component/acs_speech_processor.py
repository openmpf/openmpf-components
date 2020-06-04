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

import mpf_component_api as mpf
import mpf_component_util as util

from .azure_connect import AzureConnection

class AcsSpeechDetectionProcessor(object):

    def __init__(self, logger):
        self.logger = logger
        self.acs = AzureConnection(logger)

    def _update_acs(self, acs_url, acs_account_name, acs_subscription_key,
                    acs_speech_key, acs_container_name, acs_endpoint_suffix):
        if (self.acs.url != acs_url or
                self.acs.account_name != acs_account_name or
                self.acs.subscription_key != acs_subscription_key or
                self.acs.speech_key != acs_speech_key or
                self.acs.container_name != acs_container_name or
                self.acs.endpoint_suffix != acs_endpoint_suffix):
            self.logger.debug('Updating ACS connection')
            self.acs._update_acs(
                url=acs_url,
                account_name=acs_account_name,
                subscription_key=acs_subscription_key,
                speech_key=acs_speech_key,
                container_name=acs_container_name,
                endpoint_suffix=acs_endpoint_suffix
            )
        else:
            self.logger.debug('ACS arguments unchanged')


    def process_audio(self, target_file, start_time, stop_time, acs_url,
                      acs_account_name, acs_subscription_key, acs_speech_key,
                      acs_container_name, acs_endpoint_suffix, lang, cleanup):
        self._update_acs(
            acs_url=acs_url,
            acs_account_name=acs_account_name,
            acs_subscription_key=acs_subscription_key,
            acs_speech_key=acs_speech_key,
            acs_container_name=acs_container_name,
            acs_endpoint_suffix=acs_endpoint_suffix
        )

        try:
            self.logger.debug('Uploading file to blob')
            recordings_id, recordings_url = self.acs.upload_file_to_blob(target_file)
        except Exception as e:
            raise mpf.DetectionException(
                "Failed to upload file to blob: {:s}".format(e),
                mpf.DetectionError.DETECTION_FAILED
            )

        try:
            self.logger.debug('Submitting speech-to-text job to ACS')
            output_loc = self.acs.submit_batch_transcription(
                recordings_url=recordings_url,
                filename=os.path.split(target_file)[-1],
                language=lang,
            )
        except:
            if cleanup:
                self.logger.debug('Marking file blob for deletion')
                self.acs.container_client.delete_blob(recordings_id)
            raise

        try:
            self.logger.debug("Retrieving transcription")
            transcription_result = self.acs.get_batch_transcription(output_loc)
            self.logger.debug('Speech-to-text processing complete')
        finally:
            if cleanup:
                self.logger.debug('Marking file blob for deletion')
                self.acs.container_client.delete_blob(recordings_id)
                self.logger.debug('Deleting transcript')
                self.acs.delete_transcription(output_loc)

        self.logger.debug('Creating AudioTracks')
        audio_tracks = []
        for utt in transcription_result:
            speaker_id = utt['SpeakerId']
            utterance_confidence = utt['NBest'][0]['Confidence']
            utterance_start = utt['Offset'] / 10000.0
            utterance_stop = (utt['Offset'] + utt['Duration']) / 10000.0
            display = utt['NBest'][0]['Display']

            word_confidences = ', '.join([
                str(w['Confidence'])
                for w in utt['NBest'][0]['Words']
            ])
            word_segments = ', '.join([
                str(w['Offset'] / 10000.0) + '-' + str((w['Offset']+w['Duration']) / 10000.0)
                for w in utt['NBest'][0]['Words']
            ])
            words = [w['Word'] for w in utt['NBest'][0]['Words']]
            properties = dict(
                SPEAKER_ID=speaker_id,
                TRANSCRIPT=display,
                WORD_CONFIDENCES=word_confidences,
                WORD_SEGMENTS=word_segments,
                DECODED_LANGUAGE=lang,
            )
            track = mpf.AudioTrack(
                start_time=utterance_start,
                stop_time=utterance_stop,
                confidence=utterance_confidence,
                detection_properties=properties
            )
            audio_tracks.append(track)

        self.logger.debug('Completed process audio')
        return audio_tracks
