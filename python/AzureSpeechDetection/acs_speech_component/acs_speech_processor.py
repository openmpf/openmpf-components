#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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

from .azure_connect import AzureConnection

class AcsSpeechDetectionProcessor(object):

    def __init__(self, logger):
        self.logger = logger
        self.acs = AzureConnection(logger)


    def update_acs(self, acs_url, acs_subscription_key,
                   acs_blob_container_url, acs_blob_service_key):
        if (self.acs.url != acs_url or
                self.acs.subscription_key != acs_subscription_key or
                self.acs.blob_container_url != acs_blob_container_url or
                self.acs.blob_service_key != acs_blob_service_key):
            self.logger.debug('Updating ACS connection')
            self.acs.update_acs(
                url=acs_url,
                subscription_key=acs_subscription_key,
                blob_container_url=acs_blob_container_url,
                blob_service_key=acs_blob_service_key
            )
        else:
            self.logger.debug('ACS arguments unchanged')


    def process_audio(self, target_file, start_time, stop_time, job_name,
                      acs_url, acs_subscription_key, acs_blob_container_url,
                      acs_blob_service_key, lang, diarize, cleanup,
                      blob_access_time):
        self.update_acs(
            acs_url=acs_url,
            acs_subscription_key=acs_subscription_key,
            acs_blob_container_url=acs_blob_container_url,
            acs_blob_service_key=acs_blob_service_key
        )

        try:
            self.logger.info('Uploading file to blob')
            recording_id = os.path.split(target_file)[-1]
            recording_url = self.acs.upload_file_to_blob(
                filepath=target_file,
                recording_id=recording_id,
                blob_access_time=blob_access_time,
                start_time=start_time,
                stop_time=stop_time
            )
        except Exception as e:
            raise mpf.DetectionException(
                'Failed to upload file to blob: {}'.format(e),
                mpf.DetectionError.DETECTION_FAILED
            )

        try:
            self.logger.info('Submitting speech-to-text job to ACS')
            output_loc = self.acs.submit_batch_transcription(
                recording_url=recording_url,
                job_name=job_name,
                diarize=diarize,
                language=lang,
            )
        except:
            if cleanup:
                self.logger.info('Marking file blob for deletion')
                self.acs.delete_blob(recording_id)
            raise

        try:
            self.logger.info('Retrieving transcription')
            result = self.acs.poll_for_result(output_loc)

            if result['status'] == 'Failed':
                raise mpf.DetectionException(
                    'Transcription failed: {}'.format(result['statusMessage']),
                    mpf.DetectionError.DETECTION_FAILED
                )

            transcription = self.acs.get_transcription(result)
            self.logger.info('Speech-to-text processing complete')
        finally:
            if cleanup:
                self.logger.info('Marking file blob for deletion')
                self.acs.delete_blob(recording_id)
            self.logger.info('Deleting transcript')
            self.acs.delete_transcription(output_loc)

        self.logger.info('Completed process audio')
        self.logger.info('Creating AudioTracks')
        audio_tracks = []
        for utt in transcription['AudioFileResults'][0]['SegmentResults']:
            speaker_id = utt['SpeakerId'] if diarize else '0'
            display = utt['NBest'][0]['Display']

            # Confidence information. Utterance confidence does not seem
            #  to be a simple aggregate of word confidences.
            utterance_confidence = utt['NBest'][0]['Confidence']
            word_confidences = ', '.join([
                str(w['Confidence'])
                for w in utt['NBest'][0]['Words']
            ])

            # Timing information. Azure works in 100-nanosecond units,
            #  so divide by 1e4 to obtain milliseconds.
            utterance_start = utt['Offset'] / 10000.0
            utterance_stop = (utt['Offset'] + utt['Duration']) / 10000.0
            word_segments = ', '.join([
                str(w['Offset'] / 10000.0) + '-' + str((w['Offset']+w['Duration']) / 10000.0)
                for w in utt['NBest'][0]['Words']
            ])

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

        self.logger.info('Completed processing transcription results')
        return audio_tracks
