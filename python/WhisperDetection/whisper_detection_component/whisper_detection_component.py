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

import whisper

import logging

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from typing import Sequence

logger = logging.getLogger('WhisperDetectionComponent')


class WhisperDetectionComponent:
    detection_type = 'SPEECH'

    def __init__(self):
        logger.info('Creating instance of WhisperDetectionComponent')
        self.wrapper = WhisperDetectionWrapper()
        logger.info('WhisperDetectionComponent created')

    def get_detections_from_video(self, job: mpf.VideoJob) -> Sequence[mpf.VideoTrack]:
        logger.info('Received video job')

        start_frame = job.start_frame
        stop_frame = job.stop_frame

        media_properties = job.media_properties
        media_frame_count = int(media_properties.get('FRAME_COUNT', -1))
        media_duration = float(media_properties.get('DURATION', -1))

        fpms = float(job.media_properties['FPS']) / 1000.0
        start_time = int(start_frame / fpms)

        if stop_frame is not None and stop_frame < media_frame_count - 1:
            stop_time = int(stop_frame / fpms)
        elif media_duration > 0:
            stop_time = int(media_duration)
        elif media_frame_count > 0:
            stop_time = int(media_frame_count / fpms)
        else:
            stop_time = None

        try:
            audio_tracks = self.wrapper.process_audio(
                target_file=job.data_uri,
                start_time=start_time,
                stop_time=stop_time
            )
        except Exception as e:
            # Pass the exception up
            error_str = 'Exception raised while processing audio: ' + str(e)
            logger.exception(error_str)
            raise

        try:
            # Convert audio tracks to video tracks with placeholder frame locs
            video_tracks = []
            for track in audio_tracks:
                # Convert timestamps back to frames
                track_start_frame = int(floor(fpms * track.start_time))
                track_stop_frame = int(ceil(fpms * track.stop_time))

                video_track = mpf.VideoTrack(
                    start_frame=track_start_frame,
                    stop_frame=track_stop_frame,
                    confidence=track.confidence,
                    detection_properties=track.detection_properties
                )

                loc = mpf.ImageLocation(
                    x_left_upper=0,
                    y_left_upper=0,
                    width=0,
                    height=0,
                    confidence=track.confidence,
                    detection_properties=track.detection_properties
                )

                video_track.frame_locations[track_start_frame] = loc
                video_tracks.append(video_track)

        except Exception as e:
            # Pass the exception up
            logger.exception(
                'Exception raised while converting to video track: ' + str(e)
            )
            raise

        logger.info('Processing complete. Found %d tracks.' % len(video_tracks))
        return video_tracks

    def get_detections_from_audio(self, job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        logger.info("Received audio job")
        start_time = job.start_time
        stop_time = job.stop_time

        try:
            audio_tracks = self.wrapper.process_audio(
                target_file=job.data_uri,
                start_time=start_time,
                stop_time=stop_time,
            )
        except Exception as e:
            # Pass the exception up
            logger.exception(
                "Exception raised while processing audio: " + str(e)
            )
            raise

        logger.info('Processing complete. Found %d tracks.' % len(audio_tracks))
        return audio_tracks


class WhisperDetectionWrapper:
    def __init__(self):
        self.model = whisper.load_model("base")

    def process_audio(self, target_file, start_time, stop_time):
        audio = whisper.load_audio(target_file)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        _, probs = self.model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")

        detected_language = max(probs, key=probs.get)
        detected_lang_conf = probs[detected_language]

        audio_tracks = []

        properties = dict(
            DETECTED_LANGUAGE=detected_language,
            DETECTED_LANGUAGE_CONFIDENCE=detected_lang_conf
        )

        track = mpf.AudioTrack(
            start_time=start_time,
            stop_time=stop_time,
            confidence=detected_lang_conf,
            detection_properties=properties
        )
        audio_tracks.append(track)

        logger.debug('Completed process audio')
        return audio_tracks









