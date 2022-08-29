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
from math import floor, ceil
from typing import Union, List

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from acs_speech_component.acs_speech_processor import AcsSpeechDetectionProcessor
from acs_speech_component.job_parsing import AzureJobConfig


class MPFJobNameLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'job_name' in kwargs:
            job_name = kwargs.pop('job_name')
        elif self.extra is not None and 'job_name' in self.extra:
            job_name = self.extra['job_name']
        else:
            return msg, kwargs
        return '[%s] %s' % (job_name, msg), kwargs


logger = MPFJobNameLoggerAdapter(
    logging.getLogger('AcsSpeechComponent'),
    extra={}
)

logging.getLogger('azure').setLevel('WARN')


class AcsSpeechComponent(object):
    detection_type = 'SPEECH'

    def __init__(self):
        logger.extra = {}
        logger.info('Creating instance of AcsSpeechDetectionProcessor')
        self.processor = AcsSpeechDetectionProcessor(logger)
        logger.info('AcsSpeechDetection created')

    def get_detections_from_job(
                self,
                job: Union[mpf.AudioJob, mpf.VideoJob]
            ) -> List[mpf.AudioTrack]:
        try:
            job_config = AzureJobConfig(job)
        except mpf_util.TriggerMismatch as e:
            logger.info(f"Feed-forward track does not meet trigger condition: {e}")
            raise
        except mpf_util.NoInBoundsSpeechSegments as e:
            logger.warning(f"Feed-forward track does not contain in-bounds segments: {e}")
            raise
        except Exception as e:
            logger.exception(f'Exception raised while parsing properties: {e}')
            raise

        try:
            logger.debug("Getting transcription tracks")
            audio_tracks = self.processor.process_audio(job_config)
        except Exception as e:
            logger.exception(f'Exception raised while processing audio: {e}')
            raise

        if isinstance(job, mpf.VideoJob):
            t0, t1 = job.start_frame, job.stop_frame
        elif isinstance(job, mpf.AudioJob):
            t0, t1 = job.start_time, job.stop_time
        else:
            raise mpf.DetectionException(
                "Can only process AudioJobs and VideoJobs",
                mpf.DetectionError.UNSUPPORTED_DATA_TYPE
            )

        # If we suspect this is a subjob, overwrite SPEAKER_ID with 0 to
        #  avoid confusion
        overwrite_ids = job.feed_forward_track is not None
        if t1 > 0:
            if t0 > 0:
                overwrite_ids = True
            elif 'FRAME_COUNT' in job.media_properties:
                if t1 < int(job.media_properties['FRAME_COUNT']) - 1:
                    overwrite_ids = True
            elif 'DURATION' in job.media_properties:
                if t1 < float(job.media_properties['DURATION']):
                    overwrite_ids = True
        else:
            t1 = 'EOF'

        for track in audio_tracks:
            sid = track.detection_properties['SPEAKER_ID']
            track.detection_properties['LONG_SPEAKER_ID'] = f"{t0}-{t1}-{sid}"
            if overwrite_ids:
                track.detection_properties['SPEAKER_ID'] = '0'

        logger.info('Processing complete. Found %d tracks.' % len(audio_tracks))
        return audio_tracks

    def get_detections_from_audio(self, job: mpf.AudioJob) -> List[mpf.AudioTrack]:
        logger.extra['job_name'] = job.job_name
        logger.info('Received audio job')

        try:
            return self.get_detections_from_job(job)
        except mpf_util.TriggerMismatch:
            return [job.feed_forward_track]

    def get_detections_from_video(
                self,
                job: mpf.VideoJob
            ) -> List[mpf.VideoTrack]:
        logger.extra['job_name'] = job.job_name
        logger.info('Received video job')

        if 'FPS' not in job.media_properties:
            error_str = 'FPS must be included in video job media properties.'
            logger.error(error_str)
            raise mpf.DetectionException(
                error_str,
                mpf.DetectionError.MISSING_PROPERTY
            )
        fpms = float(job.media_properties['FPS']) / 1000.0

        try:
            audio_tracks = self.get_detections_from_job(job)
        except mpf_util.TriggerMismatch:
            return [job.feed_forward_track]

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

                # Placeholder frame location, contains full detection
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
            logger.exception(f'Exception raised while converting to video track: {e}')
            raise

        return video_tracks
