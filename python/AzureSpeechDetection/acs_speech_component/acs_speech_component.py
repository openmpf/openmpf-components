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
import logging
from math import floor, ceil

import mpf_component_api as mpf
import mpf_component_util as mpf_util
from .acs_speech_processor import AcsSpeechDetectionProcessor


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

    @staticmethod
    def _get_job_property_or_env_value(property_name, job_properties):
        property_value = job_properties.get(property_name)
        if property_value:
            return property_value

        env_value = os.getenv(property_name)
        if env_value:
            return env_value

        raise mpf.DetectionException(
            'The "{}" property must be provided as a job property or environment variable.'.format(property_name),
            mpf.DetectionError.MISSING_PROPERTY)

    @classmethod
    def _parse_properties(cls, job_properties):
        """
        :param job_properties: Properties object from AudioJob or VideoJob
        :return: Dictionary of properties, pass as **kwargs to process_audio
        """

        return dict(
            acs_url=cls._get_job_property_or_env_value(
                'ACS_URL',
                job_properties
            ),
            acs_subscription_key=cls._get_job_property_or_env_value(
                'ACS_SUBSCRIPTION_KEY',
                job_properties
            ),
            acs_blob_container_url=cls._get_job_property_or_env_value(
                'ACS_BLOB_CONTAINER_URL',
                job_properties
            ),
            acs_blob_service_key=cls._get_job_property_or_env_value(
                'ACS_BLOB_SERVICE_KEY',
                job_properties
            ),
            lang=job_properties.get('LANGUAGE', 'en-US'),
            diarize=mpf_util.get_property(job_properties, 'DIARIZE', True),
            cleanup=mpf_util.get_property(job_properties, 'CLEANUP', True),
            blob_access_time=int(job_properties.get('BLOB_ACCESS_TIME', '120'))
        )

    def get_detections_from_audio(self, audio_job):
        logger.extra['job_name'] = audio_job.job_name
        logger.info('Received audio job')

        start_time = audio_job.start_time
        stop_time = audio_job.stop_time
        if stop_time < 0:
            stop_time = None
        try:
            parsed_properties = self._parse_properties(audio_job.job_properties)
        except Exception as e:
            # Pass the exception up
            logger.exception(
                'Exception raised while parsing properties: ' + str(e)
            )
            raise

        # If we suspect this is a subjob, overwrite SPEAKER_ID with 0 to
        #  avoid confusion
        overwrite_ids = False
        if audio_job.feed_forward_track is not None:
            overwrite_ids = True
        elif stop_time is None:
            pass
        elif start_time > 0:
            overwrite_ids = True
        elif 'DURATION' in audio_job.media_properties:
            if stop_time < float(audio_job.media_properties['DURATION']):
                overwrite_ids = True

        try:
            audio_tracks = self.processor.process_audio(
                target_file=audio_job.data_uri,
                start_time=start_time,
                stop_time=stop_time,
                job_name=audio_job.job_name,
                **parsed_properties
            )
        except Exception as e:
            # Pass the exception up
            logger.exception(
                'Exception raised while processing audio: ' + str(e)
            )
            raise

        for track in audio_tracks:
            track.detection_properties['LONG_SPEAKER_ID'] = '{}-{}-{}'.format(
                audio_job.start_time,
                audio_job.stop_time if audio_job.stop_time > 0 else 'EOF',
                track.detection_properties['SPEAKER_ID']
            )
            if overwrite_ids:
                track.detection_properties['SPEAKER_ID'] = '0'

        logger.info('Processing complete. Found %d tracks.' % len(audio_tracks))
        return audio_tracks

    def get_detections_from_video(self, video_job):
        logger.extra['job_name'] = video_job.job_name
        logger.info('Received video job')

        start_frame = video_job.start_frame
        stop_frame = video_job.stop_frame
        try:
            parsed_properties = self._parse_properties(video_job.job_properties)
        except Exception as e:
            # Pass the exception up
            logger.exception(
                'Exception raised while parsing properties: ' + str(e)
            )
            raise

        if 'FPS' not in video_job.media_properties:
            error_str = 'FPS must be included in video job media properties.'
            logger.error(error_str)
            raise mpf.DetectionException(
                error_str,
                mpf.DetectionError.MISSING_PROPERTY
            )

        # Convert frame locations to timestamps
        fpms = float(video_job.media_properties['FPS']) / 1000.0
        start_time = start_frame / fpms
        stop_time = stop_frame / fpms
        if stop_time < 0:
            stop_time = None

        # If we suspect this is a subjob, overwrite SPEAKER_ID with 0 to
        #  avoid confusion
        overwrite_ids = False
        if video_job.feed_forward_track is not None:
            overwrite_ids = True
        elif stop_frame is None:
            pass
        elif start_frame > 0:
            overwrite_ids = True
        elif 'FRAME_COUNT' in video_job.media_properties:
            if stop_frame < int(video_job.media_properties['FRAME_COUNT']) - 1:
                overwrite_ids = True

        try:
            audio_tracks = self.processor.process_audio(
                target_file=video_job.data_uri,
                start_time=start_time,
                stop_time=stop_time,
                job_name=video_job.job_name,
                **parsed_properties
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
                track.detection_properties['LONG_SPEAKER_ID'] = '{}-{}-{}'.format(
                    video_job.start_frame,
                    video_job.stop_frame if video_job.stop_frame > 0 else 'EOF',
                    track.detection_properties['SPEAKER_ID']
                )
                if overwrite_ids:
                    track.detection_properties['SPEAKER_ID'] = '0'
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
            # Pass the exception up
            logger.exception(
                'Exception raised while converting to video track: ' + str(e)
            )
            raise

        logger.info('Processing complete. Found %d tracks.' % len(video_tracks))
        return video_tracks
