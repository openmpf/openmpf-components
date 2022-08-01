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

import os
import logging
from math import floor, ceil
from collections import defaultdict
from typing import Union, Optional, Any, Mapping, Dict, List

import mpf_component_api as mpf
import mpf_component_util as mpf_util
from .acs_speech_processor import AcsSpeechDetectionProcessor, SpeakerInfo
from .azure_connect import AcsServerInfo
from .azure_utils import ISO6393_2_BCP47


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
    def _get_job_property_or_env_value(
                property_name: str,
                job_properties: Mapping[str, str]
            ) -> str:
        property_value = job_properties.get(property_name)
        if property_value:
            return property_value

        env_value = os.getenv(property_name)
        if env_value:
            return env_value

        raise mpf.DetectionException(
            f'The "{property_name}" property must be provided as a '
            'job property or environment variable.',
            mpf.DetectionError.MISSING_PROPERTY)

    @classmethod
    def _parse_job_props(
                cls,
                job_properties: Mapping[str, str]
            ) -> Dict[str, Any]:
        """
        :param job_properties: job_properties from AudioJob or VideoJob
        :return: Dictionary of properties, pass as **kwargs to process_audio
        """

        acs_url = cls._get_job_property_or_env_value(
            'ACS_URL',
            job_properties
        )

        acs_subscription_key = cls._get_job_property_or_env_value(
            'ACS_SUBSCRIPTION_KEY',
            job_properties
        )

        acs_blob_container_url = cls._get_job_property_or_env_value(
            'ACS_BLOB_CONTAINER_URL',
            job_properties
        )

        acs_blob_service_key = cls._get_job_property_or_env_value(
            'ACS_BLOB_SERVICE_KEY',
            job_properties
        )

        http_retry = mpf_util.HttpRetry.from_properties(
            job_properties,
            logger.warning
        )

        http_max_attempts = mpf_util.get_property(
            properties=job_properties,
            key='COMPONENT_HTTP_RETRY_MAX_ATTEMPTS',
            default_value=10,
            prop_type=int
        )

        server_info = AcsServerInfo(
            url=acs_url,
            subscription_key=acs_subscription_key,
            blob_container_url=acs_blob_container_url,
            blob_service_key=acs_blob_service_key,
            http_retry=http_retry,
            http_max_attempts=http_max_attempts
        )

        blob_access_time = mpf_util.get_property(
            properties=job_properties,
            key='BLOB_ACCESS_TIME',
            default_value=120,
            prop_type=int
        )

        expiry = mpf_util.get_property(
            properties=job_properties,
            key='TRANSCRIPTION_EXPIRATION',
            default_value=120,
            prop_type=int
        )

        language = mpf_util.get_property(
            properties=job_properties,
            key='LANGUAGE',
            default_value='en-US',
            prop_type=str
        )

        diarize = mpf_util.get_property(
            properties=job_properties,
            key='DIARIZE',
            default_value=True,
            prop_type=bool
        )

        cleanup = mpf_util.get_property(
            properties=job_properties,
            key='CLEANUP',
            default_value=True,
            prop_type=bool
        )

        return dict(
            server_info=server_info,
            blob_access_time=blob_access_time,
            expiry=expiry,
            language=language,
            diarize=diarize,
            cleanup=cleanup,
        )

    @staticmethod
    def _parse_ff_props(
                ff_props: Mapping[str, str],
                start_time: int,
                stop_time: int
            ) -> SpeakerInfo:
        """
        :param ff_props: detection_properties from job's feed_forward_track
        :param start_time: Job start time (in milliseconds)
        :param stop_time: Job stop time (in milliseconds)
        :return: SpeakerInfo object holding speaker information from upstream
            component, including ID and language.
        """
        speaker_id = mpf_util.get_property(
            properties=ff_props,
            key='SPEAKER_ID',
            default_value='0',
            prop_type=str
        )
        # If speaker ID was overwritten, use long speaker ID
        if speaker_id == '0':
            speaker_id = mpf_util.get_property(
                properties=ff_props,
                key='LONG_SPEAKER_ID',
                default_value='0',
                prop_type=str
            )

        gender = mpf_util.get_property(
            properties=ff_props,
            key='GENDER',
            default_value='',
            prop_type=str
        )

        gender_score = mpf_util.get_property(
            properties=ff_props,
            key='GENDER_CONFIDENCE',
            default_value=-1.0,
            prop_type=float
        )

        languages = mpf_util.get_property(
            properties=ff_props,
            key='SPEAKER_LANGUAGES',
            default_value='',
            prop_type=str
        )

        language_scores = mpf_util.get_property(
            properties=ff_props,
            key='SPEAKER_LANGUAGE_CONFIDENCES',
            default_value='',
            prop_type=str
        )

        languages = [lab.strip().lower() for lab in languages.split(',')]
        language_scores = [float(s.strip()) for s in language_scores.split(',')]
        language_scores = defaultdict(
            lambda: -1.0,
            zip(languages, language_scores)
        )

        segments_str = mpf_util.get_property(
            properties=ff_props,
            key='VOICED_SEGMENTS',
            default_value='',
            prop_type=str
        )
        speech_segs = AcsSpeechDetectionProcessor.parse_segments_str(
            segments_string=segments_str,
            media_start_time=start_time,
            media_stop_time=stop_time
        )

        language_iso = mpf_util.get_property(
            properties=ff_props,
            key='LANGUAGE',
            default_value='',
            prop_type=str
        )
        language_iso = language_iso.strip().upper()
        languages = ISO6393_2_BCP47.get(language_iso, None)
        if languages is None:
            raise mpf.DetectionException(
                f"ISO 639-3 code '{language_iso}' provided in feed-forward track"
                f" does not correspond to a BCP-47 language code supported by "
                f" Azure Speech-to-Text.",
                mpf.DetectionError.INVALID_PROPERTY
            )

        return SpeakerInfo(
            speaker_id=speaker_id,
            language=languages[0],
            language_scores=language_scores,
            gender=gender,
            gender_score=gender_score,
            speech_segs=speech_segs
        )

    @classmethod
    def parse_properties(
                cls,
                job: Union[mpf.AudioJob, mpf.VideoJob]
            ) -> Optional[Dict[str, Any]]:
        """
        :param job: AudioJob or VideoJob
        :return: Dictionary of properties, pass as **kwargs to process_audio.
            If None, then skip this job and return feed-forward track.
        """
        trigger = mpf_util.get_property(
            properties=job.job_properties,
            key='TRIGGER',
            default_value='',
            prop_type=str
        )
        t_key, t_val = trigger.strip().split('=') if trigger else (None, None)

        job_props = cls._parse_job_props(job.job_properties)

        if isinstance(job, mpf.VideoJob):
            start_frame = job.start_frame
            stop_frame = job.stop_frame
            if stop_frame < 0:
                stop_frame = None

            # Convert frame locations to timestamps
            media_frame_count = int(job.media_properties.get('FRAME_COUNT', -1))
            media_duration = float(job.media_properties.get('DURATION', -1))
            fpms = float(job.media_properties['FPS']) / 1000.0
            start_time = int(start_frame / fpms)

            # The WFM will pass a job stop frame equal to FRAME_COUNT-1 for the
            #  last video segment. We want to use the detected DURATION in such
            #  cases instead to ensure we process the entire audio track.
            #  Only use the job stop frame if it differs from FRAME_COUNT-1.
            if stop_frame is not None and stop_frame < media_frame_count - 1:
                stop_time = int(stop_frame / fpms)
            elif media_duration > 0:
                stop_time = int(media_duration)
            elif media_frame_count > 0:
                stop_time = int(media_frame_count / fpms)
            else:
                stop_time = None
        else:
            start_time = job.start_time
            stop_time = job.stop_time
            if stop_time < 0:
                stop_time = None

        # Parse feed-forward properties if a feed-forward track exists
        if job.feed_forward_track is not None:
            det_props = job.feed_forward_track.detection_properties

            # If trigger key is in feed-forward properties, only run
            #  transcription if the value matches trigger_val
            if not (t_key in det_props and det_props[t_key] == t_val):
                return None

            logger.debug("Getting properties from feed-forward track")
            speaker = cls._parse_ff_props(
                det_props,
                start_time=start_time,
                stop_time=stop_time
            )
            job_props.update(
                language=speaker.language,
                diarize=False,
                speaker=speaker
            )

        job_props.update(
            target_file=job.data_uri,
            job_name=job.job_name,
            start_time=start_time,
            stop_time=stop_time
        )

        return job_props

    def get_detections_from_job(
                self,
                job: Union[mpf.AudioJob, mpf.VideoJob]
            ) -> Optional[List[mpf.AudioTrack]]:
        try:
            job_props = self.parse_properties(job)
        except Exception as e:
            logger.exception(f'Exception raised while parsing properties: {e}')
            raise

        # If parse_properties returned None, this component should be skipped
        if job_props is None:
            return None

        try:
            logger.debug("Getting transcription tracks")
            audio_tracks = self.processor.process_audio(**job_props)
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
            elif 'DURATION' in job.media_properties:
                if t1 < float(job.media_properties['DURATION']):
                    overwrite_ids = True
            elif 'FRAME_COUNT' in job.media_properties:
                if t1 < int(job.media_properties['FRAME_COUNT']) - 1:
                    overwrite_ids = True
        else:
            t1 = 'EOF'

        for track in audio_tracks:
            sid = track.detection_properties['SPEAKER_ID']
            track.detection_properties['LONG_SPEAKER_ID'] = f"{t0}-{t1}-{sid}"
            if overwrite_ids:
                track.detection_properties['SPEAKER_ID'] = '0'

        return audio_tracks

    def get_detections_from_audio(
                self,
                job: mpf.AudioJob
            ) -> List[mpf.AudioTrack]:
        logger.extra['job_name'] = job.job_name
        logger.info('Received audio job')

        tracks = self.get_detections_from_job(job)
        if tracks is not None:
            logger.info('Processing complete. Found %d tracks.' % len(tracks))
            return tracks

        # If parse_properties returned None, this component should be skipped
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

        audio_tracks = self.get_detections_from_job(job)

        # If parse_properties returned None, this component should be skipped
        if audio_tracks is None:
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
            logger.exception('Exception raised while converting to video track: {e}')
            raise

        logger.info('Processing complete. Found %d tracks.' % len(video_tracks))
        return video_tracks
