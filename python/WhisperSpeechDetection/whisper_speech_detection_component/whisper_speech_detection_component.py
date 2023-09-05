#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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
import warnings

import mpf_component_api as mpf
import mpf_component_util as mpf_util
from typing import Sequence

logger = logging.getLogger('WhisperDetectionComponent')

warnings.filterwarnings('ignore', 'FP16 is not supported on CPU; using FP32 instead', UserWarning)
warnings.filterwarnings('ignore', category=ResourceWarning, module='multilingual.tiktoken')

class WhisperSpeechDetectionComponent:
    detection_type = 'SPEECH'

    def __init__(self):
        logger.info('Creating instance of WhisperSpeechDetectionComponent')
        self.wrapper = WhisperSpeechDetectionWrapper()
        logger.info('WhisperSpeechDetectionComponent created')

    def get_detections_from_video(self, job: mpf.VideoJob) -> Sequence[mpf.VideoTrack]:
        logger.info('Received video job')

        start_frame = job.start_frame
        stop_frame = job.stop_frame

        media_properties = job.media_properties
        media_frame_count = int(media_properties.get('FRAME_COUNT', -1))
        media_duration = float(media_properties.get('DURATION', -1))

        fpms = float(job.media_properties['FPS']) / 1000.0
        start_time = int(start_frame / fpms)

        if stop_frame > 0 and stop_frame < media_frame_count - 1:
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
                stop_time=stop_time,
                job_properties=job.job_properties
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

                video_track = mpf.VideoTrack(
                    start_frame=0,
                    stop_frame=-1,
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

                video_track.frame_locations[0] = loc
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
                job_properties=job.job_properties
            )
        except Exception as e:
            # Pass the exception up
            logger.exception(
                "Exception raised while processing audio: " + str(e)
            )
            raise

        logger.info('Processing complete. Found %d tracks.' % len(audio_tracks))
        return audio_tracks


class WhisperSpeechDetectionWrapper:
    def __init__(self):
        self.model = None
        self.initialized = False
        self.size = None
        self.lang = None

        self.iso_map = {
            'af': 'afr',
            'ar': 'ara',
            'hy': 'hye',
            'az': 'aze',
            'be': 'bel',
            'bs': 'bos',
            'bg': 'bul',
            'ca': 'cat',
            'zh': 'zho',
            'cs': 'ces',
            'da': 'dan',
            'nl': 'nld',
            'en': 'eng',
            'et': 'est',
            'fi': 'fin',
            'fr': 'fra',
            'de': 'deu',
            'el': 'ell',
            'he': 'heb',
            'hi': 'hin',
            'hr': 'hrv',
            'hu': 'hun',
            'id': 'ind',
            'gl': 'glg',
            'it': 'ita',
            'is': 'isl',
            'ja': 'jpn',
            'kn': 'kan',
            'kk': 'kaz',
            'ko': 'kor',
            'lv': 'lav',
            'lt': 'lit',
            'mr': 'mar',
            'mi': 'mri',
            'mk': 'mkd',
            'ms': 'msa',
            'ne': 'nep',
            'no': 'nor',
            'fa': 'fas',
            'pl': 'pol',
            'pt': 'por',
            'ro': 'ron',
            'ru': 'rus',
            'sr': 'srp',
            'sk': 'slk',
            'sl': 'slv',
            'es': 'spa',
            'sw': 'swa',
            'sv': 'swe',
            'ta': 'tam',
            'tl': 'tgl',
            'th': 'tha',
            'tr': 'tur',
            'uk': 'ukr',
            'ur': 'urd',
            'vi': 'vie',
            'cy': 'cym'
        }

    def process_audio(self, target_file, start_time, stop_time, job_properties):
        model_size = mpf_util.get_property(job_properties, 'WHISPER_MODEL_SIZE', "base")
        model_lang = mpf_util.get_property(job_properties, 'WHISPER_MODEL_LANG', "multi")
        mode = mpf_util.get_property(job_properties, "WHISPER_MODE", "LANGUAGE_DETECTION")

        if model_lang == "en" and model_size == "large":
            raise mpf.DetectionError.INVALID_PROPERTY.exception("Whisper does not have a large English model available.")

        if not self.initialized:
            self._load_model(model_size, model_lang)
            self.initialized = True

        elif self.size != model_size or self.lang != model_lang:
            self._load_model(model_size, model_lang)

        audio_tracks = []

        if mode == "LANGUAGE_DETECTION":
            if model_lang != "multi":
                raise mpf.DetectionError.INVALID_PROPERTY.exception("Whisper does not support language detection "
                                                                    "using English models. Please use the multilingual "
                                                                    "models.")

            audio = whisper.load_audio(target_file)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            _, probs = self.model.detect_language(mel)
            logger.info(f"Detected language: {max(probs, key=probs.get)}")

            detected_language = max(probs, key=probs.get)
            detected_lang_conf = probs[detected_language]

            iso_639_3 = self.iso_map.get(detected_language, 'UNKNOWN')

            properties = dict(
                DETECTED_LANGUAGE=detected_language,
                ISO_LANGUAGE=iso_639_3
            )

            track = mpf.AudioTrack(
                start_time=start_time,
                stop_time=stop_time,
                confidence=detected_lang_conf,
                detection_properties=properties
            )
            audio_tracks.append(track)

            logger.debug('Completed process audio')

        elif mode == "TRANSCRIPTION":
            properties = self._transcribe_text(target_file, job_properties)

            track = mpf.AudioTrack(
                start_time=start_time,
                stop_time=stop_time,
                detection_properties=properties
            )

            audio_tracks.append(track)
        elif mode == "SPEECH_TRANSLATION":
            properties = self._transcribe_text(target_file, job_properties)
            result = self.model.transcribe(target_file, task="translate")

            properties["TRANSLATED_AUDIO"] = result['text'].strip()

            track = mpf.AudioTrack(
                start_time=start_time,
                stop_time=stop_time,
                detection_properties=properties
            )

            audio_tracks.append(track)

        return audio_tracks

    def _load_model(self, model_size, model_lang):
        self.size = model_size
        self.lang = model_lang

        if self.lang == "en":
            model_string = self.size + "." + "en"
        else:
            model_string = self.size

        logger.info(f'Loading the "{model_string}" model...')
        self.model = whisper.load_model(model_string)

    def _transcribe_text(self, target_file, job_properties):
        language = mpf_util.get_property(job_properties, 'AUDIO_LANGUAGE', "")

        if language == "":
            result = self.model.transcribe(target_file)

            iso_639_3 = self.iso_map.get(result['language'], 'UNKNOWN')

            properties = dict(
                DECODED_LANGUAGE=result['language'],
                ISO_LANGUAGE=iso_639_3,
                TRANSCRIPT=result['text'].strip()
            )

        else:

            iso_639_3 = self.iso_map.get(language, 'UNKNOWN')

            result = self.model.transcribe(target_file, language=language)
            properties = dict(
                DECODED_LANGUAGE=language,
                ISO_LANGUAGE=iso_639_3,
                TRANSCRIPT=result['text'].strip()
            )

        return properties
