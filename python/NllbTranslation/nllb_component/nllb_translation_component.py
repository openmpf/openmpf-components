#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
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
import re
import string
import time

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from typing import Sequence, Dict, Mapping
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .nllb_utils import NllbLanguageMapper

logger = logging.getLogger('Nllb')

class NllbTranslationComponent:

    def __init__(self):
        # get nllb-200-distilled-600M model
        self._model = AutoModelForSeq2SeqLM.from_pretrained('/models/facebook/nllb-200-distilled-600M',
                                                            token=False, local_files_only=True)
    
    def get_detections_from_image(self, job: mpf.ImageJob) -> Sequence[mpf.ImageLocation]:
        logger.info(f'Received image job.')
        return self._get_feed_forward_detections(job.job_properties, job.feed_forward_location, video_job=False)

    def get_detections_from_audio(self, job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        logger.info(f'Received audio job.')
        return self._get_feed_forward_detections(job.job_properties, job.feed_forward_track, video_job=False)
    
    def get_detections_from_video(self, job: mpf.VideoJob) -> Sequence[mpf.VideoTrack]:
        logger.info(f'Received video job.')
        return self._get_feed_forward_detections(job.job_properties, job.feed_forward_track, video_job=True)

    def get_detections_from_generic(self, job: mpf.GenericJob) -> Sequence[mpf.GenericTrack]:
        logger.info(f'Received generic job.')

        if job.feed_forward_track:
            return self._get_feed_forward_detections(job.job_properties, job.feed_forward_track)
        else:
            logger.info('Job did not contain a feed forward track. Creating '
                        'one and assuming translation is in a plain text file.')

            # open file
            with open(job.data_uri, 'r', newline='') as f:
                text = f.read()

            ff_track = mpf.GenericTrack(detection_properties=dict(TEXT=text))

            new_job_props = {
                **job.job_properties,
                'FEED_FORWARD_PROP_TO_PROCESS': 'TEXT'
            }

            return self._get_feed_forward_detections(new_job_props, ff_track)

    def _get_feed_forward_detections(self, job_properties, ff_track, video_job=False):
        try:
            if ff_track is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'Component can only process feed forward '
                    ' jobs, but no feed forward track provided. ')

            # load config
            config = JobConfig(job_properties, ff_track.detection_properties)

            self._get_translation(ff_track, config)

            if video_job:
                for ff_location in ff_track.frame_locations.values():
                    self._get_translation(ff_location, config)

            return [ff_track]

        except Exception:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise
        
    def _get_translation(self, ff_track, config):

        # get tokenizer
        start = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(config.cached_model_location,
                                                  use_auth_token=False, local_files_only=True, src_lang=config.translate_from_language)
        elapsed = time.time() - start
        logger.info(f"Successfully loaded tokenizer in {elapsed} seconds.")

        logger.info(f'Getting Translation....')

        text_to_translate: dict = {}

        for prop_to_translate in config.props_to_translate:
            input_text = ff_track.detection_properties.get(prop_to_translate, None)
            # remove number and punctuation only inputs to prevent bad translations (https://github.com/facebookresearch/fairseq/issues/4854)
            if input_text and not input_text.isdigit() and not (re.match(("^[" + string.punctuation + "]*$").replace("/","\/"), input_text)):
                text_to_translate[prop_to_translate] = input_text

        logger.info(f' Translating from {config.translate_from_language} to {config.translate_to_language}: {text_to_translate}')
        for prop_to_translate, text in text_to_translate.items():
            inputs = self._tokenizer(text, return_tensors="pt")
            translated_tokens = self._model.generate(
                **inputs, forced_bos_token_id=self._tokenizer.encode(config.translate_to_language)[1], max_length=360)

            tranlsation: str = self._tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

            logger.info(f'{prop_to_translate} translation is: {tranlsation}')

            ff_prop_name: str = prop_to_translate + " TRANSLATION"
            ff_track.detection_properties[ff_prop_name] = tranlsation

class JobConfig:
    def __init__(self, props: Mapping[str, str], ff_props):

        self.props_to_translate: list[str] = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=props,
                key='FEED_FORWARD_PROP_TO_PROCESS',
                default_value='TEXT,TRANSCRIPT',
                prop_type=str
            ).split(',')
        ]

        # cached model
        self.cached_model_location: str = mpf_util.get_property(props, 'PRETRAINED_MODEL',
                                                                '/models/facebook/nllb-200-distilled-600M')

        # language to translate to
        self.translate_to_language: str = NllbLanguageMapper.get_code(
            mpf_util.get_property(props, 'TARGET_LANGUAGE', 'eng'),
            mpf_util.get_property(props, 'TARGET_SCRIPT', 'Latn'))

        # get language to translate from
        ff_lang_props: list[str] = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=props,
                key='LANGUAGE_FEED_FORWARD_PROP',
                default_value='ISO_LANGUAGE,DECODED_LANGUAGE,LANGUAGE',
                prop_type=str
            ).split(',')
        ]
        sourceLanguage: str = ''
        # first try getting LANGUAGE_FEED_FORWARD_PROP otherwise use DEFAULT_SOURCE_LANGUAGE
        language = ff_props.get("LANGUAGE", None)
        for lang in ff_lang_props:
            if ff_props.get(lang, ''):
                sourceLanguage = ff_props.get(lang, '')
                break
        if not sourceLanguage:
            sourceLanguage = mpf_util.get_property(props, 'DEFAULT_SOURCE_LANGUAGE', '')

        # get script to translate from
        ff_script_props: list[str] = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=props,
                key='SCRIPT_FEED_FORWARD_PROP',
                default_value='ISO_SCRIPT,DECODED_SCRIPT,SCRIPT',
                prop_type=str
            ).split(',')
        ]
        # first try getting SCRIPT_FEED_FORWARD_PROP otherwise use DEFAULT_SOURCE_SCRIPT
        sourceScript: str = ''
        for script in ff_script_props:
            if ff_props.get(script, ''):
                sourceScript = ff_props.get(script, '')
                break
        if not sourceScript:
            sourceScript = mpf_util.get_property(props, 'DEFAULT_SOURCE_SCRIPT', '')

        self.translate_from_language = NllbLanguageMapper.get_code(
            sourceLanguage,
            sourceScript)
        
        # if no source language is provided or language is not supported, no translation can be done
        if not bool(self.translate_from_language):
            logger.exception('Unsupported or no source language provided')
            raise mpf.DetectionException(
                f'Target translation ({sourceLanguage}) language is empty or unsupported',
                mpf.DetectionError.INVALID_PROPERTY)
