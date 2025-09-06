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
import regex as re
import time
import os

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from typing import Dict, Optional, Sequence, Mapping, TypeVar
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .nllb_utils import NllbLanguageMapper
from nlp_text_splitter import TextSplitterModel, TextSplitter, WtpLanguageSettings

logger = logging.getLogger('NllbTranslationComponent')

# Roll-up TypeDef for different track types
T_FF_OBJ = TypeVar('T_FF_OBJ', mpf.AudioTrack, mpf.GenericTrack, mpf.ImageLocation, mpf.VideoTrack)

# default NLLB model
DEFAULT_NLLB_MODEL = 'facebook/nllb-200-3.3B'

# compile this pattern once
NO_TRANSLATE_PATTERN = re.compile(r'[[:space:][:digit:][:punct:]\p{Nonspacing_Mark}\u1734\p{Spacing_Mark}\p{Enclosing_Mark}\p{Decimal_Number}\p{Letter_Number}\p{Other_Number}\p{Format}]*')

class NllbTranslationComponent:

    def __init__(self) -> None:
        self._load_model()
        self._tokenizer = None

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

    def _get_feed_forward_detections(self, job_properties: Dict[str, str],
                                     ff_track: T_FF_OBJ,
                                     video_job: bool = False) -> Sequence[T_FF_OBJ]:
        try:
            if ff_track is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'Component can only process feed forward '
                    ' jobs, but no feed forward track provided. ')

            # load config
            config = JobConfig(job_properties, ff_track.detection_properties)

            self._add_translations(ff_track, config)

            if video_job:
                for ff_location in ff_track.frame_locations.values():
                    self._add_translations(ff_location, config)

            return [ff_track]

        except Exception:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise

    def _load_tokenizer(self, config: Dict[str, str]) -> None:
        start = time.time()
        model_path = '/models/' + config.nllb_model
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(model_path,
                                            use_auth_token=False, local_files_only=True,
                                            src_lang=config.translate_from_language, device_map=self._model.device)
        elif self._tokenizer.src_lang != config.translate_from_language: # reload with updated src_lang
            logger.info(f"Detected a change in tokenizer source language ({self._tokenizer.src_lang} -> {config.translate_from_language}). Re-initializing tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                use_auth_token=False, local_files_only=True,
                                                src_lang=config.translate_from_language, device_map=self._model.device)
            elapsed = time.time() - start
            logger.debug(f"Successfully loaded tokenizer in {elapsed} seconds.")
    
    def _load_model(self, model_name: str = None, config: Dict[str, str] = None) -> None:
        try:
            if model_name is None:
                if config is None:
                    model_name = DEFAULT_NLLB_MODEL
                else:
                    model_name = config.nllb_model
            
            model_path = '/models/' + model_name
            offload_folder = model_path + '/.weights'
            
            if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json")):
                # model is stored locally; we do not need to load the tokenizer here
                logger.info(f"Loading model from local directory: {model_path}")
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=False, local_files_only=True,
                                                                    device_map="auto", offload_folder=offload_folder)
            else:
                # model is not stored locally, download and save
                logger.info(f"Downloading model from Hugging Face: {model_name}")
                self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=False, local_files_only=False,
                                                                    device_map="auto", offload_folder=offload_folder)
                self._tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=False, local_files_only=False,
                                                                device_map=self._model.device)
                logger.debug(f"Saving model in {model_path}")
                self._model.save_pretrained(model_path)
                self._tokenizer.save_pretrained(model_path)
    
        except Exception:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise

    def _check_model(self, config: Dict[str, str]) -> None:
        loaded_model = self._model.name_or_path
        config_model = '/models/' + config.nllb_model
        if loaded_model != config_model:
            logger.info(f"Current model '{loaded_model}' does not match JobConfig model. Loading '{config.nllb_model}'...")
            self._model = None
            self._tokenizer = None
            self._load_model(config=config)

    def _add_translations(self, ff_track: T_FF_OBJ, config: Dict[str, str]) -> None:
        for prop_name in config.props_to_translate:
            text_to_translate = ff_track.detection_properties.get(prop_name, None)
            if text_to_translate:
                translation = self._get_translation(config, {prop_name: text_to_translate})
                ff_prop_name = self._get_ff_prop_name(prop_name, config)
                ff_track.detection_properties[ff_prop_name] = translation
                if not config.translate_all_ff_properties:
                    break

    def _get_translation(self, config: Dict[str, str], text_to_translate: str) -> str:
        # make sure the model loaded matches model set in job config
        self._check_model(config)
        self._load_tokenizer(config)

        logger.info(f'Performing translation...')
        for prop_to_translate, text in text_to_translate.items():
            logger.debug(f'Translating from {config.translate_from_language} to {config.translate_to_language}: {text_to_translate}')

            # split input text into a list of sentences to support max translation length of 360 characters
            logger.info(f'Translating character limit set to: {config.nllb_character_limit}')
            if len(text) < config.nllb_character_limit:
                text_list = [text]
            else:
                # split input values & model
                wtp_lang: Optional[str] = WtpLanguageSettings.convert_to_iso(
                    NllbLanguageMapper.get_normalized_iso(config.translate_from_language))
                if wtp_lang is None:
                    wtp_lang = WtpLanguageSettings.convert_to_iso(config.nlp_model_default_language)

                text_splitter_model = TextSplitterModel(config.nlp_model_name, config.nlp_model_setting, wtp_lang)

                logger.debug(f'Text to translate is larger than the {config.nllb_character_limit} character limit, splitting into smaller sentences')
                if config._incl_input_lang:
                    input_text_sentences = TextSplitter.split(
                        text,
                        config.nllb_character_limit,
                        0,
                        len,
                        text_splitter_model,
                        wtp_lang)
                else:
                    input_text_sentences = TextSplitter.split(
                        text,
                        config.nllb_character_limit,
                        0,
                        len,
                        text_splitter_model)

                text_list = list(input_text_sentences)
                logger.debug(f'Input text split into {len(text_list)} segments.')

            translations = []

            for sentence in text_list:
                if should_translate(sentence):
                    inputs = self._tokenizer(sentence, return_tensors="pt").to(self._model.device)
                    translated_tokens = self._model.generate(
                        **inputs, forced_bos_token_id=self._tokenizer.encode(config.translate_to_language)[1], max_length=config.nllb_character_limit)

                    sentence_translation: str = self._tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

                    translations.append(sentence_translation)
                else:
                    translations.append(sentence)

            # spaces between sentences are added
            translation = " ".join(translations)

            logger.debug(f'{prop_to_translate} translation is: {translation}')

            return translation

    def _get_ff_prop_name(self, prop_to_translate: str, config: Dict[str, str]) -> str:
        if config.translate_all_ff_properties:
            ff_prop_name: str = prop_to_translate + " TRANSLATION"
        else:
            ff_prop_name: str = "TRANSLATION"
        return ff_prop_name

class JobConfig:
    def __init__(self, props: Mapping[str, str], ff_props: Dict[str, str]) -> None:

        self.translate_all_ff_properties = mpf_util.get_property(props, 'TRANSLATE_ALL_FF_PROPERTIES', False)

        self.props_to_translate: list[str] = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=props,
                key='FEED_FORWARD_PROP_TO_PROCESS',
                default_value='TEXT,TRANSCRIPT',
                prop_type=str
            ).split(',')
        ]

        # default model, cached
        self.nllb_model = mpf_util.get_property(props, "NLLB_MODEL", DEFAULT_NLLB_MODEL)

        # language to translate to
        self.translate_to_language: Optional[str] = NllbLanguageMapper.get_code(
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

        try:
            self.translate_from_language = NllbLanguageMapper.get_code(
                sourceLanguage,
                sourceScript)
        except KeyError:
            logger.exception(
                f'Unsupported or no source script provided')
            raise mpf.DetectionException(
                 f'Source script ({sourceScript}) is empty or unsupported',
                mpf.DetectionError.INVALID_PROPERTY)
        
        if not self.translate_from_language:
            logger.exception('Unsupported or no source language provided')
            raise mpf.DetectionException(
                f'Source language ({sourceLanguage}) is empty or unsupported',
                mpf.DetectionError.INVALID_PROPERTY)

        # set translation limit. default to 360 if no value set
        self.nllb_character_limit = mpf_util.get_property(props, 'SENTENCE_SPLITTER_CHAR_COUNT', 360)

        self.nlp_model_name = mpf_util.get_property(props, "SENTENCE_MODEL", "wtp-bert-mini")

        nlp_model_cpu_only = mpf_util.get_property(props, "SENTENCE_MODEL_CPU_ONLY", True)
        if not nlp_model_cpu_only:
            self.nlp_model_setting = "cuda"
        else:
            self.nlp_model_setting = "cpu"

        # SENTENCE_SPLITTER_INCLUDE_INPUT_LANG and SENTENCE_MODEL_WTP_DEFAULT_ADAPTOR_LANGUAGE
        sentence_splitter_include_input_lang = mpf_util.get_property(props, "SENTENCE_SPLITTER_INCLUDE_INPUT_LANG", True)
        if sentence_splitter_include_input_lang:
            self._incl_input_lang = True
            self.nlp_model_default_language = mpf_util.get_property(props, "SENTENCE_MODEL_WTP_DEFAULT_ADAPTOR_LANGUAGE", 'en')
        else:
            self.nlp_model_default_language = None
            self._incl_input_lang = False

def should_translate(sentence: any) -> bool:
    if sentence and not NO_TRANSLATE_PATTERN.fullmatch(sentence):
        return True
    else:
        return False
