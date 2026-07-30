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

import ctypes
import logging
import regex as re
import time
import os

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from typing import Dict, Optional, Sequence, Mapping, TypeVar, Callable
from .nllb_utils import NllbLanguageMapper
from nlp_text_splitter import TextSplitterModel, TextSplitter, WtpLanguageSettings

import ctranslate2
import sentencepiece as spm

logger = logging.getLogger('NllbTranslationComponent')

# Roll-up TypeDef for different track types
T_FF_OBJ = TypeVar('T_FF_OBJ', mpf.AudioTrack, mpf.GenericTrack, mpf.ImageLocation, mpf.VideoTrack)

# default NLLB model
DEFAULT_NLLB_MODEL = 'OpenNMT/nllb-200-3.3B-ct2-int8'
SP_MODEL_PATH = '/models/OpenNMT/flores200_sacrebleu_tokenizer_spm.model'

# compile this pattern once
NO_TRANSLATE_PATTERN = re.compile(r'[[:space:][:digit:][:punct:]\p{Nonspacing_Mark}\u1734\p{Spacing_Mark}\p{Enclosing_Mark}\p{Decimal_Number}\p{Letter_Number}\p{Other_Number}\p{Format}]*')

class NllbTranslationComponent:

    def __init__(self) -> None:
        self._load_model()
        self._tokenizer = None
        self._current_model_name = None

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
        # The CTranslate2 model does not carry its own tokenizer; tokenization is
        # done with the FLORES-200 SentencePiece model, which is language-agnostic
        # (the source language is prepended as a token at encode time, not baked
        # into the tokenizer), so a single instance is cached for all jobs.
        if self._tokenizer is not None:
            return
        start = time.time()
        self._tokenizer = spm.SentencePieceProcessor()
        self._tokenizer.load(SP_MODEL_PATH)
        logger.debug(f"Successfully loaded tokenizer in {time.time() - start} seconds.")

    def _load_model(self, model_name: str = None, config: Dict[str, str] = None) -> None:
        try:
            if model_name is None:
                if config is None:
                    model_name = DEFAULT_NLLB_MODEL
                else:
                    model_name = config.nllb_model

            model_path = '/models/' + model_name
            device = self._resolve_device()

            if not (os.path.isdir(model_path)
                    and os.path.isfile(os.path.join(model_path, "config.json"))):
                raise mpf.DetectionException(
                    f'CTranslate2 model directory not found: {model_path}',
                    mpf.DetectionError.COULD_NOT_READ_DATAFILE)

            logger.info(f"Loading model from local directory: {model_path} (device={device})")
            self._model = ctranslate2.Translator(model_path, device=device)
            self._current_model_name = model_name
            logger.info(f"Model loaded: {model_path} "
                        f"(device={device}, compute_type={self._model.compute_type})")

        except Exception:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise

    @staticmethod
    def _resolve_device() -> str:
        """Choose the CTranslate2 device.

        Use CUDA only when a GPU is visible *and* the CUDA runtime library the
        engine needs (cuBLAS) can actually be loaded. This avoids the runtime
        failure "Library libcublas.so.12 is not found or cannot be loaded" that
        occurs when a GPU is exposed to the container (e.g. ``--gpus all``) but
        the CUDA runtime is not installed in the image. Otherwise fall back to
        CPU.
        """
        try:
            if ctranslate2.get_cuda_device_count() <= 0:
                return "cpu"
        except Exception:
            return "cpu"

        # ctranslate2 dlopen's cuBLAS from the default loader path; probe the
        # same way so our choice matches what the engine can actually load.
        for lib in ("libcublas.so.12", "libcublas.so"):
            try:
                ctypes.CDLL(lib)
                return "cuda"
            except OSError:
                continue

        logger.warning(
            "A CUDA device is visible but the CUDA runtime (libcublas) could "
            "not be loaded; falling back to CPU for NLLB translation.")
        return "cpu"

    def _check_model(self, config: Dict[str, str]) -> None:
        # TODO (Phase 4): ctranslate2.Translator has no name_or_path, so the
        # loaded model is tracked in self._current_model_name. This still does
        # not reload on a name change -- see PLAN.md Phase 4.
        if not self._model.model_is_loaded:
            self._load_model(config=config)

    def _get_text_size_function(self, config: Dict[str, str]) -> Callable[[str], int]:
        if config.use_token_length:
            # SentencePiece equivalent of develop's HF token count. The +2
            # accounts for the source-language token and </s> that _encode adds,
            # matching what the model actually receives.
            # TODO (Phase 2): replace with the tokenizer backend's count_tokens
            # so this works for the HuggingFace backend too.
            count_tokens: Callable[[str], int] = (
                lambda txt: len(self._tokenizer.encode_as_pieces(txt)) + 2
            )
            return count_tokens
        else:
            return len

    def _add_translations(self, ff_track: T_FF_OBJ, config: Dict[str, str]) -> None:
        for prop_name in config.props_to_translate:
            text_to_translate = ff_track.detection_properties.get(prop_name, None)
            if text_to_translate:
                translation = self._get_translation(config, {prop_name: text_to_translate})
                ff_prop_name = self._get_ff_prop_name(prop_name, config)
                ff_track.detection_properties[ff_prop_name] = translation
                if not config.translate_all_ff_properties:
                    break

    def _get_translation(self, config: Dict[str, str], text_to_translate: Dict[str, str]) -> str:
        # make sure the model loaded matches model set in job config
        self._check_model(config)
        self._load_tokenizer(config)
        get_size_fn = self._get_text_size_function(config)

        logger.info(f'Translating from {config.translate_from_language} to {config.translate_to_language}')

        for prop_to_translate, text in text_to_translate.items():
            if config.use_token_length:
                hard_limit = config.nllb_token_limit
                preferred_limit = config.nllb_token_soft_limit
            else:
                hard_limit = config.nllb_character_limit
                preferred_limit = -1

            split_mode = config._sentence_split_mode.upper()
            difficult_set = config.difficult_languages

            # Difficult-language override: replace the preferred token limit with more strict limit.
            if config.use_token_length and _is_difficult_language(config.translate_from_language, difficult_set):
                diff_limit = config.difficult_language_token_limit
                if diff_limit > 0:
                    old_preferred = preferred_limit
                    preferred_limit = diff_limit
                    logger.warning(
                        "Difficult language detected (%s). Applying DIFFICULT_LANGUAGE_TOKEN_LIMIT "
                        "as the preferred limit: %s -> %d. "
                        "Translations may be less reliable for this language.",
                        config.translate_from_language,
                        old_preferred,
                        preferred_limit
                    )
                else:
                    logger.warning(
                        "Difficult language detected (%s), but no DIFFICULT_LANGUAGE_TOKEN_LIMIT "
                        "override is configured.",
                        config.translate_from_language
                    )

            if preferred_limit is None or preferred_limit <= 0:
                preferred_limit = -1
            else:
                preferred_limit = min(int(preferred_limit), int(hard_limit))

            current_text_size = get_size_fn(text)
            effective_split_threshold = hard_limit if preferred_limit <= 0 else preferred_limit

            logger.info(
                f"Translation chunking limits: hard={hard_limit}"
                + (f", preferred={preferred_limit}" if preferred_limit > 0 else "")
                + f" ({'tokens' if config.use_token_length else 'characters'}); "
                f"split_mode={split_mode}"
            )

            if current_text_size <= effective_split_threshold:
                text_list = [text]
            else:
                # Determine WtP language for sentence splitting.
                wtp_lang: Optional[str] = WtpLanguageSettings.convert_to_iso(config.translate_from_language)

                if wtp_lang is None:
                    default_adaptor = config.nlp_model_default_language
                    # Allow default_adaptor to already be ISO ("en", "fr", ...) or an NLLB tag.
                    wtp_lang = WtpLanguageSettings.convert_to_iso(default_adaptor) or default_adaptor

                if not wtp_lang:
                    wtp_lang = "en"

                text_splitter_model = TextSplitterModel(
                    config.nlp_model_name,
                    config.nlp_model_setting,
                    wtp_lang
                )

                if config.use_token_length:
                    logger.info(
                        f"Text size ({current_text_size}) exceeds split threshold ({effective_split_threshold}) tokens. "
                        f"Splitting with hard_limit={hard_limit}, preferred_limit={preferred_limit}."
                    )
                else:
                    logger.info(
                        f"Text size ({current_text_size}) exceeds split threshold ({effective_split_threshold}) characters. "
                        f"Splitting with hard_limit={hard_limit}."
                    )

                if config._incl_input_lang:
                    input_text_sentences = TextSplitter.split(
                        text,
                        hard_limit,
                        0,
                        get_size_fn,
                        text_splitter_model,
                        wtp_lang,
                        split_mode=split_mode,
                        newline_behavior=config._newline_behavior,
                        preferred_limit=preferred_limit
                    )
                else:
                    input_text_sentences = TextSplitter.split(
                        text,
                        hard_limit,
                        0,
                        get_size_fn,
                        text_splitter_model,
                        split_mode=split_mode,
                        newline_behavior=config._newline_behavior,
                        preferred_limit=preferred_limit
                    )

                text_list = list(input_text_sentences)
                logger.info(f'Input text split into {len(text_list)} chunks.')

            logger.info('Translating chunks...')

            # Chunks that should not be translated (whitespace/punctuation/digits
            # only) are held back rather than sent through the model and swapped
            # out afterwards: CTranslate2 translates in batches, so filtering
            # first saves work instead of discarding it.
            to_translate = [(i, s) for i, s in enumerate(text_list) if should_translate(s)]
            translated_indices = {i for i, _ in to_translate}
            translations: list[str] = list(text_list)

            for i, sentence in enumerate(text_list):
                if i not in translated_indices:
                    logger.debug(f'Skipping translation for:\n{sentence.strip()}')

            if to_translate:
                sentences_subworded = [
                    [config.translate_from_language] + pieces + ["</s>"]
                    for pieces in self._tokenizer.encode_as_pieces(
                        [s for _, s in to_translate])
                ]
                target_prefix = [[config.translate_to_language]] * len(sentences_subworded)

                results = self._model.translate_batch(
                    sentences_subworded,
                    batch_type="tokens",
                    max_batch_size=config.nllb_max_batch_size,
                    beam_size=config.nllb_beam_size,
                    max_decoding_length=hard_limit,
                    target_prefix=target_prefix)

                hypotheses = [r.hypotheses[0] for r in results]
                for h in hypotheses:
                    if config.translate_to_language in h:
                        h.remove(config.translate_to_language)

                for (idx, source), decoded in zip(to_translate,
                                                  self._tokenizer.decode(hypotheses)):
                    translations[idx] = decoded
                    logger.debug(f'Translated:\n{source.strip()}\nto:\n{decoded.strip()}')

            # Keep existing behavior: add spaces between translated chunks
            translation = " ".join(translations)

            logger.debug(f'Translated {prop_to_translate} property to:\n{translation.strip()}')
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

        self._sentence_split_mode = mpf_util.get_property(
            props, 'SENTENCE_SPLITTER_MODE', 'DEFAULT')

        self._newline_behavior = mpf_util.get_property(
            props, 'SENTENCE_SPLITTER_NEWLINE_BEHAVIOR', 'GUESS')

        # default model, cached
        self.nllb_model = mpf_util.get_property(props, "NLLB_MODEL", DEFAULT_NLLB_MODEL)

        # language/script to translate to
        targetLanguage: str = mpf_util.get_property(props, 'TARGET_LANGUAGE', 'eng')
        targetScript: str = mpf_util.get_property(props, 'TARGET_SCRIPT', 'Latn')
        try:
            self.translate_to_language: Optional[str] = NllbLanguageMapper.get_code(
                targetLanguage,
                targetScript)
        except KeyError:
            logger.exception(
                f'Unsupported target script provided')
            raise mpf.DetectionException(
                 f'Target script ({targetScript}) is unsupported',
                mpf.DetectionError.INVALID_PROPERTY)
        except:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise

        if not self.translate_to_language:
            logger.exception('Unsupported target language provided')
            raise mpf.DetectionException(
                f'Target language ({targetLanguage}) is not supported',
                mpf.DetectionError.INVALID_PROPERTY)

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
        except:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise


        if not self.translate_from_language:
            logger.exception('Unsupported or no source language provided')
            raise mpf.DetectionException(
                f'Source language ({sourceLanguage}) is empty or unsupported',
                mpf.DetectionError.INVALID_PROPERTY)

        self.use_token_length = mpf_util.get_property(props, 'USE_NLLB_TOKEN_LENGTH', True)
        self.nllb_token_limit = mpf_util.get_property(props, 'NLLB_TRANSLATION_TOKEN_LIMIT', 512)
        # set translation limit. default to 360 if no value set
        self.nllb_character_limit = mpf_util.get_property(props, 'SENTENCE_SPLITTER_CHAR_COUNT', 360)


        self.nllb_token_soft_limit = mpf_util.get_property(
            props, 'NLLB_TRANSLATION_TOKEN_SOFT_LIMIT', 130
        )

        # CTranslate2 decoding. Defaults preserve the values the ctranslate2
        # branch hardcoded, so behavior is unchanged by exposing them.
        # TODO (Phase 5.1): document these in descriptor.json.
        self.nllb_beam_size = mpf_util.get_property(props, 'NLLB_BEAM_SIZE', 4)
        self.nllb_max_batch_size = mpf_util.get_property(props, 'NLLB_MAX_BATCH_SIZE', 2024)

        difficult_lang_list = mpf_util.get_property(
            props, 'PROCESS_DIFFICULT_LANGUAGES', 'arabic'
        )

        self.difficult_languages = {
            x.strip().lower() for x in difficult_lang_list.split(',') if x.strip()
        }

        # Opt-in token limit override for difficult languages (0 disables)
        self.difficult_language_token_limit = mpf_util.get_property(
            props, 'DIFFICULT_LANGUAGE_TOKEN_LIMIT', 50
        )

        self.nlp_model_name = mpf_util.get_property(props, "SENTENCE_MODEL", "sat-3l-sm")

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


# Arabic languages are marked as difficult for translation.
# These are NLLB/Flores language IDs
_ARABIC_FLORES_LANGS = {
    "arb",  # Modern Standard Arabic
    "acm",  # Mesopotamian Arabic
    "acq",  # Ta’izzi-Adeni Arabic
    "aeb",  # Tunisian Arabic
    "ajp",  # South Levantine Arabic
    "apc",  # North Levantine Arabic
    "ars",  # Najdi Arabic
    "ary",  # Moroccan Arabic
    "arz",  # Egyptian Arabic
}

def _is_difficult_language(source_flores_code: str, configured: set[str]) -> bool:
    """
    Return True if source language should trigger difficult-language logic.
    - configured is a set of normalized strings, e.g. {"arabic"} or {"arb"}.
    - apply this to arabic languages over arabic script.
    """
    if not source_flores_code:
        return False

    code = source_flores_code.strip().lower()
    base = code.split("_", 1)[0]

    if code in configured or base in configured:
        return True

    # Apply to known Arabic languages in NLLB/Flores
    if "arabic" in configured and base in _ARABIC_FLORES_LANGS:
        return True

    return False