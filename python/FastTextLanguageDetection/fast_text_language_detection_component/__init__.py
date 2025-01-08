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

from __future__ import annotations

import dataclasses
import importlib.resources
import logging
import pathlib
import typing
from typing import Dict, Mapping, NamedTuple, Optional, Set, Tuple, TypeVar

import fasttext
import numpy as np
import numpy.typing as npt
from fasttext.FastText import _FastText as FastTextModel
from huggingface_hub import hf_hub_download

import mpf_component_api as mpf

log = logging.getLogger('FastTextLanguageDetectionComponent')


class FastTextLanguageDetectionComponent:

    @staticmethod
    def get_detections_from_generic(job: mpf.GenericJob) -> Tuple[mpf.GenericTrack]:
        if job.feed_forward_track:
            return get_detections_from_non_composite(job.job_properties, job.feed_forward_track)

        text = job.media_properties.get('SELECTED_CONTENT')
        if not text:
            log.info(
                'Job did not contain a feed forward track or specify selected content. Assuming '
                'media file is a plain text file containing the text to be processed.')
            text = pathlib.Path(job.data_uri).read_text().strip()

        track = mpf.GenericTrack(detection_properties={'TEXT': text})
        modified_job_props = {**job.job_properties, 'FEED_FORWARD_PROP_TO_PROCESS': 'TEXT'}
        return get_detections_from_non_composite(modified_job_props, track)


    @staticmethod
    def get_detections_from_image(job: mpf.ImageJob) -> Tuple[mpf.ImageLocation]:
        return get_detections_from_non_composite(job.job_properties, job.feed_forward_location)


    @staticmethod
    def get_detections_from_audio(job: mpf.AudioJob) -> Tuple[mpf.AudioTrack]:
        return get_detections_from_non_composite(job.job_properties, job.feed_forward_track)

    @staticmethod
    def get_detections_from_video(job: mpf.VideoJob) -> Tuple[mpf.VideoTrack]:
        try:
            log.info('Received video job.')
            ff_track = job.feed_forward_track
            if not ff_track:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    'Component can only process feed forward jobs, '
                    'but no feed forward track provided. ')
            detector = LanguageDetector(job.job_properties)
            detector.add_language_detections(ff_track.detection_properties)
            for ff_location in ff_track.frame_locations.values():
                detector.add_language_detections(ff_location.detection_properties)
            log.info('Processing complete.')

            return (ff_track,)
        except Exception:
            log.exception('Failed to complete job due to the following exception:')
            raise


T_FF_OBJ = TypeVar('T_FF_OBJ', mpf.AudioTrack, mpf.GenericTrack, mpf.ImageLocation)

def get_detections_from_non_composite(
        job_properties: Mapping[str, str],
        ff_track: Optional[T_FF_OBJ]) -> Tuple[T_FF_OBJ]:
    try:
        log.info('Received job.')
        if ff_track is None:
            raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                'Component can only process feed forward jobs, '
                'but no feed forward track provided.')
        detector = LanguageDetector(job_properties)
        detector.add_language_detections(ff_track.detection_properties)
        log.info('Processing complete.')
        return (ff_track,)
    except Exception:
        log.exception('Failed to complete job due to the following exception:')
        raise


_UNKNOWN = '<UNKNOWN>'

class LanguageDetector:
    _CACHED_MODEL: Optional[CachedModel] = None

    def __init__(self, job_properties: Mapping[str, str]):
        self._model = self._get_model(job_properties)
        ff_prop_names = job_properties.get('FEED_FORWARD_PROP_TO_PROCESS', 'TEXT,TRANSCRIPT')
        self._ff_props_to_process = [p.strip() for p in ff_prop_names.split(',')]
        # Certain components that process videos and don't do tracking will create tracks with a
        # single detection. The track and detection will have the same set of properties, so
        # it is relatively likely we will run in to duplicate text.
        self._lang_id_cache: Dict[str, LangDetectResult] = {}


    def add_language_detections(self, detection_properties: Dict[str, str]) -> None:
        for prop_name in self._ff_props_to_process:
            text_to_process = detection_properties.get(prop_name)
            if not text_to_process:
                continue
            log.info(f'Attempting to detect the language of the "{prop_name}" property...')
            detect_result = self._model_predict(text_to_process)
            log.info(f'Language detection result: {detect_result}')
            detection_properties['ISO_LANGUAGE'] = detect_result.lang
            detection_properties['ISO_SCRIPT'] = detect_result.script
            detection_properties['PRIMARY_LANGUAGE_CONFIDENCE'] = str(detect_result.score)
            return # Only process first matched property.


    def _model_predict(self, text: str) -> LangDetectResult:
        if cached := self._lang_id_cache.get(text):
            return cached

        if not self._has_alpha_chars(text):
            log.warning(
                'Cannot detect language because the text did not include any Unicode letters or '
                'CJK characters.')
            return LangDetectResult.unknown()

        # The model does not allow newline characters.
        text_no_newline = text.replace('\n', ' ')
        labels, scores = typing.cast(
            Tuple[npt.NDArray[np.str_], npt.NDArray[np.float64]],
            self._model.predict(text_no_newline))
        result = self._post_process_model_output(text, labels[0], scores[0])
        self._lang_id_cache[text] = result
        return result

    @staticmethod
    def _has_alpha_chars(text: str) -> bool:
        # str.isAlpha() checks if the character is categorized as "Letter" in the Unicode database.
        # This includes characters for non-alphabetic languages such as Chinese.
        return any(c.isalpha() for c in text)

    _LABEL_PREFIX_LEN = len('__label__')

    @classmethod
    def _post_process_model_output(cls, text: str, label: str, score: float) -> LangDetectResult:
        # Example label: "__label__spa_Latn"
        no_prefix = label[cls._LABEL_PREFIX_LEN:]
        lang, _, script = no_prefix.upper().partition('_')
        if not script:
            script = _UNKNOWN
        if lang == 'LZH':
            # The model reports Classical Chinese (LZH) with very high confidence when provided
            # with things like an empty string or a string that is all emoji.
            return LangDetectResult.unknown()
        if lang == 'CMN':
            lang = 'ZHO'
        if script == 'HANI':
            script = 'HANT' if TraditionalChineseChars.has_any(text) else 'HANS'

        # Sometimes the score is slightly over 1. For example: 1.0000077486038208
        fixed_score = min(score, 1)
        return LangDetectResult(lang, script, fixed_score)


    @classmethod
    def _get_model(cls, job_properties: Mapping[str, str]) -> FastTextModel:
        model_params = ModelLoadParams(job_properties)
        if cls._CACHED_MODEL and cls._CACHED_MODEL.parameters == model_params:
            return cls._CACHED_MODEL.model

        if model_params.path:
            path = model_params.path
        else:
            assert model_params.hf_repo and model_params.hf_file
            path = hf_hub_download(repo_id=model_params.hf_repo, filename=model_params.hf_file)
        log.info(f'Loading model from: {path}')
        cls._CACHED_MODEL = CachedModel(model_params, fasttext.load_model(path))
        return cls._CACHED_MODEL.model


class CachedModel(NamedTuple):
    parameters: ModelLoadParams
    model: FastTextModel


@dataclasses.dataclass
class ModelLoadParams:
    job_properties: dataclasses.InitVar[Mapping[str, str]]
    path: Optional[str] = None
    hf_repo: Optional[str] = None
    hf_file: Optional[str] = None

    def __post_init__(self, job_properties: Mapping[str, str]):
        self.path = self._get_prop('MODEL_PATH', job_properties)
        if self.path:
            self.hf_repo = None
            self.hf_file = None
        else:
            self.hf_repo = self._get_prop('HUGGING_FACE_REPO', job_properties, 'cis-lmu/glotlid')
            self.hf_file = self._get_prop('HUGGING_FACE_FILE', job_properties, 'model.bin')

    @staticmethod
    def _get_prop(name: str, job_properties: Mapping[str, str], default=None):
        prop_value = job_properties.get(name)
        if prop_value and (stripped := prop_value.strip()):
            return stripped
        else:
            return default


class LangDetectResult(NamedTuple):
    lang: str
    script: str
    score: float

    @classmethod
    def unknown(cls):
        return cls(_UNKNOWN, _UNKNOWN, 0)


class TraditionalChineseChars:
    @classmethod
    def has_any(cls, text: str) -> bool:
        traditional_only_chars = cls._get_traditional_only_char_set()
        return any(ch in traditional_only_chars for ch in text)

    @classmethod
    def _get_traditional_only_char_set(cls) -> Set[str]:
        # chinese-traditional-only.txt is a plain text file containing each Chinese character that
        # is only used in Traditional Chinese.
        chars = importlib.resources.read_text(__name__, 'chinese-traditional-only.txt').strip()
        char_set = set(chars)
        cls._get_traditional_only_char_set = lambda: char_set
        return char_set
