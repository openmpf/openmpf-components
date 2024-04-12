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
import os
import pkg_resources

import spacy
from wtpsplit import WtP
from typing import Callable, List, Optional, Tuple

from .wtp_lang_settings import WtpLanguageSettings


DEFAULT_WTP_MODELS = "/wtp_models"

# If we want to package model installation with this utility in the future:
WTP_MODELS_PATH = pkg_resources.resource_filename(
   __name__, "models"
)

log = logging.getLogger(__name__)

# These models must have an specified language during sentence splitting.
WTP_MANDATORY_ADAPTOR = ['wtp-canine-s-1l',
                         'wtp-canine-s-3l',
                         'wtp-canine-s-6l',
                         'wtp-canine-s-9l',
                         'wtp-canine-s-12l']

class TextSplitterModel:
    # To hold spaCy, WtP, and other potential sentence detection models in cache

    def __init__(self, model_name: str, model_setting: str, default_lang: str = "en") -> None:
        self._model_name = ""
        self._default_lang = default_lang
        self._mandatory_wtp_language = False
        self.split = lambda t, **param: [t]
        self.update_model(model_name, model_setting)

    def update_model(self, model_name: str, model_setting: str = "", default_lang: str="en"):
        if model_name:
            if "wtp" in model_name:
                self._update_wtp_model(model_name, model_setting, default_lang)
                self.split = self._split_wtp
                log.info(f"Setup WtP model: {model_name}")
            else:
                self._update_spacy_model(model_name)
                self.split = self._split_spacy
                log.info(f"Setup spaCy model: {model_name}")

    def _update_wtp_model(self, wtp_model_name: str,
                          model_setting: str = "cpu",
                          default_lang: str="en") -> None:

        if wtp_model_name in WTP_MANDATORY_ADAPTOR:
            self._mandatory_wtp_language = True
            self._default_lang = default_lang

        if self._model_name != wtp_model_name:
            self._model_name = wtp_model_name
            # Check if model has been downloaded
            if os.path.exists(os.path.join(WTP_MODELS_PATH, wtp_model_name)):
                log.info(f"Using downloaded {wtp_model_name} model.")
                wtp_model_name = os.path.join(WTP_MODELS_PATH, wtp_model_name)

            elif os.path.exists(os.path.join(DEFAULT_WTP_MODELS,
                                             wtp_model_name)):

                log.info(f"Using downloaded {wtp_model_name} model.")
                wtp_model_name = os.path.join(DEFAULT_WTP_MODELS,
                                              wtp_model_name)

            else:
                log.warning(f"Model {wtp_model_name} not found, "
                             "downloading from hugging face.")

            self.wtp_model =  WtP(wtp_model_name)

            if model_setting != "cpu" and model_setting != "cuda":
                log.warning(f"Invalid setting for WtP runtime {model_setting}. "
                             "Defaulting to CPU mode.")
                model_setting = "cpu"
            self.wtp_model.to(model_setting)

    def _split_wtp(self, text: str, lang: Optional[str] = None) -> List[str]:
        if lang:
            iso_lang = WtpLanguageSettings.convert_to_iso(lang)
            if iso_lang:
                return self.wtp_model.split(text, lang_code=iso_lang)
            else:
                log.warning(f"Language {lang} was not used to train WtP model. "
                            "If text splitting is not working well with WtP, "
                            "consider trying spaCy's sentence detection model."
                            )
        if self._mandatory_wtp_language:
            log.warning("WtP model requires a language."
                        f"Using default language : {self._default_lang}.")
            iso_lang = WtpLanguageSettings.convert_to_iso(self._default_lang)
            return self.wtp_model.split(text, lang_code=iso_lang)
        return self.wtp_model.split(text)

    def _update_spacy_model(self, spacy_model_name: str):
        self.spacy_model = spacy.load(spacy_model_name, exclude=["parser"])
        self.spacy_model.enable_pipe("senter")

    def _split_spacy(self, text: str, lang: Optional[str] = None) -> List[str]:
        # TODO: We may add an auto model selection for spaCy in the future.
        # However, the drawback is we will also need to
        # download a large number of spaCy models beforehand.
        processed_text = self.spacy_model(text)
        return [sent.text_with_ws for sent in processed_text.sents]

class TextSplitter:
    # Authors: Brian Rosenberg, Howard Huang

    def __init__(
        self, text: str, limit: int, num_boundary_chars: int,
        get_text_size: Callable[[str], int],
        sentence_model: TextSplitterModel,
        in_lang: Optional[str] = None) -> None:
        self._sentence_model = sentence_model
        self._limit = limit
        self._num_boundary_chars = num_boundary_chars
        self._get_text_size = get_text_size
        self._text = ""
        self._text_full_size = 0
        self._overhead_size = 0
        self._soft_limit = self._limit
        self._in_lang = in_lang

        if text:
            self.set_text(text)

    def set_text(self, text: str):
        self._text = text
        self._text_full_size = self._get_text_size(text)
        chars_per_size = len(text) / self._text_full_size
        self._overhead_size = self._get_text_size('')

        self._soft_limit = int(self._limit * chars_per_size) - self._overhead_size

        if self._soft_limit <= 1:
            # Caused by an unusually large overhead relative to text.
            # This is unlikely to occur except during testing of small text limits.
            # Recalculate soft limit by subtracting overhead from limit
            # before applying chars_per_size weighting.
            self._soft_limit = max(1,
                                   int((self._limit - self._overhead_size) * chars_per_size))

    def _isolate_largest_section(self, text:str) -> str:
        # Using cached word splitting model, isolate largest section of text
        string_length = len(text)

        if self._num_boundary_chars <= 0:
            num_chars_to_process = string_length
        else:
            num_chars_to_process = self._num_boundary_chars

        start_indx = max(0, string_length - num_chars_to_process)
        substring = text[start_indx: string_length]
        substring_list = self._sentence_model.split(substring, lang = self._in_lang)
        div_index = string_length - len(substring_list[-1])

        if div_index==start_indx:
            return text

        return text[0:div_index]

    @classmethod
    def split(cls,
              text: str, limit: int, num_boundary_chars: int, get_text_size: Callable[[str], int],
              sentence_model: TextSplitterModel,
              in_lang: Optional[str] = None
             ):
        return cls(text, limit, num_boundary_chars, get_text_size, sentence_model, in_lang)._split()


    def _split(self):
        if self._text_full_size <= self._limit:
            yield self._text
        else:
            yield from self._split_internal(self._text)

    def _split_internal(self, text):
        right = text
        while True:
            left, right = self._divide(right)
            yield left
            if not right:
                return

    def _divide(self, text) -> Tuple[str, str]:
        limit = self._soft_limit
        while True:
            left = text[:limit]
            left_size = self._get_text_size(left)

            if left_size <= self._limit:
                if left != text:
                    # If dividing into two parts
                    # Determine soft boundary for left segment
                    left = self._isolate_largest_section(left)
                return left, text[len(left):]

            char_per_size = len(left) / left_size


            limit = int(self._limit * char_per_size) - self._overhead_size

            if limit < 1:
            # Caused by an unusually large overhead relative to text.
            # This is unlikely to occur except during testing of small text limits.
            # Recalculate soft limit by subtracting overhead from limit before
            # applying chars_per_size weighting.
                limit = max(1, int((self._limit - self._overhead_size) * char_per_size))