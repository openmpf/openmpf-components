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

"""Swappable tokenizer backends for the CTranslate2 NLLB component.

CTranslate2 works in **token strings**, not ids, so this interface is
string-oriented: ``encode`` yields the piece lists handed to ``translate_batch``
and ``decode`` turns hypotheses back into text.

Two backends are provided, and they are functionally interchangeable:

* ``SentencePieceBackend`` (default) reads ``sentencepiece.bpe.model`` from the
  converted model directory.
* ``HuggingFaceBackend`` uses ``transformers.AutoTokenizer`` over the
  ``tokenizer.json`` trio in the same directory.

Both were verified against the same CTranslate2 vocabulary: all 256,000
SentencePiece pieces are present in the model's ``shared_vocabulary.json`` (0
missing), and the id spaces agree (``por_Latn`` is 256141 in both). Two benign
divergences are known and are asserted as *expected* by the tests rather than
treated as regressions:

1. Unknown characters. An em-dash is in neither vocabulary; SentencePiece emits
   the raw character as a surface piece while HuggingFace emits ``<unk>``.
   CTranslate2 resolves the SentencePiece form to ``<unk>`` anyway, so the model
   sees the same input.
2. Trailing whitespace. HuggingFace appends an extra ``▁`` that SentencePiece
   does not.
"""

import logging
import os
import time
from typing import List, Optional, Protocol, runtime_checkable

import mpf_component_api as mpf

logger = logging.getLogger('NllbTranslationComponent')

SENTENCEPIECE = 'SENTENCEPIECE'
HUGGINGFACE = 'HUGGINGFACE'

SP_MODEL_FILENAME = 'sentencepiece.bpe.model'
HF_TOKENIZER_FILENAME = 'tokenizer.json'

# The end-of-sequence token NLLB expects after the source pieces.
EOS_TOKEN = '</s>'


@runtime_checkable
class NllbTokenizerBackend(Protocol):
    """String-oriented tokenizer interface, shaped by what CTranslate2 consumes."""

    def encode(self, texts: List[str], src_lang: str) -> List[List[str]]:
        """Return one token-string list per input, ready for ``translate_batch``.

        The result must include the source-language token and ``</s>``.
        """

    def decode(self, token_lists: List[List[str]]) -> List[str]:
        """Turn hypothesis token-string lists back into text."""

    def count_tokens(self, text: str) -> int:
        """Token count as the model will see it, including the language and EOS tokens.

        Used by the sentence splitter, so it must match ``encode`` or chunks will
        be sized against the wrong budget.
        """


class SentencePieceBackend:
    """FLORES-200 SentencePiece. The default, and what the evaluation measured."""

    name = SENTENCEPIECE

    def __init__(self, model_dir: str) -> None:
        import sentencepiece as spm

        path = os.path.join(model_dir, SP_MODEL_FILENAME)
        if not os.path.isfile(path):
            raise mpf.DetectionException(
                f'SentencePiece model not found at {path}. The converted model '
                f'directory must include {SP_MODEL_FILENAME} '
                f'(ct2-transformers-converter --copy_files).',
                mpf.DetectionError.COULD_NOT_READ_DATAFILE)
        start = time.time()
        self._sp = spm.SentencePieceProcessor()
        self._sp.load(path)
        logger.debug(f'Loaded SentencePiece tokenizer from {path} '
                     f'in {time.time() - start:.2f}s.')

    def encode(self, texts: List[str], src_lang: str) -> List[List[str]]:
        return [[src_lang] + pieces + [EOS_TOKEN]
                for pieces in self._sp.encode_as_pieces(texts)]

    def decode(self, token_lists: List[List[str]]) -> List[str]:
        return self._sp.decode(token_lists)

    def count_tokens(self, text: str) -> int:
        # +2 for the source-language token and </s> that encode() adds.
        return len(self._sp.encode_as_pieces(text)) + 2


class HuggingFaceBackend:
    """transformers AutoTokenizer over the tokenizer files in the model directory.

    ``AutoTokenizer`` already emits the source-language prefix and ``</s>``, so
    unlike the SentencePiece backend nothing is added around its output.
    """

    name = HUGGINGFACE

    def __init__(self, model_dir: str) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise mpf.DetectionException(
                f'NLLB_TOKENIZER={HUGGINGFACE} requires the transformers package, '
                f'which is not installed: {e}. Use NLLB_TOKENIZER={SENTENCEPIECE} '
                f'or add transformers to the component environment.',
                mpf.DetectionError.COULD_NOT_READ_DATAFILE)

        if not os.path.isfile(os.path.join(model_dir, HF_TOKENIZER_FILENAME)):
            raise mpf.DetectionException(
                f'{HF_TOKENIZER_FILENAME} not found in {model_dir}. The converted '
                f'model directory must include the HuggingFace tokenizer files '
                f'(ct2-transformers-converter --copy_files).',
                mpf.DetectionError.COULD_NOT_READ_DATAFILE)

        start = time.time()
        self._auto_tokenizer = AutoTokenizer
        self._model_dir = model_dir
        self._src_lang: Optional[str] = None
        self._tok = None
        logger.debug(f'HuggingFace tokenizer backend ready for {model_dir} '
                     f'in {time.time() - start:.2f}s (loaded lazily per source language).')

    def _for_lang(self, src_lang: str):
        # AutoTokenizer bakes src_lang in at construction, so it is rebuilt when the
        # source language changes -- which is rare relative to the number of jobs.
        if self._tok is None or self._src_lang != src_lang:
            if self._tok is not None:
                logger.info(f'Tokenizer source language changed '
                            f'({self._src_lang} -> {src_lang}); re-initializing.')
            self._tok = self._auto_tokenizer.from_pretrained(
                self._model_dir, local_files_only=True, src_lang=src_lang)
            self._src_lang = src_lang
        return self._tok

    def encode(self, texts: List[str], src_lang: str) -> List[List[str]]:
        tok = self._for_lang(src_lang)
        return [tok.convert_ids_to_tokens(ids) for ids in tok(texts).input_ids]

    def decode(self, token_lists: List[List[str]]) -> List[str]:
        tok = self._for_lang(self._src_lang) if self._src_lang else self._tok
        return [tok.decode(tok.convert_tokens_to_ids(tokens), skip_special_tokens=True)
                for tokens in token_lists]

    def count_tokens(self, text: str) -> int:
        # Already includes the language token and </s>.
        return len(self._for_lang(self._src_lang or 'eng_Latn')(text).input_ids)


def create_backend(name: str, model_dir: str) -> NllbTokenizerBackend:
    """Build the requested backend, defaulting to SentencePiece on an unknown name."""
    requested = (name or SENTENCEPIECE).strip().upper()
    if requested == HUGGINGFACE:
        return HuggingFaceBackend(model_dir)
    if requested != SENTENCEPIECE:
        logger.warning(
            "Unrecognised NLLB_TOKENIZER '%s'; falling back to %s. Valid values: %s, %s.",
            name, SENTENCEPIECE, SENTENCEPIECE, HUGGINGFACE)
    return SentencePieceBackend(model_dir)
