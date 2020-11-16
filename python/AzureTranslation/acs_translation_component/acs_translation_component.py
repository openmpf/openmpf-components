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

import bisect
import io
import json
import os
import pathlib
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Dict, List, Literal, Mapping, Match, NamedTuple, Optional, \
    Sequence, TypedDict, TypeVar, Union

import mpf_component_api as mpf

log = mpf.configure_logging('acs-translation.log', __name__ == '__main__')


class AcsTranslationComponent:
    detection_type = 'TRANSLATION'

    @staticmethod
    def get_detections_from_video(job: mpf.VideoJob) -> Sequence[mpf.VideoTrack]:
        try:
            log.info(f'[{job.job_name}] Received video job: {job}')
            ff_track = job.feed_forward_track
            if ff_track is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'[{job.job_name}] Component can only process feed forward jobs, '
                    'but no feed forward track provided. ')

            tc = TranslationClient(job.job_properties)
            tc.add_translations(ff_track.detection_properties)
            for ff_location in ff_track.frame_locations.values():
                tc.add_translations(ff_location.detection_properties)

            log.info(f'[{job.job_name}] Processing complete. '
                     f'Translated {tc.translation_count} properties.')
            return (ff_track,)

        except Exception:
            log.exception(
                f'[{job.job_name}] Failed to complete job due to the following exception:')
            raise

    @staticmethod
    def get_detections_from_image(job: mpf.ImageJob) -> Sequence[mpf.ImageLocation]:
        return get_detections_from_non_composite(job, job.feed_forward_location)

    @staticmethod
    def get_detections_from_audio(job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        return get_detections_from_non_composite(job, job.feed_forward_track)

    @staticmethod
    def get_detections_from_generic(job: mpf.GenericJob) -> Sequence[mpf.GenericTrack]:
        if job.feed_forward_track:
            return get_detections_from_non_composite(job, job.feed_forward_track)
        else:
            log.info(f'[{job.job_name}] Job did not contain a feed forward track. Assuming media '
                     'file is a plain text file containing the text to be translated.')
            text = pathlib.Path(job.data_uri).read_text().strip()
            track = mpf.GenericTrack(detection_properties=dict(TEXT=text))
            modified_job_props = {**job.job_properties, 'TRANSLATE_PROPERTIES': 'TEXT'}
            modified_job = job._replace(job_properties=modified_job_props)
            return get_detections_from_non_composite(modified_job, track)


T_FF_OBJ = TypeVar('T_FF_OBJ', mpf.AudioTrack, mpf.GenericTrack, mpf.ImageLocation)

def get_detections_from_non_composite(
        job: Union[mpf.AudioJob, mpf.GenericJob, mpf.ImageJob],
        ff_track: Optional[T_FF_OBJ]) -> Sequence[T_FF_OBJ]:
    try:
        log.info(f'[{job.job_name}] Received job: {job}')
        if ff_track is None:
            raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                f'[{job.job_name}] Component can only process feed forward jobs, '
                'but no feed forward track provided. ')

        tc = TranslationClient(job.job_properties)
        tc.add_translations(ff_track.detection_properties)
        log.info(f'[{job.job_name}] Processing complete. '
                 f'Translated {tc.translation_count} properties.')
        return (ff_track,)

    except Exception:
        log.exception(
            f'[{job.job_name}] Failed to complete job due to the following exception:')
        raise


class TranslationResult(NamedTuple):
    translated_text: str
    detected_language: Optional[str]
    detected_language_confidence: Optional[float]

class SplitTextResult(NamedTuple):
    chunks: List[str]
    detected_language: Optional[str] = None
    detected_language_confidence: Optional[float] = None


NO_SPACE_LANGS = ('JA',  # Japanese
                  'YUE',  # Cantonese (Traditional)
                  'ZH-HANS',  # Chinese Simplified
                  'ZH-HANT')  # Chinese Traditional


class TranslationClient:
    def __init__(self, job_properties: Mapping[str, str]):
        self._subscription_key = get_property_or_env_value('ACS_SUBSCRIPTION_KEY', job_properties)

        url_builder = AcsTranslateUrlBuilder(job_properties)
        self._translate_url = url_builder.url
        self._to_language = url_builder.to_language.upper()
        # Need to know the language's word separator in case the text needs to be split up in to
        # multiple translation requests. In that case we will need to combine the results from
        # each request.
        self._to_lang_word_separator = '' if self._to_language in NO_SPACE_LANGS else ' '

        self._newline_behavior = NewLineBehavior.get(job_properties, url_builder.from_language)
        self._break_sentence_client = BreakSentenceClient(job_properties, self._subscription_key)

        prop_names = job_properties.get('TRANSLATE_PROPERTIES', 'TEXT,TRANSCRIPT')
        self._props_to_translate = [p.strip() for p in prop_names.split(',')]

        # Certain components that process videos and don't do tracking will create tracks with a
        # single detection. The track and detection will have the same set of properties, so
        # it is relatively likely we will run in to duplicate text.
        self._translation_cache: Dict[str, TranslationResult] = {}
        self.translation_count = 0


    def add_translations(self, detection_properties: Dict[str, str]) -> None:
        """
        Adds translations to detection_properties. detection_properties is modified in place.
        The set of properties in detection_properties to be translated is configured using
        job properties.
        """
        for prop_name in self._props_to_translate:
            text_to_translate = detection_properties.get(prop_name)
            if not text_to_translate:
                continue

            log.info(f'Attempting to translate the "{prop_name}" property...')
            translation_results = self._translate_text(text_to_translate)
            detection_properties[f'{prop_name} (TRANSLATION)'] = translation_results.translated_text
            detection_properties['TRANSLATION TO LANGUAGE'] = self._to_language

            if translation_results.detected_language:
                detection_properties[f'{prop_name} (SOURCE LANG)'] \
                    = translation_results.detected_language

                detection_properties[f'{prop_name} (SOURCE LANG CONFIDENCE)'] \
                    = str(translation_results.detected_language_confidence)

            log.info(f'Successfully translated the "{prop_name}" property.')
            self.translation_count += 1


    def _translate_text(self, text: str) -> TranslationResult:
        """
        Translates the given text. If the text is longer than ACS allows, we will break up the
        text and translate each part separately. If, during the current job, we have seen the
        exact text before, we return a cached result instead of making a REST call.
        """
        if cached_translation := self._translation_cache.get(text):
            return cached_translation

        text_replaced_newlines = self._newline_behavior(text)
        chunks = self._break_sentence_client.split_input_text(text_replaced_newlines)
        translation_info = self._translate_chunks(chunks)
        self._translation_cache[text] = translation_info
        return translation_info


    def _translate_chunks(self, chunks: SplitTextResult) -> TranslationResult:
        translated_text_chunks = []
        detected_lang = chunks.detected_language
        detected_lang_confidence = chunks.detected_language_confidence

        for chunk in chunks.chunks:
            response_body = self._send_translation_request(chunk)
            translated_text = response_body[0]['translations'][0]['text']
            translated_text_chunks.append(translated_text)
            if not detected_lang:
                if detected_lang_info := response_body[0].get('detectedLanguage'):
                    detected_lang = detected_lang_info['language']
                    detected_lang_confidence = detected_lang_info['score']

        combined_chunks = self._to_lang_word_separator.join(translated_text_chunks)
        return TranslationResult(combined_chunks, detected_lang, detected_lang_confidence)


    def _send_translation_request(self, text: str) -> 'AcsResponses.Translate':
        request_body = [
            {'Text': text}
        ]
        encoded_body = json.dumps(request_body).encode('utf-8')
        request = urllib.request.Request(self._translate_url, encoded_body,
                                         get_acs_headers(self._subscription_key))
        try:
            log.info(f'Sending POST to {self._translate_url}')
            with urllib.request.urlopen(request) as response:
                response_body: AcsResponses.Translate = json.load(response)
                log.info(f'Received response from {self._translate_url}.')
                return response_body
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'Request failed with HTTP status {e.code} and message: {response_content}') \
                from e



class BreakSentenceClient:
    """
    Class to interact with Azure's "/breaksentence" endpoint. It is only used when the text to
    translate exceeds the translation endpoint's character limit.
    """

    # ACS limits the number of characters that can be translated in a single REST call.
    # Taken from https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-translate
    TRANSLATION_MAX_CHARS = 10_000

    def __init__(self, job_properties: Mapping[str, str], subscription_key: str):
        self._break_sentence_url = self._get_break_sentence_url(job_properties)
        self._subscription_key = subscription_key


    def split_input_text(self, text: str) -> SplitTextResult:
        """
        Breaks up the given text in to chunks that are under TRANSLATION_MAX_CHARS. Each chunk
        will contain one or more complete sentences as reported by the break sentence endpoint.
        """
        if len(text) < self.TRANSLATION_MAX_CHARS:
            return SplitTextResult([text])

        log.info('Splitting input text because the translation endpoint allows a maximum of '
                 f'{self.TRANSLATION_MAX_CHARS} characters, but the text contained '
                 f'{len(text)} characters.')

        response_body = self._send_break_sentence_request(text)

        chunks = []
        current_chunk_length = 0
        with io.StringIO(text) as sio:
            for length in response_body[0]['sentLen']:
                if length + current_chunk_length < self.TRANSLATION_MAX_CHARS:
                    current_chunk_length += length
                else:
                    chunks.append(sio.read(current_chunk_length))
                    current_chunk_length = length
            chunks.append(sio.read(current_chunk_length))

        log.info('Grouped sentences in to %s chunks.', len(chunks))

        if detected_lang_info := response_body[0].get('detectedLanguage'):
            return SplitTextResult(chunks, detected_lang_info['language'],
                                   detected_lang_info['score'])
        else:
            return SplitTextResult(chunks)


    def _send_break_sentence_request(self, text: str) -> 'AcsResponses.BreakSentence':
        request_body = [
            {'Text': text}
        ]
        encoded_body = json.dumps(request_body).encode('utf-8')
        request = urllib.request.Request(self._break_sentence_url, encoded_body,
                                         get_acs_headers(self._subscription_key))
        try:
            log.info(f'Sending POST {self._break_sentence_url}')
            with urllib.request.urlopen(request) as response:
                response_body: AcsResponses.BreakSentence = json.load(response)
                log.info('Received break sentence response with %s sentences.',
                         len(response_body[0]['sentLen']))
                return response_body
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'Request failed with HTTP status {e.code} and message: {response_content}') from e


    @staticmethod
    def _get_break_sentence_url(job_properties: Mapping[str, str]) -> str:
        url = get_property_or_env_value('ACS_URL', job_properties)
        url_parts = urllib.parse.urlparse(url)
        path = url_parts.path + '/breaksentence'

        query_dict = urllib.parse.parse_qs(url_parts.query)
        query_dict.setdefault('api-version', ['3.0'])

        query_string = urllib.parse.urlencode(query_dict, doseq=True)
        replaced_parts = url_parts._replace(path=path, query=query_string)
        return urllib.parse.urlunparse(replaced_parts)



def get_acs_headers(subscription_key: str) -> Dict[str, str]:
    return {'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-type': 'application/json; charset=UTF-8',
            'X-ClientTraceId': str(uuid.uuid4())}


class AcsTranslateUrlBuilder:
    url: str
    to_language: str
    from_language: Optional[str]

    def __init__(self, job_properties: Mapping[str, str]):
        base_url = get_property_or_env_value('ACS_URL', job_properties)
        url_parts = urllib.parse.urlparse(base_url)

        query_dict: Dict[str, List[str]] = urllib.parse.parse_qs(url_parts.query)
        query_dict.setdefault('api-version', ['3.0'])

        self.to_language = (job_properties.get('TO_LANGUAGE')
                            or self.from_query_dict('to', query_dict)
                            or 'en')
        query_dict['to'] = [self.to_language]

        self.from_language = (job_properties.get('FROM_LANGUAGE')
                              or self.from_query_dict('from', query_dict))
        if self.from_language:
            query_dict['from'] = [self.from_language]

        if suggested_from := job_properties.get('SUGGESTED_FROM_LANGUAGE'):
            query_dict['suggestedFrom'] = [suggested_from]

        if category := job_properties.get('CATEGORY'):
            query_dict['category'] = [category]

        query_string = urllib.parse.urlencode(query_dict, doseq=True)
        path = url_parts.path + '/translate'
        replaced_parts = url_parts._replace(path=path, query=query_string)
        self.url = urllib.parse.urlunparse(replaced_parts)

    @staticmethod
    def from_query_dict(key: str, query_dict: Dict[str, List[str]]) -> Optional[str]:
        return query_dict.get(key, (None,))[0]


def get_property_or_env_value(property_name: str, job_properties: Mapping[str, str]) -> str:
    property_value = job_properties.get(property_name)
    if property_value:
        return property_value

    env_value = os.getenv(property_name)
    if env_value:
        return env_value

    raise mpf.DetectionError.MISSING_PROPERTY.exception(
        f'The "{property_name}" property must be provided as a job property or environment '
        'variable.')


class NewLineBehavior:
    """
    The Azure translation service treats newlines a separator between sentences. This results in
    incorrect translations. We can't simply replace newlines with spaces because not all languages
    put spaces between words. When testing with Chinese, spaces resulted in completely different
    translations.
    """

    @classmethod
    def get(cls,
            job_properties: Mapping[str, str],
            provided_from_language: Optional[str]) -> Callable[[str], str]:
        behavior = job_properties.get('STRIP_NEW_LINE_BEHAVIOR') or 'GUESS'
        if behavior == 'GUESS':
            if provided_from_language:
                if provided_from_language.upper() in NO_SPACE_LANGS:
                    return lambda s: cls._replace_new_lines(s, '')
                else:
                    return lambda s: cls._replace_new_lines(s, ' ')
            else:
                return lambda s: cls._replace_new_lines(s, cls._guess_lang_separator(s))
        elif behavior == 'REMOVE':
            return lambda s: cls._replace_new_lines(s, '')
        elif behavior == 'SPACE':
            return lambda s: cls._replace_new_lines(s, ' ')
        elif behavior == 'NONE':
            return lambda s: s
        else:
            raise mpf.DetectionError.INVALID_PROPERTY.exception(
                f'"{behavior}" is not a valid value for the "STRIP_NEW_LINE_BEHAVIOR" property. '
                'Valid value are GUESS, REMOVE, SPACE, NONE.')


    REPLACE_NEW_LINE_REGEX = re.compile(r'''
        \s? # Include preceding whitespace character if present
        (?<!\n) # Make sure previous character isn't a newline
        \n
        (?!\n) # Make sure next character isn't a newline
        \s? # Include next character if it is whitespace
        ''', flags=re.MULTILINE | re.IGNORECASE | re.UNICODE | re.DOTALL | re.VERBOSE)


    @classmethod
    def _replace_new_lines(cls, text: str, replacement: str) -> str:

        def do_replacement(match: Match[str]) -> str:
            match_text = match.group(0)
            if match_text == '\n':
                # Surrounding characters are not whitespace.
                return replacement
            else:
                # There is already whitespace next to newline character, so it can just be removed.
                return match_text.replace('\n', '', 1)

        return cls.REPLACE_NEW_LINE_REGEX.sub(do_replacement, text)

    @staticmethod
    def _guess_lang_separator(text: str) -> Literal['', ' ']:
        first_alpha_letter = next((ch for ch in text if ch.isalpha()), 'a')
        if ChineseAndJapaneseCodePoints.check_char(first_alpha_letter):
            return ''
        else:
            return ' '


class ChineseAndJapaneseCodePoints:
    # From http://www.unicode.org/charts/
    RANGES = sorted((
        range(0x2e80, 0x2fe0),
        range(0x2ff0, 0x3130),
        range(0x3190, 0x3300),
        range(0x3400, 0x4dc0),
        range(0x4e00, 0xa4d0),
        range(0xf900, 0xfb00),
        range(0xfe10, 0xfe20),
        range(0xfe30, 0xfe70),
        range(0xff00, 0xffa0),
        range(0x16f00, 0x16fa0),
        range(0x16fe0, 0x18d09),
        range(0x1b000, 0x1b300),
        range(0x1f200, 0x1f300),
        range(0x20000, 0x2a6de),
        range(0x2a700, 0x2ebe1),
        range(0x2f800, 0x2fa20),
        range(0x30000, 0x3134b)
    ), key=lambda r: r.start)

    RANGE_BEGINS = [r.start for r in RANGES]

    @classmethod
    def check_char(cls, char: str) -> bool:
        """
        Determine whether or not the given character is in the Unicode code point ranges assigned
        to Chinese and Japanese.
        """
        code_point = ord(char[0])
        if code_point < cls.RANGE_BEGINS[0]:
            return False
        else:
            idx = bisect.bisect_right(cls.RANGE_BEGINS, code_point)
            return code_point in cls.RANGES[idx - 1]


class AcsResponses:
    """
    The members of this class are used to type check the code that processes the JSON response
    from ACS. Things like response_body[0]['translations'][0]['text'] are easy to confuse with
    response_body[0]['translation']['text'] without type checking.
    """

    class _TranslationResult(TypedDict):
        text: str
        to: str

    class _DetectedLangInfo(TypedDict):
        language: str
        score: float

    class _TranslateTextInfo(TypedDict):
        translations: List['AcsResponses._TranslationResult']
        detectedLanguage: Optional['AcsResponses._DetectedLangInfo']

    Translate = List[_TranslateTextInfo]

    class _SentenceLengthInfo(TypedDict):
        sentLen: List[int]
        detectedLanguage: Optional['AcsResponses._DetectedLangInfo']

    BreakSentence = List[_SentenceLengthInfo]
