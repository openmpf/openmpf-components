#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
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
import logging
import pathlib
import re
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Callable, Dict, Iterator, List, Literal, Mapping, Match, NamedTuple, Optional, \
    Sequence, TypedDict, TypeVar, Union

import mpf_component_api as mpf
import mpf_component_util as mpf_util

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
                    f'Component can only process feed forward jobs, '
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
            modified_job_props = {**job.job_properties, 'FEED_FORWARD_PROP_TO_PROCESS': 'TEXT'}
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


class DetectResult(NamedTuple):
    primary_language: str
    primary_language_confidence: float
    alternative_language: Optional[str] = None
    alternative_language_confidence: Optional[float] = None


class TranslationResult(NamedTuple):
    translated_text: str
    detect_result: DetectResult


class SplitTextResult(NamedTuple):
    chunks: List[str]
    detected_language: Optional[str] = None
    detected_language_confidence: Optional[float] = None


NO_SPACE_LANGS = ('JA',  # Japanese
                  'YUE',  # Cantonese (Traditional)
                  'ZH-HANS',  # Chinese Simplified
                  'ZH-HANT')  # Chinese Traditional


class TranslationClient:
    DETECT_MAX_CHARS = 50_000

    def __init__(self, job_properties: Mapping[str, str]):
        self._subscription_key = get_required_property('ACS_SUBSCRIPTION_KEY', job_properties)

        url_builder = AcsTranslateUrlBuilder(job_properties)
        self._translate_url = url_builder.url
        self._to_language = url_builder.to_language.upper()
        self._provided_from_language = url_builder.from_language
        # Need to know the language's word separator in case the text needs to be split up in to
        # multiple translation requests. In that case we will need to combine the results from
        # each request.
        self._to_lang_word_separator = '' if self._to_language in NO_SPACE_LANGS else ' '

        self._newline_behavior = NewLineBehavior.get(job_properties)

        self._detect_before_translate = mpf_util.get_property(job_properties,
                                                              'DETECT_BEFORE_TRANSLATE', True)
        acs_url = get_required_property('ACS_URL', job_properties)
        self._detect_url = create_url(acs_url, 'detect', {})

        self._break_sentence_client = BreakSentenceClient(job_properties, self._subscription_key)

        prop_names = job_properties.get('FEED_FORWARD_PROP_TO_PROCESS', 'TEXT,TRANSCRIPT')
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
            detection_properties['TRANSLATION'] = translation_results.translated_text
            detection_properties['TRANSLATION TO LANGUAGE'] = self._to_language

            if detect_result := translation_results.detect_result:
                source_lang = detect_result.primary_language
                source_lang_confidence = str(detect_result.primary_language_confidence)
                if detect_result.alternative_language:
                    source_lang += f'; {detect_result.alternative_language}'
                    source_lang_confidence += f'; {detect_result.alternative_language_confidence}'

                detection_properties['TRANSLATION SOURCE LANGUAGE'] = source_lang

                detection_properties['TRANSLATION SOURCE LANGUAGE CONFIDENCE'] \
                    = source_lang_confidence

            log.info(f'Successfully translated the "{prop_name}" property.')
            self.translation_count += 1
            return  # Only process first matched property.


    def _translate_text(self, text: str) -> TranslationResult:
        """
        Translates the given text. If the text is longer than ACS allows, we will break up the
        text and translate each part separately. If, during the current job, we have seen the
        exact text before, we return a cached result instead of making a REST call.
        """
        if cached_translation := self._translation_cache.get(text):
            return cached_translation

        if self._provided_from_language:
            detect_result = DetectResult(self._provided_from_language, 1)
            from_lang, from_lang_confidence, *_ = detect_result
        elif self._detect_before_translate:
            detect_result = self._detect_language(text)
            if detect_result.alternative_language:
                from_lang = detect_result.alternative_language
                from_lang_confidence = detect_result.alternative_language_confidence
            else:
                from_lang, from_lang_confidence, *_ = detect_result
        else:
            detect_result = None
            from_lang = from_lang_confidence = None


        if from_lang and from_lang.casefold() == self._to_language.casefold():
            translation_info = TranslationResult('', DetectResult(from_lang, from_lang_confidence))
        else:
            text_replaced_newlines = self._newline_behavior(text, from_lang)
            grouped_sentences = self._break_sentence_client.split_input_text(
                text_replaced_newlines, from_lang, from_lang_confidence)
            if not detect_result and grouped_sentences.detected_language:
                detect_result = DetectResult(grouped_sentences.detected_language,
                                             grouped_sentences.detected_language_confidence)
            translation_info = self._translate_chunks(grouped_sentences, detect_result)

        self._translation_cache[text] = translation_info
        return translation_info


    def _translate_chunks(self, chunks: SplitTextResult,
                          detect_result: Optional[DetectResult]) -> TranslationResult:
        translated_text_chunks = []
        detected_lang = chunks.detected_language
        detected_lang_confidence = chunks.detected_language_confidence

        for chunk in chunks.chunks:
            response_body = self._send_translation_request(chunk, detected_lang)
            translated_text = response_body[0]['translations'][0]['text']
            translated_text_chunks.append(translated_text)
            if not detected_lang:
                if detected_lang_info := response_body[0].get('detectedLanguage'):
                    detected_lang = detected_lang_info['language']
                    detected_lang_confidence = detected_lang_info['score']

        combined_chunks = self._to_lang_word_separator.join(translated_text_chunks)
        if detect_result:
            return TranslationResult(combined_chunks, detect_result)

        else:
            return TranslationResult(combined_chunks,
                                     DetectResult(detected_lang, detected_lang_confidence))


    def _send_translation_request(self, text: str,
                                  from_language: Optional[str]) -> 'AcsResponses.Translate':
        if from_language:
            url = set_query_params(self._translate_url, {'from': from_language})
        else:
            url = self._translate_url
        request_body = [
            {'Text': text}
        ]
        encoded_body = json.dumps(request_body).encode('utf-8')
        request = urllib.request.Request(url, encoded_body,
                                         get_acs_headers(self._subscription_key))
        try:
            log.info(f'Sending POST to {url}')
            log_json(request_body)
            with urllib.request.urlopen(request) as response:
                response_body: AcsResponses.Translate = json.load(response)
                log.info(f'Received response from {url}.')
                log_json(response_body)
                return response_body
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'Request failed with HTTP status {e.code} and message: {response_content}') \
                from e


    def _detect_language(self, text: str) -> DetectResult:
        response = self._send_detect_request(text)
        primary_language = response[0]['language']
        primary_language_score = response[0]['score']
        log.info(f'Detected the primary language of the text was {primary_language} with score '
                 f'{primary_language_score}')

        if primary_language.casefold() == self._to_language.casefold():
            return self._handle_matching_from_and_to_lang(primary_language, primary_language_score,
                                                          response)
        elif response[0]['isTranslationSupported']:
            return DetectResult(primary_language, primary_language_score)
        else:
            return self._handle_untranslatable_primary_language(primary_language, response)


    @staticmethod
    def _handle_matching_from_and_to_lang(primary_language: str, primary_language_score: float,
                                          response: 'AcsResponses.Detect') -> DetectResult:
        alternatives = response[0].get('alternatives')
        if not alternatives:
            log.warning('Detected that the text is already in the requested TO_LANGUAGE and'
                        ' no alternatives were available. No translation will occur.')
            return DetectResult(primary_language, primary_language_score)

        alternative = next((alt for alt in alternatives if alt['isTranslationSupported']), None)
        if not alternative:
            log.warning('Detected that the text is already in the requested TO_LANGUAGE and'
                        ' none of the alternatives support translation. No translation will occur.')
            return DetectResult(primary_language, primary_language_score)

        alternative_lang = alternative['language']
        alternative_lang_score = alternative['score']
        log.warning('Detected that the text is already in the requested TO_LANGUAGE,'
                    f' but the highest score alternative was {alternative_lang} with'
                    f' score {alternative_lang_score}. Will attempt to translate text to'
                    f' {alternative_lang}.')
        return DetectResult(primary_language, primary_language_score, alternative_lang,
                            alternative_lang_score)


    @staticmethod
    def _handle_untranslatable_primary_language(
            primary_language: str, response: 'AcsResponses.Detect') -> DetectResult:
        alternatives = response[0].get('alternatives')
        if not alternatives:
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'The detected language, {primary_language}, does not support translation and '
                'there were no alternative languages.')

        alternative = next((alt for alt in alternatives if alt['isTranslationSupported']),
                           None)
        if not alternative:
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'The detected language, {primary_language}, does not support translation and '
                'none of the alternatives support translation.')

        alternative_lang = alternative['language']
        alternative_lang_score = alternative['score']
        log.info(f'The detected language, {primary_language}, does not support translation,'
                 f' but an alternative language, {alternative_lang} with score '
                 f'{alternative_lang_score}, does support translation. Will attempt to '
                 f'translate text to {alternative_lang}.')
        return DetectResult(alternative_lang, alternative_lang_score)


    def _send_detect_request(self, text) -> 'AcsResponses.Detect':
        request_body = [
            {'Text': text[:self.DETECT_MAX_CHARS]}
        ]
        encoded_body = json.dumps(request_body).encode('utf-8')
        request = urllib.request.Request(self._detect_url, encoded_body,
                                         get_acs_headers(self._subscription_key))
        try:
            log.info(f'Sending POST {self._detect_url}')
            log_json(request_body)
            with urllib.request.urlopen(request) as response:
                response_body: AcsResponses.Detect = json.load(response)
                log.info(f'Received response from {self._detect_url}.')
                log_json(response_body)
                return response_body
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'Request failed with HTTP status {e.code} and message: {response_content}') from e


class BreakSentenceClient:
    """
    Class to interact with Azure's "/breaksentence" endpoint. It is only used when the text to
    translate exceeds the translation endpoint's character limit.
    """

    # ACS limits the number of characters that can be translated in a single /translate call.
    # Taken from https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-translate
    TRANSLATION_MAX_CHARS = 10_000

    # ACS limits the number of characters that can be translated in a single /breaksentence call.
    # Taken from https://docs.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-break-sentence
    BREAK_SENTENCE_MAX_CHARS = 50_000


    def __init__(self, job_properties: Mapping[str, str], subscription_key: str):
        self._acs_url = get_required_property('ACS_URL', job_properties)
        self._subscription_key = subscription_key


    def split_input_text(self, text: str, from_lang: Optional[str],
                         from_lang_confidence: Optional[float]) -> SplitTextResult:
        """
        Breaks up the given text in to chunks that are under TRANSLATION_MAX_CHARS. Each chunk
        will contain one or more complete sentences as reported by the break sentence endpoint.
        """
        if len(text) <= self.TRANSLATION_MAX_CHARS:
            return SplitTextResult([text], from_lang, from_lang_confidence)

        log.info('Splitting input text because the translation endpoint allows a maximum of '
                 f'{self.TRANSLATION_MAX_CHARS} characters, but the text contained '
                 f'{len(text)} characters.')

        if len(text) > self.BREAK_SENTENCE_MAX_CHARS:
            log.warning('Guessing sentence breaks because the break sentence endpoint allows a '
                        f'maximum of {self.BREAK_SENTENCE_MAX_CHARS} characters, but the text '
                        f'contained {len(text)} characters.')
            chunks = list(SentenceBreakGuesser.guess_breaks(text))
            log.warning(f'Broke text up in to {len(chunks)} chunks. Each chunk will be sent to '
                        'the break sentence endpoint.')
        else:
            chunks = (text,)

        if from_lang:
            break_sentence_url = create_url(self._acs_url, 'breaksentence',
                                            dict(language=from_lang))
        else:
            break_sentence_url = create_url(self._acs_url, 'breaksentence', {})

        chunk_iter = iter(chunks)
        chunk = next(chunk_iter)
        response_body = self._send_break_sentence_request(break_sentence_url, chunk)
        if not from_lang:
            detected_lang_info = response_body[0].get('detectedLanguage')
            if detected_lang_info:
                from_lang = detected_lang_info['language']
                from_lang_confidence = detected_lang_info['score']
        grouped_sentences = list(self._process_break_sentence_response(chunk, response_body))

        for chunk in chunk_iter:
            response_body = self._send_break_sentence_request(break_sentence_url, chunk)
            grouped_sentences.extend(self._process_break_sentence_response(chunk, response_body))

        log.info('Grouped sentences into %s chunks.', len(grouped_sentences))
        return SplitTextResult(grouped_sentences, from_lang, from_lang_confidence)


    def _send_break_sentence_request(
            self, break_sentence_url: str, text: str) -> 'AcsResponses.BreakSentence':
        request_body = [
            {'Text': text}
        ]
        encoded_body = json.dumps(request_body).encode('utf-8')
        request = urllib.request.Request(break_sentence_url, encoded_body,
                                         get_acs_headers(self._subscription_key))
        try:
            log.info(f'Sending POST {break_sentence_url}')
            log_json(request_body)
            with urllib.request.urlopen(request) as response:
                response_body: AcsResponses.BreakSentence = json.load(response)
                log.info('Received break sentence response with %s sentences.',
                         len(response_body[0]['sentLen']))
                log_json(response_body)
                return response_body
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f'Request failed with HTTP status {e.code} and message: {response_content}') from e


    @classmethod
    def _process_break_sentence_response(
            cls, text: str, response_body: 'AcsResponses.BreakSentence') -> Iterator[str]:
        current_chunk_length = 0
        with io.StringIO(text) as sio:
            for length in response_body[0]['sentLen']:
                if length + current_chunk_length <= cls.TRANSLATION_MAX_CHARS:
                    current_chunk_length += length
                else:
                    yield sio.read(current_chunk_length)
                    current_chunk_length = length
            yield sio.read(current_chunk_length)


def create_url(base_url: str, path: str, query_params: Mapping[str, str]) -> str:
    url_parts = urllib.parse.urlparse(base_url)
    full_path = url_parts.path + '/' + path

    query_dict = urllib.parse.parse_qs(url_parts.query)
    query_dict.setdefault('api-version', ['3.0'])
    for k, v in query_params.items():
        query_dict.setdefault(k, [v])

    query_string = urllib.parse.urlencode(query_dict, doseq=True)
    replaced_parts = url_parts._replace(path=full_path, query=query_string)
    return urllib.parse.urlunparse(replaced_parts)


def set_query_params(url: str, query_params: Mapping[str, str]) -> str:
    url_parts = urllib.parse.urlparse(url)
    query_dict = urllib.parse.parse_qs(url_parts.query)
    for k, v in query_params.items():
        query_dict[k] = [v]
    query_string = urllib.parse.urlencode(query_dict, doseq=True)
    replaced_parts = url_parts._replace(query=query_string)
    return urllib.parse.urlunparse(replaced_parts)


class SentenceBreakGuesser:
    @classmethod
    def guess_breaks(cls, text: str) -> Iterator[str]:
        """
        Splits text up in to substrings that are all at most
        BreakSentenceClient.BREAK_SENTENCE_MAX_CHARS in length. It is preferable to use the
        /breaksentence endpoint because splitting a sentence in the middle will cause incorrect
        translations. When the input text is too long for /breaksentence, our only option is to
        use some heuristics to guess a good location to split the input text.
        We attempt to do the minimal number of splits with this method. The substrings produced
        by this method will be further split up using the much more accurate /breaksentence
        endpoint.

        :param text: Text to split up
        :return: Generator producing substrings of input text
        """
        current_pos = 0
        max_chars = BreakSentenceClient.BREAK_SENTENCE_MAX_CHARS
        while chunk := text[current_pos:current_pos + max_chars]:
            break_pos = cls._get_break_pos(chunk)
            current_pos += break_pos
            yield chunk[:break_pos]


    # Characters we know indicate the end of a sentence. The list is not exhaustive and may need to
    # be updated if we come across others.
    SENTENCE_END_PUNCTUATION = {
        '.', '!', '?',  # Latin scripts
        '。', '！', '？'}  # Chinese (full width) versions


    @classmethod
    def _get_break_pos(cls, text: str) -> int:
        # Two newlines in a row result in a blank line. Blank lines are commonly used to delimit
        # paragraphs.
        double_newline_pos = text.rfind('\n\n')
        if double_newline_pos > 0:
            return double_newline_pos + 2

        # Look for the last sentence breaking punctuation character in the text.
        last_punctuation_pos = next(
            (i for i in reversed(range(len(text))) if text[i] in cls.SENTENCE_END_PUNCTUATION),
            -1)
        if last_punctuation_pos > 0:
            return last_punctuation_pos + 1

        # Look for last punctuation character in the text.
        # This will catch non-sentence breaking punctuation, but we already made our best effort
        # to use sentence breaking punctuation above.
        last_punctuation_pos = next(
            (i for i in reversed(range(len(text))) if cls._is_punctuation(text[i])),
            -1)
        if last_punctuation_pos > 0:
            return last_punctuation_pos + 1

        if (last_space_pos := text.rfind(' ')) > 0:
            return last_space_pos + 1

        # No suitable break found. Use entire input.
        return len(text)

    @staticmethod
    def _is_punctuation(char):
        return unicodedata.category(char) == 'Po'


def get_acs_headers(subscription_key: str) -> Dict[str, str]:
    return {'Ocp-Apim-Subscription-Key': subscription_key,
            'Content-type': 'application/json; charset=UTF-8',
            'X-ClientTraceId': str(uuid.uuid4())}


class AcsTranslateUrlBuilder:
    url: str
    to_language: str
    from_language: Optional[str]

    def __init__(self, job_properties: Mapping[str, str]):
        base_url = get_required_property('ACS_URL', job_properties)
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


def get_required_property(property_name: str, job_properties: Mapping[str, str]) -> str:
    if property_value := job_properties.get(property_name):
        return property_value
    else:
        raise mpf.DetectionError.MISSING_PROPERTY.exception(
            f'The "{property_name}" property must be provided as a job property.')


class NewLineBehavior:
    """
    The Azure translation service treats newlines a separator between sentences. This results in
    incorrect translations. We can't simply replace newlines with spaces because not all languages
    put spaces between words. When testing with Chinese, spaces resulted in completely different
    translations.
    """
    @classmethod
    def get(cls, job_properties: Mapping[str, str]) -> Callable[[str, Optional[str]], str]:
        behavior = job_properties.get('STRIP_NEW_LINE_BEHAVIOR') or 'GUESS'
        if behavior == 'GUESS':
            return lambda s, l: cls._replace_new_lines(s, cls._guess_lang_separator(s, l))
        elif behavior == 'REMOVE':
            return lambda s, _: cls._replace_new_lines(s, '')
        elif behavior == 'SPACE':
            return lambda s, _: cls._replace_new_lines(s, ' ')
        elif behavior == 'NONE':
            return lambda s, _: s
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
    def _guess_lang_separator(text: str, language: Optional[str]) -> Literal['', ' ']:
        if language:
            if language.upper() in NO_SPACE_LANGS:
                return ''
            else:
                return ' '
        else:
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


    class _AlternativeDetectedLang(TypedDict):
        language: str
        score: float
        isTranslationSupported: bool
        isTransliterationSupported: bool

    class _DetectedLangWithAlts(TypedDict):
        language: str
        score: float
        isTranslationSupported: bool
        isTransliterationSupported: bool
        alternatives: Optional[List['AcsResponses._AlternativeDetectedLang']]

    Detect = List[_DetectedLangWithAlts]


def log_json(json_response):
    if log.isEnabledFor(logging.DEBUG):
        log.debug(json.dumps(json_response, indent=4, ensure_ascii=False))
