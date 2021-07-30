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
from typing import Iterable

import mpf_component_api as mpf
from typing import Mapping, Sequence
from hunspell import Hunspell
import re
import os
import string
import logging

log = logging.getLogger('NlpCorrectionComponent')


class NlpCorrectionComponent(object):
    detection_type = 'TEXT'

    def __init__(self):
        self.initialized = False
        self.wrapper = None

    def get_detections_from_image(self, image_job: mpf.ImageJob) -> Sequence[mpf.ImageLocation]:
        try:
            ff_track = image_job.feed_forward_location
            detection_properties = ff_track.detection_properties

            if ff_track is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'Component can only process feed forward image jobs, '
                    'but no feed forward track provided. '
                )

            text = detection_properties.get("TEXT")
            custom_dictionary_path = image_job.job_properties.get('CUSTOM_DICTIONARY', "")

            if not self.initialized:
                self.wrapper = HunspellWrapper(image_job.job_properties, image_job.job_name)
                self.initialized = True
            else:
                if custom_dictionary_path != \
                        self.wrapper.get_custom_dictionary_path():
                    self.wrapper = HunspellWrapper(image_job.job_properties, image_job.job_name)

            self.wrapper.get_suggestions(text, detection_properties)

            log.info(f'[{image_job.job_name}] Processing complete.')
            return ff_track,

        except Exception:
            log.exception(
                f'[{image_job.job_name}] Failed to complete job due to the following exception:'
            )
            raise

    def get_detections_from_generic(self, job: mpf.GenericJob) -> Iterable[mpf.GenericTrack]:
        try:
            log.info(f'[{job.job_name}] Received generic job: {job}')
            ff_track = job.feed_forward_track

            custom_dictionary_path = job.job_properties.get('CUSTOM_DICTIONARY', "")

            if not self.initialized:
                self.wrapper = HunspellWrapper(job.job_properties, job.job_name)
                self.initialized = True
            else:
                if custom_dictionary_path != \
                        self.wrapper.get_custom_dictionary_path():
                    self.wrapper = HunspellWrapper(job.job_properties, job.job_name)

            if ff_track is not None:
                detection_properties = ff_track.detection_properties
                text = detection_properties.get("TEXT")

                self.wrapper.get_suggestions(text, detection_properties)

                return ff_track,

            else:
                with open(job.data_uri, "r") as f:
                    text = f.read()

                f.close()

                detection_properties = dict(TEXT=text)

                self.wrapper.get_suggestions(text, detection_properties)
                generic_track = mpf.GenericTrack(detection_properties=detection_properties)

                log.info(f'[{job.job_name}] Processing complete.')
                return generic_track,

        except Exception:
            log.exception(
                f'[{job.job_name}] Failed to complete job due to the following exception:')
            raise


class HunspellWrapper(object):

    def __init__(self, job_properties: Mapping[str, str], job_name):
        self._job_properties = job_properties

        self._hunspell = Hunspell('en_US')
        self._hunspell.clear_cache()

        self._unicode_error = False

        self._full_output = job_properties.get("FULL_TEXT_CORRECTION_OUTPUT", False)

        self._custom_dictionary_path = job_properties.get('CUSTOM_DICTIONARY', "")

        # load custom dictionary if one is specified in the job properties
        if self._custom_dictionary_path != "":
            if os.path.exists(self._custom_dictionary_path):
                self._hunspell.add_dic(self._custom_dictionary_path)
            else:
                log.exception(f'[{job_name}] '
                              f'Failed to complete job due incorrect file path for the custom dictionary: '
                              f'"{self._custom_dictionary_path}"')
                raise mpf.DetectionException(
                    f'Invalid path provided for custom dictionary: '
                    f'"{self._custom_dictionary_path}"',
                    mpf.DetectionError.COULD_NOT_READ_DATAFILE)

    # Adds corrected text to detection_properties.
    # Detection_properties is modified in place.
    def get_suggestions(self, original_text, detection_properties):
        log.debug(f'Attempting to correct text: "{original_text}"')
        self._unicode_error = False
        unicode_error_words = []

        def repl(match: re.Match) -> str:
            spacer = match.group(1)
            leading_punc = match.group(2)
            word = match.group(3)
            trailing_punc = match.group(4)

            try:
                if self._hunspell.spell(word):
                    return spacer + leading_punc + word + trailing_punc

                word_no_inner_punc = word.translate(str.maketrans("", "", string.punctuation))
                if self._hunspell.spell(word_no_inner_punc):
                    return spacer + leading_punc + word_no_inner_punc + trailing_punc

                temp = self._hunspell.suggest(word)
                if len(temp) > 0:
                    if self._full_output:
                        corrected_text = '[' + ', '.join(temp) + ']'
                    else:
                        corrected_text = temp[0]
                else:
                    corrected_text = word

            except UnicodeEncodeError:
                self._unicode_error = True
                corrected_text = word

                if len(unicode_error_words) < 3:
                    unicode_error_words.append(word)

            return spacer + leading_punc + corrected_text + trailing_punc

        punc = r'!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'  # based on string.punctuation.

        suggestions = re.sub(fr'(\s|^)([{punc}]*)([^\s]*\w)([{punc}]*)', repl, original_text)

        if self._unicode_error:
            log.warning(f'Encountered words with unsupported characters. '
                        f'The first three (or less) words with the issue are:'
                        f' ' + ' '.join(map(str, unicode_error_words)))

        detection_properties["CORRECTED TEXT"] = suggestions
        log.debug("Successfully corrected text")
        log.debug(f'Corrected text: "{suggestions}"')
        return True

    def get_custom_dictionary_path(self):
        return self._custom_dictionary_path
