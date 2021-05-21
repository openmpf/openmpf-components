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
import pkg_resources
from symspellpy import SymSpell
from typing import Mapping, Sequence
import nltk
import os

import logging

log = logging.getLogger('NlpCorrectionComponent')


class NlpCorrectionComponent(object):
    detection_type = 'TEXT'
    initialized = False
    wrapper = None

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
                self.wrapper = SymspellWrapper(image_job.job_properties, image_job.job_name)
                self.initialized = True
            else:
                if custom_dictionary_path != \
                        self.wrapper.get_custom_dictionary_path():
                    self.wrapper = SymspellWrapper(image_job.job_properties, image_job.job_name)

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
                self.wrapper = SymspellWrapper(job.job_properties, job.job_name)
                self.initialized = True
            else:
                if custom_dictionary_path != \
                        self.wrapper.get_custom_dictionary_path():
                    self.wrapper = SymspellWrapper(job.job_properties, job.job_name)

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


class SymspellWrapper(object):

    def __init__(self, job_properties: Mapping[str, str], job_name):
        self._job_properties = job_properties

        self._sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        self._custom_dictionary_path = job_properties.get('CUSTOM_DICTIONARY', "")

        self._dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        self._bigram_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_bigramdictionary_en_243_342.txt")

        self._sym_spell.load_dictionary(self._dictionary_path, term_index=0, count_index=1)
        self._sym_spell.load_bigram_dictionary(self._bigram_path, term_index=0, count_index=0)

        # load custom dictionary if one is specified in the job properties
        if self._custom_dictionary_path != "":
            if os.path.exists(self._custom_dictionary_path):
                self._sym_spell.load_dictionary(self._custom_dictionary_path, term_index=0,
                                                count_index=1)
            else:
                log.exception(f'[{job_name}] '
                              f'Failed to complete job due incorrect file path for the custom dictionary:')
                raise mpf.DetectionException(
                    'Invalid path provided for custom dictionary',
                    mpf.DetectionError.COULD_NOT_READ_DATAFILE)

    # Adds corrected text to detection_properties.
    # Detection_properties is modified in place.
    def get_suggestions(self, original_text, detection_properties):
        log.info(f'Attempting to correct text {original_text}')

        print(self._sym_spell.words["the"])

        suggestions = ""
        punctuation = self.get_punctuation(original_text)

        split = original_text.rsplit("\n\n")
        split_cleaned = [x.rstrip("\n") for x in split if x != ""]

        # item for sublist in t for item in sublist
        tokenized_sentences = [nltk.sent_tokenize(x) for x in split_cleaned]
        tokenized_sentences = [x for sent in tokenized_sentences for x in sent]

        for sentence, punctuation in zip(tokenized_sentences, punctuation):
            corrected = self._sym_spell.lookup_compound(
                sentence, max_edit_distance=2, transfer_casing=True,
                ignore_non_words=True, ignore_term_with_digits=True)[0].term
            suggestions = suggestions + corrected + punctuation

        detection_properties["CORRECTED TEXT"] = suggestions
        log.info("Successfully corrected text")
        return True

    def get_custom_dictionary_path(self):
        return self._custom_dictionary_path

    def get_punctuation(self, original_text: str):
        split = original_text.rsplit("\n\n")

        punctuation = []
        newline_counter = 0

        first_block = True

        for block in reversed(split):
            first_sent = True
            newline = ""

            if block == '':
                newline_counter += 2
            else:
                newline_count = len(block) - len(block.rstrip("\n"))
                newline_string = "\n" * (newline_counter + newline_count)

                if first_block:
                    newline = newline_string
                    first_block = False
                else:
                    newline = "\n\n" + newline_string

                newline_counter = 0

            sentences = nltk.sent_tokenize(block)

            for sentence in reversed(sentences):
                if sentence[-1] not in "!?.":
                    if sentence[-1] in ")\"" and sentence[-2] in "!?.":
                        punc = sentence[-2]
                    else:
                        punc = " "
                else:
                    punc = sentence[-1]

                if first_sent:
                    punctuation.append(punc + newline)
                    first_sent = False
                else:
                    punctuation.append(punc + " ")

        return punctuation[::-1]