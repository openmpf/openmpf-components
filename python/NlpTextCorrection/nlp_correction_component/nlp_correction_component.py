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
import os
from pathlib import Path

import logging

log = logging.getLogger('NlpCorrectionComponent')


class NlpCorrectionComponent(object):
    detection_type = 'TEXT'

    @staticmethod
    def get_detections_from_image(image_job: mpf.ImageJob) -> Sequence[mpf.ImageLocation]:
        try:
            ff_track = image_job.feed_forward_location
            detection_properties = ff_track.detection_properties

            if ff_track is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'Component can only process feed forward image jobs, '
                    'but no feed forward track provided. '
                )

            text = detection_properties.get("TEXT")

            runner = JobRunner(image_job.job_properties)
            runner.get_suggestions(text, detection_properties)

            log.info(f'[{image_job.job_name}] Processing complete.')
            return ff_track,

        except Exception:
            log.exception(
                f'[{image_job.job_name}] Failed to complete job due to the following exception:'
            )
            raise

    @staticmethod
    def get_detections_from_generic(job: mpf.GenericJob) -> Iterable[mpf.GenericTrack]:
        try:
            log.info(f'[{job.job_name}] Received generic job: {job}')
            ff_track = job.feed_forward_track

            if ff_track is not None:
                detection_properties = ff_track.detection_properties
                text = detection_properties.get("TEXT")

                runner = JobRunner(job.job_properties)
                runner.get_suggestions(text, detection_properties)

                return ff_track,

            else:
                with open(job.data_uri, "r") as f:
                    text = f.read()

                f.close()

                detection_properties = dict(TEXT=text)

                runner = JobRunner(job.job_properties)
                runner.get_suggestions(text, detection_properties)

                generic_track = mpf.GenericTrack(detection_properties=detection_properties)

                log.info(f'[{job.job_name}] Processing complete.')
                return generic_track,

        except Exception:
            log.exception(
                f'[{job.job_name}] Failed to complete job due to the following exception:')
            raise


class JobRunner(object):

    def __init__(self, job_properties: Mapping[str, str]):
        self._job_properties = job_properties
        self._sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        custom_dictionary_path = job_properties.get('CUSTOM_DICTIONARY', "")

        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")

        self._sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self._sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=0)

        curr_dir = Path(__file__).absolute().parent.parent

        # load custom dictionary if one is specified in the job properties
        if custom_dictionary_path:
            full_custom_dict_path = os.path.join(curr_dir, "plugin-files/config/" + custom_dictionary_path)
            self._sym_spell.load_dictionary(full_custom_dict_path, term_index=0, count_index=1)

    # Adds corrected text to detection_properties.
    # Detection_properties is modified in place.
    def get_suggestions(self, original_text: str, detection_properties):
        log.debug(f'Attempting to correct text: \"{original_text}\"')
        suggestions = self._sym_spell.lookup_compound(
            original_text, max_edit_distance=2)

        detection_properties["CORRECTED TEXT"] = suggestions[0].term
        log.debug(f'Successfully corrected text to: \"{suggestions[0].term}\"')
        return suggestions[0].term
