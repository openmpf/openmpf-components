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

import collections
from typing import Dict, NamedTuple
import logging

class DetectedLangInfo(NamedTuple):
    language: str
    script: str
    start_idx: int
    end_idx: int
    conf: float

class TranslationMetrics:
    def __init__(self):
        self.language_reports = []
        self.translations = []
        self.lang_conf = collections.defaultdict(lambda: [])
        self.lang_text_count = collections.defaultdict(lambda: 0)
        self.skipped_translation = True
        self.unknown_lang = set()
        self.singleton = True

# TODO: Transfer improvements to other components.
class MultiLanguageProcessor:
    @staticmethod
    def extract_lang_report(report: str) -> DetectedLangInfo:
        """This helper function extracts the language, script, and start-end indexes
            of TEXT submitted with an associated Multiple Language Detection report

            The function is called whenever the MULTI_LANGUAGE_REPORT is
            present as a feed-forward property.

        Args:
            report (str): A single language-script report.
                Each report is formatted as follows:
                    `language-script, start-end indexes, prediction confidence`

                Ex. `eng-latin, 10-120, 0.998` corresponds to an english section of
                text starting at char index 10 and ending at char index 120.

        Returns:
            DetectedLangInfo: The language, script, start index, end index,
                and confidence of prediction respectively.
        """

        # Split `language-script, start-end indexes, prediction confidence` report into
        # respective parts for translation.
        report_list = report.split(",")
        language_info = report_list[0].split()[1]
        if '-' in language_info:
            (lang, script) = language_info.split("-")
        else:
            lang = language_info
            script = ""

        (start, end) = report_list[1].split()[1].split("-")
        conf = report_list[2].split()[1]

        return DetectedLangInfo(lang, script, int(start), int(end), float(conf))

    @staticmethod
    def aggregate_translation_results(metrics: TranslationMetrics,
                                      prop_to_translate:str,
                                      to_lang: str,
                                      to_lang_word_separator: str,
                                      detections: Dict[str, str],
                                      logger: logging.Logger,
                                      skip_aggregation: bool = False):

        if not skip_aggregation:
            if metrics.skipped_translation:
                if metrics.unknown_lang:
                    logger.info(f'Skipped translation of the "{prop_to_translate}" '
                    f'property.')
                else:
                    logger.info(f'Skipped translation of the "{prop_to_translate}" '
                                f'property because it was already in the target language.')
                detections['SKIPPED TRANSLATION'] = 'TRUE'
            else:
                main_source_lang = max(metrics.lang_text_count.items(), key=lambda x: x[1])[0]
                detections['TRANSLATION SOURCE LANGUAGE'] = main_source_lang
                detections['TRANSLATION TO LANGUAGE'] = to_lang
                detections['TRANSLATION'] = to_lang_word_separator.join(metrics.translations)
                detections['TRANSLATION SOURCE LANGUAGE CONFIDENCE'] = str(sum(metrics.lang_conf[main_source_lang])/\
                    len(metrics.lang_conf[main_source_lang]))

                logger.info(f'Successfully translated the "{prop_to_translate}" property.')
