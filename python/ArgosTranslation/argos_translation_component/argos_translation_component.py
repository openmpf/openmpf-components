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


from typing import Optional, Sequence, Dict, NamedTuple

import pathlib
import logging
import mpf_component_api as mpf
import mpf_component_util as mpf_util

from argostranslate import package, translate

from argos_translation_component.multi_language_processor import DetectedLangInfo, TranslationMetrics, MultiLanguageProcessor
from .argos_language_mapper import ArgosLanguageMapper

logger = logging.getLogger('ArgosTranslationComponent')

MULTI_LANG_REPORT = "MULTI_LANGUAGE_REPORT"
UNIDENTIFIED_LANGUAGE = 'UNKNOWN'

class ArgosTranslationCache(NamedTuple):
    source_lang: str
    target_lang: str
    text: str
class ArgosTranslationModel(NamedTuple):
    source_lang: str
    target_lang: str
    model: Optional[translate.ITranslation]

class ArgosTranslationComponent:

    def get_detections_from_video(self, job: mpf.VideoJob) -> Sequence[mpf.VideoTrack]:
        logger.info(f'Received video job.')
        return self.get_feed_forward_detections(job, job.feed_forward_track, video_job=True)

    def get_detections_from_image(self, job: mpf.ImageJob) -> Sequence[mpf.ImageLocation]:
        logger.info(f'Received image job.')
        return self.get_feed_forward_detections(job, job.feed_forward_location)

    def get_detections_from_audio(self, job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        logger.info(f'Received audio job.')
        return self.get_feed_forward_detections(job, job.feed_forward_track)

    @staticmethod
    def get_detections_from_generic(job: mpf.GenericJob) -> Sequence[mpf.GenericTrack]:
        logger.info(f'Received generic job.')
        if job.feed_forward_track:
            tw = TranslationWrapper(job.job_properties)
            tw.add_translations(job.feed_forward_track.detection_properties)
            return [job.feed_forward_track]
        else:
            logger.info('Job did not contain a feed forward track. Assuming '
                        'media file is a plain text file containing the text to '
                        'be translated.')
            text = pathlib.Path(job.data_uri).read_text().strip()

            new_job_props = {
                **job.job_properties,
                'FEED_FORWARD_PROP_TO_PROCESS': 'TEXT'
            }
            new_ff_props = dict(TEXT=text)
            ff_track = mpf.GenericTrack(detection_properties=new_ff_props)

            tw = TranslationWrapper(new_job_props)
            tw.add_translations(new_ff_props)

            return [ff_track]

    @staticmethod
    def get_feed_forward_detections(job, job_feed_forward, video_job=False):
        try:
            if job_feed_forward is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'Component can only process feed forward '
                    ' jobs, but no feed forward track provided. ')

            tw = TranslationWrapper(job. job_properties)
            tw.add_translations(job_feed_forward.detection_properties)

            if video_job:
                for ff_location in job.feed_forward_track.frame_locations.values():
                    tw.add_translations(ff_location.detection_properties)

            return [job_feed_forward]

        except Exception:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise


class TranslationWrapper:
    def __init__(self, job_props):
        self.supported_languages = self.get_supported_languages_codes()
        self._installed_languages = translate.get_installed_languages()
        self.installed_lang_codes = [lang.code for lang in self._installed_languages]
        # (Input Lang, Output Lang, Translation Model)
        self._cached_translation_model = ArgosTranslationModel("", "", None)

        self._props_to_translate = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=job_props,
                key='FEED_FORWARD_PROP_TO_PROCESS',
                default_value='TEXT,TRANSCRIPT',
                prop_type=str
            ).split(',')
        ]

        self._lang_prop_names = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=job_props,
                key='LANGUAGE_FEED_FORWARD_PROP',
                default_value='ISO_LANGUAGE,DECODED_LANGUAGE,LANGUAGE',
                prop_type=str
            ).split(',')
        ]

        self._script_prop_names = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=job_props,
                key='SCRIPT_FEED_FORWARD_PROP',
                default_value='ISO_SCRIPT',
                prop_type=str
            ).split(',')
        ]
        # Set a default language to translate from.
        self._from_lang = mpf_util.get_property(
            properties=job_props,
            key='DEFAULT_SOURCE_LANGUAGE',
            default_value='',
            prop_type=str
        ).lower().strip()

        self._from_script = mpf_util.get_property(
            properties=job_props,
            key='DEFAULT_SOURCE_SCRIPT',
            default_value='',
            prop_type=str
        ).lower().strip()

        if self._from_lang == "":
            logger.info("No default source language selected!")
        elif self._from_lang in ArgosLanguageMapper.iso_map:
            self._from_lang = ArgosLanguageMapper.get_code(self._from_lang, self._from_script)
        elif self._from_lang != "" and self._from_lang not in self.supported_languages:
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f"Default source language, {self._from_lang}, is not supported."
            )

        # TODO: Add support for non-English translations in the future.
        self._to_lang_word_separator = " "
        self._to_lang = "en"
        self._translation_cache: Dict[ArgosTranslationCache, str] = {}

    @staticmethod
    def get_supported_languages_codes():
        try:
            available_packages = package.get_available_packages()
        except Exception:
            logger.info("Downloading package index.")
            package.update_package_index()
            available_packages = package.get_available_packages()

        available_packages = [y.from_code for y in available_packages]
        return available_packages


    def _check_lang_script_codes(self, lang:str, script:str):
        if lang:
            if lang in self.supported_languages:
                return lang
            elif lang in ArgosLanguageMapper.iso_map:
                lang = ArgosLanguageMapper.get_code(lang, script)
                return lang
            else:
                raise mpf.DetectionError.DETECTION_FAILED.exception(
                    f"Source language, {lang}, is not supported."
                )
        else:
            raise mpf.DetectionError.MISSING_PROPERTY.exception(
                'None of the properties from "LANGUAGE_FEED_FORWARD_PROP" '
                f'({self._lang_prop_names}) were found in the feed forward track and no '
                '"DEFAULT_SOURCE_LANGUAGE" was provided.')

    def _check_installed_lang(self, source_lang:str):
        if source_lang not in self.installed_lang_codes:
            logger.info(f"Language {source_lang} is not installed. Installing package.")

            # From Argos Translate for downloading language models.
            available_packages = package.get_available_packages()
            available_package = list(
                filter(
                    lambda x: x.from_code == source_lang and x.to_code == self._to_lang, available_packages
                )
            )[0]

            download_path = available_package.download()
            package.install_from_path(download_path)

            logger.info(f"Successfully installed {source_lang}.")

            self.installed_lang_codes = [lang.code for lang in translate.get_installed_languages()]
            self._installed_languages = translate.get_installed_languages()

        # (Input Lang, Output Lang, Translation Model)
        if self._cached_translation_model.source_lang == source_lang and \
            self._cached_translation_model.target_lang == self._to_lang:
            return

        from_lang = list(filter(
            lambda x: x.code == source_lang,
            self._installed_languages))[0]
        to_lang = list(filter(
            lambda x: x.code == self._to_lang,
            self._installed_languages))[0]

        translation_model = from_lang.get_translation(to_lang)

        self._cached_translation_model = ArgosTranslationModel(source_lang, self._to_lang, translation_model)

    def _run_translation(self, source_lang: str, prop_to_translate: str, input_text:str) -> str:
        if source_lang == self._to_lang:
            logger.info(f'Skipped translation of the "{prop_to_translate}" '
            f'property because it was already in the target language.')
            return input_text

        if cached_translation := self._translation_cache.get(ArgosTranslationCache(source_lang, self._to_lang, input_text)):
            return cached_translation

        self._check_installed_lang(source_lang)

        if self._cached_translation_model.model is None:
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f"No valid translation model from {source_lang} to {self._to_lang}, "
                f"check if any packages are missing."
            )
        logger.info(f"Translating the {prop_to_translate} property.")
        translated_text = self._cached_translation_model.model.translate(input_text)
        trans_key = ArgosTranslationCache(source_lang, self._to_lang, input_text)
        self._translation_cache[trans_key] = translated_text
        logger.info("Translation complete.")
        return translated_text

    def _run_reports_through_translation(self,
                                         metrics: TranslationMetrics,
                                         prop_to_translate: str,
                                         input_text: str):
        for report in metrics.language_reports:
            sub_text = input_text[report.start_idx:report.end_idx]
            sub_lang = report.language
            if sub_lang == UNIDENTIFIED_LANGUAGE:
                logger.info(f'Text to translate contains an UNIDENTIFIED language '
                            f'for this section [{report.start_idx}-{report.end_idx}].')
                metrics.translations.append(sub_text)
                metrics.unknown_lang = True
                continue
            sub_script = report.script
            sub_source_lang = self._check_lang_script_codes(sub_lang, sub_script)
            translated_text = self._run_translation(sub_source_lang,
                                                    prop_to_translate,
                                                    sub_text)
            if translated_text == sub_text:
                logger.info(f'Text to translate unchanged '
                            f'for this section [{report.start_idx}-{report.end_idx}].')
                metrics.translations.append(sub_text)
            else:
                metrics.skipped_translation = False
                metrics.translations.append(translated_text)
                metrics.lang_text_count[sub_source_lang] += len(sub_text)
                metrics.lang_conf[sub_source_lang].append(report.conf)

    def add_translations(self, ff_props: Dict[str, str]):
        for prop_to_translate in self._props_to_translate:
            input_text = ff_props.get(prop_to_translate, None)
            if input_text:
                break
        else:
            logger.warning("No text to translate found in track.")
            return

        source_script = self._from_script
        for script_prop_name in self._script_prop_names:
            if script_prop_name in ff_props:
                source_script = ff_props.get(script_prop_name).lower().strip()
                break

        source_lang = self._from_lang
        for lang_prop_name in self._lang_prop_names:
            if lang_prop_name in ff_props:
                source_lang = ff_props.get(lang_prop_name).lower().strip()
                break

        metrics = TranslationMetrics()
        if text_language_report := ff_props.get(MULTI_LANG_REPORT):
            for report in text_language_report.split(";"):
                metrics.language_reports.append(MultiLanguageProcessor.extract_lang_report(report))
        else:
            metrics.language_reports.append(DetectedLangInfo(source_lang, source_script, 0, len(input_text), 1))

        self._run_reports_through_translation(metrics, prop_to_translate, input_text)
        MultiLanguageProcessor.aggregate_translation_results(metrics,
            prop_to_translate,
            self._to_lang,
            self._to_lang_word_separator,
            ff_props,
            logger)