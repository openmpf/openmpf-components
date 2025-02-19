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


from typing import Sequence, Dict, Tuple

import pathlib
import logging
import mpf_component_api as mpf
import mpf_component_util as mpf_util

from argostranslate import package, translate

from .argos_language_mapper import ArgosLanguageMapper

logger = logging.getLogger('ArgosTranslationComponent')


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

        text = job.media_properties.get('SELECTED_CONTENT')
        if not text:
            logger.info('Job did not contain a feed forward track or specify selected content. '
                        'Assuming media file is a plain text file containing the text to '
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
                    'Component requires a feed forward track when processing video, image, and '
                    'audio jobs, but no feed forward track was provided.')

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


        # TODO: Add support for non-English translations in the future.
        self._to_lang = "en"

        self._translation_cache: Dict[str, Tuple[str, str]] = {}

    @staticmethod
    def get_supported_languages_codes():
        try:
            available_packages = package.get_available_packages()
        except Exception:
            logger.info("Downloading package index.")
            package.update_package_index()
            available_packages = package.get_available_packages()

        # TODO: Update if we want to support translations to non-English languages.
        available_packages = [y.from_code for y in list(
            filter(
                lambda x: x.to_code == "en", available_packages
            )
        )]
        return available_packages

    def add_translations(self, ff_props: Dict[str, str]):
        for prop_to_translate in self._props_to_translate:
            input_text = ff_props.get(prop_to_translate, None)
            if input_text:
                break
        else:
            logger.warning("No text to translate found in track.")
            return

        if cached_translation := self._translation_cache.get(input_text):
            ff_props['TRANSLATION'] = cached_translation[0]
            ff_props['TRANSLATION_SOURCE_LANGUAGE'] = cached_translation[1]
            return

        for script_prop_name in self._script_prop_names:
            if script_prop_name in ff_props:
                self._from_script = ff_props.get(script_prop_name).lower().strip()
                break

        for lang_prop_name in self._lang_prop_names:
            if lang_prop_name in ff_props:
                lang = ff_props.get(lang_prop_name).lower().strip()
                if lang in self.supported_languages:
                    self._from_lang = lang
                    break
                # TODO: Change if supporting non-English translations.
                elif lang in ('en','eng'):
                    ff_props['SKIPPED_TRANSLATION'] = 'TRUE'
                    logger.info(f'Skipped translation of the "{prop_to_translate}" '
                        f'property because it was already in the target language.')
                    return
                elif lang in ArgosLanguageMapper.iso_map:
                    # Convert supported languages to ISO-639-1
                    self._from_lang = ArgosLanguageMapper.get_code(lang, self._from_script)
                    break
                else:
                    raise mpf.DetectionError.DETECTION_FAILED.exception(
                        f"Source language, {lang}, is not supported."
                    )
        else:
            # Before converting to IS0-639-1, keep name of original default setting.
            source_lang_name = self._from_lang
            if self._from_lang in ArgosLanguageMapper.iso_map:
                self._from_lang = ArgosLanguageMapper.get_code(self._from_lang, self._from_script)
            if self._from_lang == 'en':
                ff_props['SKIPPED_TRANSLATION'] = 'TRUE'
                logger.info(f'Skipped translation of the "{prop_to_translate}" '
                            f'property because it was already in the target language.')
                return

            if self._from_lang == "":
                raise mpf.DetectionError.MISSING_PROPERTY.exception(
                    'None of the properties from "LANGUAGE_FEED_FORWARD_PROP" '
                    f'({self._lang_prop_names}) were found in the feed forward track and no '
                    '"DEFAULT_SOURCE_LANGUAGE" was provided.')

            if self._from_lang != "" and self._from_lang not in self.supported_languages:
                raise mpf.DetectionError.DETECTION_FAILED.exception(
                    f"Default source language, {source_lang_name}, is not supported."
                )

        if self._from_lang not in self.installed_lang_codes:
            logger.info(f"Language {self._from_lang} is not installed. Installing package.")

            # From Argos Translate for downloading language models.
            available_packages = package.get_available_packages()
            available_package = list(
                filter(
                    lambda x: x.from_code == self._from_lang and x.to_code == self._to_lang, available_packages
                )
            )[0]

            download_path = available_package.download()
            package.install_from_path(download_path)

            logger.info(f"Successfully installed {self._from_lang}.")

            self.installed_lang_codes = [lang.code for lang in translate.get_installed_languages()]
            self._installed_languages = translate.get_installed_languages()

        from_lang = list(filter(
            lambda x: x.code == self._from_lang,
            self._installed_languages))[0]
        to_lang = list(filter(
            lambda x: x.code == self._to_lang,
            self._installed_languages))[0]

        translation = from_lang.get_translation(to_lang)

        if translation is None:
            raise mpf.DetectionError.DETECTION_FAILED.exception(
                f"No valid translation model from {self._from_lang} to {self._to_lang}, "
                f"check if any packages are missing."
            )

        logger.info(f"Translating the {prop_to_translate} property.")

        translated_text = translation.translate(input_text)

        self._translation_cache[input_text] = (translated_text, self._from_lang)

        logger.info("Translation complete.")

        ff_props['TRANSLATION_SOURCE_LANGUAGE'] = self._from_lang
        ff_props['TRANSLATION'] = translated_text
