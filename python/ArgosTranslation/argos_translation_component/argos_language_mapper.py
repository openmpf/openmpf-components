#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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

class ArgosLanguageMapper:
    # Supported languages for Argos Translate
    # Source link: https://www.argosopentech.com/argospm/index/

    # Note Malaysian, Slovak, and Romanian have two variants of ISO-639-2
    # https://www.loc.gov/standards/iso639-2/php/code_list.php
    iso_map = {
        "alb": "sq",
        "ara": "ar",
        "aze": "az",
        "ben": "bn",
        "bul": "bg",
        "cat": "ca",
        "cmn": "zh",
        "ces": "cs",
        "dan": "da",
        "nld": "nl",
        "epo": "eo",
        "est": "et",
        "fin": "fi",
        "fra": "fr",
        "deu": "de",
        "ell": "el",
        "heb": "he",
        "hin": "hi",
        "hun": "hu",
        "ind": "id",
        "gle": "ga",
        "ita": "it",
        "jpn": "ja",
        "kor": "ko",
        "lav": "lv",
        "lit": "lt",
        "may": "ms",
        "msa": "ms",
        "nor": "no",
        "fas": "fa",
        "pol": "pl",
        "por": "pt",
        "rum": "ro",
        "ron": "ro",
        "rus": "ru",
        "srp": "sr",
        "slk": "sk",
        "slo": "sk",
        "slv": "sl",
        "spa": "es",
        "swe": "sv",
        "tgl": "tl",
        "tha": "th",
        "tur": "tr",
        "ukr": "uk",
        "zho": "zh",
        "zho-hans": "zh",
        "zho-hant": "zt"
    }

    # Chinese has two scripts, traditional and simplified.
    _iso_lang_script = {
        "zho" : {"hans":"zh", "hant":"zt"}
    }

    @classmethod
    def get_code(cls, lang : str, script : str):
        # Resolve script for a specific language.
        if script and lang in cls._iso_lang_script:
            return cls._iso_lang_script[lang][script]
        return cls.iso_map[lang]