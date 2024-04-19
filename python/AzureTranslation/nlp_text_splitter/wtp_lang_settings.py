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

from typing import Optional

class WtpLanguageSettings:
    # Supported languages and ISO 639-1, 639-2 codes for WtP models.
    # https://github.com/bminixhofer/wtpsplit?tab=readme-ov-file#supported-languages
    # https://www.loc.gov/standards/iso639-2/php/code_list.php
    _wtp_lang_map = {
        'afrikaans': 'af',
        'afr': 'af',
        'amharic': 'am',
        'amh': 'am',
        'arabic': 'ar',
        'ara': 'ar',
        'azerbaijani': 'az',
        'aze': 'az',
        'belarusian': 'be',
        'bel': 'be',
        'bulgarian': 'bg',
        'bul': 'bg',
        'bengali': 'bn',
        'ben': 'bn',
        'catalan': 'ca',
        'valencian': 'ca',
        'cat': 'ca',
        'cebuano': 'ceb', # In some cases, ISO-639-1 is not available, use ISO-639-2
        'ceb': 'ceb',
        'czech': 'cs',
        'cze': 'cs',
        'ces': 'cs',
        'welsh': 'cy',
        'wel': 'cy',
        'cym': 'cy',
        'danish': 'da',
        'dan': 'da',
        'german': 'de',
        'ger': 'de',
        'deu': 'de',
        'greek': 'el',
        'gre': 'el',
        'ell': 'el',
        'english': 'en',
        'eng': 'en',
        'esperanto': 'eo',
        'epo': 'eo',
        'spanish': 'es',
        'castilian': 'es',
        'spa': 'es',
        'estonian': 'et',
        'est': 'et',
        'basque': 'eu',
        'baq': 'eu',
        'eus': 'eu',
        'persian': 'fa',
        'per': 'fa',
        'fas': 'fa',
        'finnish': 'fi',
        'fin': 'fi',
        'french': 'fr',
        'fre': 'fr',
        'fra': 'fr',
        'western frisian': 'fy',
        'fry': 'fy',
        'irish': 'ga',
        'gle': 'ga',
        'gaelic': 'gd',
        'scottish gaelic': 'gd',
        'gla': 'gd',
        'galician': 'gl',
        'glg': 'gl',
        'gujarati': 'gu',
        'guj': 'gu',
        'hausa': 'ha',
        'hau': 'ha',
        'hebrew': 'he',
        'heb': 'he',
        'hindi': 'hi',
        'hin': 'hi',
        'hungarian': 'hu',
        'hun': 'hu',
        'armenian': 'hy',
        'arm': 'hy',
        'hye': 'hy',
        'indonesian': 'id',
        'ind': 'id',
        'igbo': 'ig',
        'ibo': 'ig',
        'icelandic': 'is',
        'ice': 'is',
        'isl': 'is',
        'italian': 'it',
        'ita': 'it',
        'japanese': 'ja',
        'jpn': 'ja',
        'javanese': 'jv',
        'jav': 'jv',
        'georgian': 'ka',
        'geo': 'ka',
        'kat': 'ka',
        'kazakh': 'kk',
        'kaz': 'kk',
        'central khmer': 'km',
        'khm': 'km',
        'kannada': 'kn',
        'kan': 'kn',
        'korean': 'ko',
        'kor': 'ko',
        'kurdish': 'ku',
        'kur': 'ku',
        'kirghiz': 'ky',
        'kyrgyz': 'ky',
        'kir': 'ky',
        'latin': 'la',
        'lat': 'la',
        'lithuanian': 'lt',
        'lit': 'lt',
        'latvian': 'lv',
        'lav': 'lv',
        'malagasy': 'mg',
        'mlg': 'mg',
        'macedonian': 'mk',
        'mac': 'mk',
        'mkd': 'mk',
        'malayalam': 'ml',
        'mal': 'ml',
        'mongolian': 'mn',
        'mon': 'mn',
        'marathi': 'mr',
        'mar': 'mr',
        'malay': 'ms',
        'may': 'ms',
        'msa': 'ms',
        'maltese': 'mt',
        'mlt': 'mt',
        'burmese': 'my',
        'bur': 'my',
        'mya': 'my',
        'nepali': 'ne',
        'nep': 'ne',
        'dutch': 'nl',
        'flemish': 'nl',
        'dut': 'nl',
        'nld': 'nl',
        'norwegian': 'no',
        'nor': 'no',
        'panjabi': 'pa',
        'punjabi': 'pa',
        'pan': 'pa',
        'polish': 'pl',
        'pol': 'pl',
        'pushto': 'ps',
        'pashto': 'ps',
        'pus': 'ps',
        'portuguese': 'pt',
        'por': 'pt',
        'romanian': 'ro',
        'moldavian': 'ro',
        'moldovan': 'ro',
        'rum': 'ro',
        'ron': 'ro',
        'russian': 'ru',
        'rus': 'ru',
        'sinhala': 'si',
        'sinhalese': 'si',
        'sin': 'si',
        'slovak': 'sk',
        'slo': 'sk',
        'slk': 'sk',
        'slovenian': 'sl',
        'slv': 'sl',
        'albanian': 'sq',
        'alb': 'sq',
        'sqi': 'sq',
        'serbian': 'sr',
        'srp': 'sr',
        'swedish': 'sv',
        'swe': 'sv',
        'tamil': 'ta',
        'tam': 'ta',
        'telugu': 'te',
        'tel': 'te',
        'tajik': 'tg',
        'tgk': 'tg',
        'thai': 'th',
        'tha': 'th',
        'turkish': 'tr',
        'tur': 'tr',
        'ukrainian': 'uk',
        'ukr': 'uk',
        'urdu': 'ur',
        'urd': 'ur',
        'uzbek': 'uz',
        'uzb': 'uz',
        'vietnamese': 'vi',
        'vie': 'vi',
        'xhosa': 'xh',
        'xho': 'xh',
        'yiddish': 'yi',
        'yid': 'yi',
        'yoruba': 'yo',
        'yor': 'yo',
        'chinese': 'zh',
        'chi': 'zh',
        'zho': 'zh',
        'zulu': 'zu',
        'zul': 'zu',
        'hans':'zh', # Also check for chinese scripts
        'hant': 'zh',
        'cmn':'zh' # In some cases we use 'cmn' = 'Mandarin'
    }

    _wtp_iso_set = set(_wtp_lang_map.values())

    @classmethod
    def convert_to_iso(cls, lang: str) -> Optional[str]:
        # ISO 639-2 (language) is sometimes paired with ISO 15924 (script).
        # Extract the language portion and check if supported in WtP.
        if not lang:
            return None

        if '-' in lang:
            lang = lang.split('-')[0]
        if '_' in lang:
            lang = lang.split('_')[0]

        lang = lang.strip().lower()

        if lang in cls._wtp_iso_set:
            return lang

        if lang in cls._wtp_lang_map:
            return cls._wtp_lang_map[lang]

        return None
