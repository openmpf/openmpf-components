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

import logging
from typing import Optional

import langcodes

logger = logging.getLogger('AcsTranslationComponent')

# A full list of supported languages can be found here:
# https://learn.microsoft.com/en-us/azure/ai-services/translator/language-support

# For some cases, we'll need to distinguish incoming script info
# As general practice these script codes are attached
# to the ISO-639.
ISO639_WITH_SCRIPT_TO_BCP47 = {
    "ZHO-HANS":"zh-hans",
    "ZHO-HANT":"zh-hant"
}

ISO6393_TO_BCP47 = dict(
    AFR='af', # Afrikaans
    AMH='am', # Amharic
    ARA='ar', # Note, Large number of variants for Arabic
    ASM='as', # Assamese
    AZE='az', # Azerbaijani
    BAK='ba', # Bashkir
    BEN='bn', # Bengali
    BHO='bho', # Azure uses ISO code for Bhojpuri.
    BOD='bo', # Tibetan
    BOS='bs', # Bosnian
    BRX='brx', # Azure uses ISO code for Boro/Bodo.
    BUL='bg', # Bulgarian
    CAT='ca', # Catalan, Valencian
    CES='cs', # Czech
    ZHO='zh-hans', # Choosing to associate baseline Chinese as simplfied variant.
    ZH='zh-hans',# Choosing to associate baseline Chinese as simplfied variant.
    # Change to zh-hant if needed.
    CMN='zh-hans', # Mandarin Chinese
    CYM='cy', # Welsh
    DAN='da', # Insular Danish
    DEU='de', # German, note: a lot of variants exist
    DIV='dv', # Divehi, Dhivehi, Maldivian
    DOI='doi', # Two other variants for Dogri.
    DSB='dsb', # Lower Sorbian
    EUS='eu', # Basque
    ELL='el', # Modern Greek
    ENG='en', # English
    EST='et', # Two other variants of Estonian
    FAO='fo', # Faroese
    FIJ='fj', # Fijian
    FIL='fil', # Filipino
    FIN='fi', # Finnish
    FRA='fr', # French
    GLE='ga', # Irish
    GLG='gl', # Galician
    GUJ='gu', # Gujarati
    HAT='ht', # Haitian, Haitian Creole
    HAU='ha', # Hausa
    HEB='he', # Several archaic forms: https://en.wikipedia.org/wiki/Hebrew_language
    HIN='hi', # Hindi
    HRV='hr', # Croatian
    HSB='hsb', # Upper Sorbian
    HUN='hu', # Hungarian. Note, Old Hungarian also exists as code `OHU`
    HYE='hy', # Armenian
    IBO='ig', # Igbo
    IND='id', # Indonesian
    IKT='ikt', # Inuinnaqtun
    IKU='iu', # Inuktitut
    ISL='is', # Icelandic
    ITA='it', # Italian
    JPN='ja', # Japanese
    KAT='ka', # Georgian
    KAN='kn', # Kannada
    KAS='ks', # Kashmiri
    KAZ='kk', # Kazakh
    KHM='km', # Central Khmer
    KIN='rw', # Kinyarwanda
    KIR='ky', # Kirghiz, Kyrgyz
    LUG='lug', # Luganda
    GOM='gom', # Goan Konkani, other two ISO variants redirected to this
    KOR='ko', # Korean
    KUR='ku', # Kurdish
    CKB='ku', # Azure noted Central Kurdish is supported as Ku
    KMR='kmr', # Northern Kurdish
    # There are two other variants of Kurdish but they don't seem to be directly supported.
    LAO='lo', # Lao
    LAV='lv', # Latvian
    LIN='ln', # Lingala
    LIT='lt', # Lithuanian. Note, there is an old Lithuanian variant (OLT)
    LZH='lzh', # Chinese (Literary)
    MAI='mai', # Maithili
    MAL='ml', # Malayalam
    MAR='mr', # Marathi. Note, there's also an old variant Marathi variant (OMR)
    MKD='mk', # Macedonian
    MLG='mg', # Malagasy. Note: Many regional forms: https://en.wikipedia.org/wiki/Malagasy_language
    MLT='mt', # Maltese
    MON = 'mn-cyrl', # Note: Azure also supports the traditional Mongolian script as `mn-Mong`
    # The primary script these days is Cyrllic/Latin.
    # From https://en.wikipedia.org/wiki/Mongolian_writing_systems:
    # "In March 2020, the Government of Mongolia announced plans to use the
    # traditional Mongolian script alongside Cyrillic in official documents starting from 2025."
    KHK='mn-cyrl', # Khalkha Mongolian
    MVF='mn-mong', # Peripheral Mongolian (part)
    MWW='mww', # Hmong Daw (Latin)
    MRI='mi', # Maori
    MSA='ms', # Note: Many regional forms: https://en.wikipedia.org/wiki/Malay_language
    MYA='my', # Burmese/Myanmar. Note, several variants exist.
    NAN='zh-hant', # Southern Min Chinese
    NEP='ne', # Nepali
    NPI='ne', # Nepali (individual language)
    NLD='nl', # Dutch, Flemish
    #NOR='nb', # Note: `nn`, `no`, and `nb` correspond to variants of Norwegian. But Azure
               # only supports `nb`.
    NOB='nb', # Norwegian Bokmål (Norway)
    #NNO='nb', # Norwegian Nynorsk
    NYA='nya', # `Azure uses ISO-639-3 variant for Nyanja instead of ISO-639-2
    NSO='nso', # Sesotho sa Leboa
    ORI='or', # Several variants exist for Odia
    ORY='or', # Odia
    OTQ='otq', # Queretaro Otomi
    PAN='pa', # Punjabi, Panjabi
    PES='fa', # Iranian Persian
    POL='pl', # Polish
    POR='pt', # Defaulting to Portuguese (Brazil) - Other variant below
    PRS='prs', # Dari
    PUS='ps', # Several variants exist for Pashto, Pushto
    RON='ro', # Romanian, Moldavian, Moldovan
    RUN='run', # Azure uses ISO-639-3 instead of ISO-639-2 for Rundi.
    RUS='ru', # Russian
    SIN='si', # Sinhala, Sinhalese
    SLK='sk', # Slovak
    SLV='sl', # Slovenian
    SMO='sm', # Samoan Latin
    SOM='so', # Somali
    SOT='st', # Southern Sotho
    SRP='sr-Cyrl', # Note: Serbian language is fully digraphic, two popular script forms exist
    # Cyrillic is the official version adopted by Serbia
    SPA='es', # Spanish
    SNA='sn', # Three other variants exist under Shona language.
    SND='sd', # Sindhi
    SQI='sq', # Albanian
    SWA='sw', # Swahili
    SWE='sv', # Swedish
    TAH='ty', # Tahitian
    TAM='ta', # Tamil. Note, Old Tamil variant exists as OTY
    TAT='tt', # Tatar
    TEL='te', # Telugu. Related language: wbq – Waddar (Vadari), not included.
    THA='th', # Thai
    TIR='ti', # Tigrinya
    TSN='tn', # Tswana
    TUK='tk', # Turkmen
    TUR='tr', # Turkish
    TON='to', # Tonga (Tonga Islands)
    UIG='ug', # Uighur, Uyghur
    UKR='uk', # Ukrainian
    URD='ur', # Urdu
    UZB='uz', # Uzbek
    VIE='vi', # Vietnamese
    XHO='xh', # Xhosa
    YOR='yo', # Yoruba
    YUE='yue', # Cantonese (Traditional)
    YUA='yua', # Yucatec Maya
    ZUL='zu', # Zulu
)

# These cover conflicting 639-2 codes and less common variants
# A warning will be issued if these are used.
ISO639_VAR_TO_BCP47 = dict(
    # ISO Code Variant = Same language but two ISO-639 codes match to it.
    # Variant = May be different from primary ISO code.
    # Note: We're avoiding adding in archaic/extinct variants, attaching a note to
    # languages with those present.

    ALB = 'sq', # 639-2 Code Variant
    ARM = 'hy', # 639-2 Code Variant
    AZJ = 'az', # North Azerbaijani Variant
    AZB = 'az', # South Azerbaijani Variant
    BAQ = 'eu', # 639-2 Code Variant
    CZE = 'cs', # 639-2 Code Variant Czech

    TWL = 'sn', # 639-3 Variants of Shona
    MXC = 'sn', # 639-3 Variants of Shona
    TWX = 'sn', # 639-3 Variants of Shona

    JUT = 'da', # 639-3: Jutish - Danish Dialect

    DGO = 'doi', # Dogri proper
    XNR = 'doi', # Kangri

    DUT = 'nl', # 639-2 Code Variant of Dutch

    EKK = 'et', # Standard Estonian
    VRO = 'et', # Võro - Estonian Dialect (Debated)

    FRE = 'fr', # 639-2 Code Variant of French

    GEO = 'ka', # 639-2 Code Variant of Georgian

    GER = 'de', # 639-2 Code Variant of German
    # Warning: There's many other variants of old and regional forms.
    # https://en.wikipedia.org/wiki/German_language

    GRE = 'el', # 639-2 Code Variant of Greek
    # Note: There's several other variants
    # https://en.wikipedia.org/wiki/Greek_language

    ICE = 'is', # 639-2 Code Variant of Icelandic

    IKE = 'iu', # Eastern Canadian Inuktitut

    KXM = 'km', # Northern Khmer

    KOK = 'gom', # Kokani
    KNN = 'gom', # Maharashtrian Konkani

    TTS = 'lo', # Isan (Thailand Lao)

    LVS = 'lv', # Standard Latvian language
    # LTG = 'lv' # Latgalian language (Historical Form)

    MAC = 'mk', # 639-2 Code Variant of Macedonian

    MAY = 'ms', # 639-2 Code Variant of Malay
    ZLM = 'ms', # Malay (individual language)
    ZSM = 'ms', # Malaysian Malay

    MAO = 'mi', # 639-2 Code Variant of Maori

    BUR = 'my', # 639-2 Code Variant of Burmese (Myanmar)
    # Several other closely related Burmese variants exist below:
    INT = 'my', # Intha
    TCO = 'my', # Taungyo
    RKI = 'my', # Rakhine
    RMZ = 'my', # Marma
    TAY = 'my', # Tavoyan dialects

    # Variants of Odia
    SPV = 'or', # Sambalpuri
    ORT = 'or', # Adivasi Odia (Kotia)
    DSO = 'or', # Desiya


    # Variants of Pashto
    PST = 'ps', # Central Pashto
    PBU = 'ps', # Northern Pashto
    PBT = 'ps', # Southern Pashto
    # WNE - Archaic

    # Persian has many variants
    # Only including top three
    # PES = 'fa' # Iranian Persian - Default below
    # PRS = 'fa' # Dari - Default Below
    PER = 'fa', # Persian (Inclusive)
    FAS = 'fa', # Persian (Inclusive)
    TGK = 'fa', # Tajik

    PNB = 'pa', # Western Punjabi/Panjabi

    RUM = 'ro', # 639-2 Code Variant of Romanian

    SLO = 'sk', # 639-2 Code Variant of Slovak

    # Swahili Variants
    SWC='sw', # Congo Swahili
    SWH='sw', # Coastal Swahili
    # YMK='sw', # Makwe (?)
    # WMW='sw', # Mwani (?)

    TIB='bo', # 639-2 Code Variant of Tibetan

    UZN='uz', # Northern Uzbek
    UZS='uz', # Southern Uzbek

    WEL='cy',  # 639-2 Code Variant of Welsh
 )

BCP_CODES_ONLY = {
    'iu-latn', # Inuktitut (Latin)
    'fr-ca', # French Canadian
    'tlh-latn', # Klingon Latin
    'tlh-piqd', # Klingon (plqaD)
    'mn-cyrl', # Mongolian (Cyrllic)
    'mn-mong', # Mongolian (Traditional)
    'pt-pt', # Portuguese  (Portugal)
    'sr-latn', # Serbian (Latin)
}


BCP_CODES = BCP_CODES_ONLY | \
            set(ISO6393_TO_BCP47.values()) | \
            set(ISO639_WITH_SCRIPT_TO_BCP47.values())

def iso_to_bcp(language_code: str) -> Optional[str]:
    # First check if we have matching scripts/regional variants
    language_code = language_code.strip()
    if bcp_code := ISO639_WITH_SCRIPT_TO_BCP47.get(language_code.upper()):
        return bcp_code
    elif language_code.lower() in BCP_CODES:
        return language_code.lower()

    lang_code = language_code.upper()

    # Remove attached script/variant info,
    # Check language portion of ISO code next.
    if '-' in lang_code:
        lang_code = lang_code.split('-')[0]
    if bcp_code := ISO6393_TO_BCP47.get(lang_code):
        return bcp_code
    elif lang_code.lower() in BCP_CODES:
        return lang_code.lower()
    elif lang_info := langcodes.get(lang_code):
        # TODO, after langcodes conversion, we may want to consider double checking the BCP codes again
        # discard if value does not match supported codes.
        return lang_info.language
    elif bcp_code_var := ISO639_VAR_TO_BCP47.get(lang_code):
        logger.warning(
            f"Unable to find direct a BCP code match for {language_code}. "
            f"Found a potential BCP match or variant: {bcp_code_var}. "
            f"Using `{bcp_code_var}` as input language.")
        return bcp_code_var
    else:
        return None
