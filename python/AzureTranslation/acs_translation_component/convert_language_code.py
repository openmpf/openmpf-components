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
    #LTG = 'lv' # Latgalian language (Historical Form)

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
    #PES = 'fa' # Iranian Persian - Default below
    #PRS = 'fa' # Dari - Default Below
    TGK = 'fa', # Tajik

    PNB = 'pa', # Western Punjabi/Panjabi

    RUM = 'ro', # 639-2 Code Variant of Romanian

    SLO = 'sk', # 639-2 Code Variant of Slovak

    # Swahili Variants
    SWC='sw', # Congo Swahili
    SWH='sw', # Coastal Swahili
    #YMK='sw', # Makwe (?)
    #WMW='sw', # Mwani (?)

    TIB='bo', # 639-2 Code Variant of Tibetan

    UZN='uz', # Northern Uzbek
    UZS='uz', # Southern Uzbek

    WEL='cy',  # 639-2 Code Variant of Welsh
 )


# For some cases, we'll need to distinguish incoming script info
# As general practice these script codes are attached
# to the ISO-639.
ISO639_WITH_SCRIPT_TO_BCP47 = {
    "ZHO-HANS":"zh-hans",
    "ZHO-HANT":"zh-hant"
}

ISO6393_TO_BCP47 = dict(
    AFK='af',
    AMH='am',
    ARA='ar', # Note, Large number of variants
    ASM='as',
    AZE='az',
    BAK='bk',
    BEN='bn',
    BHO='bho', # Azure uses ISO code.
    BOD='bo',
    BOS='bs',
    BRX='brx', # Azure uses ISO code.
    BUL='bg',
    CAT='ca',
    CES='cs',
    ZHO='zh-hans', # Choosing to associate baseline Chinese as simplfied variant.
    ZH='zh-hans',# Choosing to associate baseline Chinese as simplfied variant.
    # Change to zh-hant if needed.
    CMN='zh-hans',
    CYM='cy',
    DAN='da', # Insular Danish
    DEU='de', # German, note: a lot of variants exist
    DIV='dv',
    DOI='doi', # Two other variants
    DSB='dsb',
    EUS='eu',
    ELL='el',
    ENG='en',
    EST='et', # Two other variants
    FAO='fo',
    FIJ='fj',
    FIL='fil',
    FIN='fi',
    FRA='fr',
    GLE='ga',
    GLG='gl',
    GUJ='gu',
    HAT='ht',
    HAU='ha',
    HEB='he', # Several archaic forms: https://en.wikipedia.org/wiki/Hebrew_language
    HIN='hi',
    HRV='hr',
    HSB='hsb',
    HUN='hu', # Old Hungarian also exists as code `OHU`
    HYE='hy',
    IBO='ig',
    IND='id',
    IKT='ikt',
    IKU='iu',
    ISL='is',
    ITA='it',
    JPN='ja',
    KAT='ka',
    KAN='kn',
    KAS='ks',
    KAZ='kk',
    KHM='km',
    KIN='rw',
    KIR='ky',
    LUG='lug',
    GOM='gom', # Goan Konkani, other two ISO variants redirected to this
    KOR='ko',
    KUR='ku',
    CKB='ku', # Azure noted Central Kurdish is supported as Ku
    KMR='kmr', # Northern Kurdish
    # There areNorthern two other variants of Kurdish but them don't seem to be directly supported.
    LAO='lo',
    LAV='lv',
    LIN='ln',
    LIT='lt', # There is an old Lithuanian variant (OLT)
    LZH='lzh',
    MAI='mai',
    MAL='ml',
    MAR='mr', # There's also an old variant Marathi variant (OMR)
    MKD='mk',
    MLG='mg', # Note: Many regional forms: https://en.wikipedia.org/wiki/Malagasy_language
    MLT='mt',
    MON = 'mn-cyrl', # Note: Azure also supports the traditional Mongolian script as `mn-Mong`
    # The primary script these days is Cyrllic/Latin.
    # From https://en.wikipedia.org/wiki/Mongolian_writing_systems:
    # "In March 2020, the Government of Mongolia announced plans to use the
    # traditional Mongolian script alongside Cyrillic in official documents starting from 2025."
    KHK='mn-cyrl', # Khalkha Mongolian
    MVF='mn-mong', # Peripheral Mongolian (part)
    MWW='mww',
    MRI='mi',
    MSA='ms', # Note: Many regional forms: https://en.wikipedia.org/wiki/Malay_language
    MYA='my', # Several variants exist.
    NAN='zh-hant',
    NEP='ne',
    NPI='ne',
    NLD='nl',
    NOR='no',
    NOB='no', # Two subtypes of Norwegian in active use.
    NNO='no',
    NYA='ny',
    NSO='nso',
    ORI='or', # Several variants exist.
    ORY='or',
    OTQ='otq',
    PAN='pa',
    PES='fa',
    POL='pl',
    POR='pt', # Defaulting to Portuguese (Brazil) - Other variant below
    PRS='prs',
    PUS='ps', # Several variants exist.
    RON='ro',
    RUN='rn',
    RUS='ru',
    SIN='si',
    SLK='sk',
    SLV='sl',
    SMO='sm', # Samoan Latin
    SOM='so',
    SOT='st',
    SRP='sr-Cyrl', # Note: Serbian language is fully digraphic, two popular script forms exist
    # Cyrillic is the official version adopted by Serbia
    SPA='es',
    SNA='sn', # Three other variants exist under Shona language.
    SND='sd',
    SQI='sq',
    SWA='sw',
    SWE='sv',
    TAH='ty',
    TAM='ta', # Old Tamil variant exists as OTY
    TAT='tt',
    TEL='te', # Related language: wbq – Waddar (Vadari), not included.
    THA='th',
    TIR='ti',
    TSN='tn',
    TUK='tk',
    TUR='tr',
    TON='to',
    UIG='ug',
    UKR='uk',
    URD='ur',
    UZB='uz',
    VIE='vi',
    XHO='xh',
    YOR='yo',
    YUE='yue',
    YUA='yua',
    ZUL='zu',
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
    if bcp_code := ISO639_WITH_SCRIPT_TO_BCP47.get(language_code.upper()):
        return bcp_code
    elif language_code.lower() in BCP_CODES:
        return language_code.lower()

    lang_code = language_code.upper().strip()

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
            f"Unable to find direct a BCP code match for {language_code}\n"
            f"Found a potential BCP match or variant: {bcp_code_var}\n"
            f"Using `{bcp_code_var}` as input language ")
        return bcp_code_var
    else:
        return None
