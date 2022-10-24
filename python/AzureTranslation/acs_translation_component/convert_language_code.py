#!/usr/bin/env python3
#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
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

import langcodes

ISO6393_TO_BCP47 = dict(
    AMH='am',
    ARA='ar',
    AZE='az',
    BOD='bo',
    BUL='bg',
    CES='cs',
    CMN='zh-hans',
    ELL='el',
    ENG='en',
    FRA='fr',
    HAT='ht',
    HIN='hi',
    HYE='hy',
    IND='id',
    JPN='ja',
    KAT='ka',
    KAZ='kk',
    KIR='ky',
    KOR='ko',
    KUR='ku',
    LAO='lo',
    LIT='lt',
    MKD='mk',
    MYA='my',
    NAN='zh-hant',
    PAN='pa',
    PES='fa',
    POL='pl',
    POR='pt',
    PRS='prs',
    PUS='ps',
    RON='ro',
    RUS='ru',
    SLK='sk',
    SOM='so',
    SPA='es',
    SQI='sq',
    SWA='sw',
    TAM='ta',
    TAT='tt',
    THA='th',
    TIR='ti',
    TUR='tr',
    UKR='uk',
    URD='ur',
    UZB='uz',
    VIE='vi',
    YUE='yue',
    ZUL='zu'
)

BCP_CODES = set(ISO6393_TO_BCP47.values())

def iso_to_bcp(language_code: str) -> Optional[str]:
    if bcp_code := ISO6393_TO_BCP47.get(language_code.upper()):
        return bcp_code
    elif language_code.lower() in BCP_CODES:
        return language_code
    elif lang_info := langcodes.get(language_code):
        return lang_info.language
    else:
        return None
