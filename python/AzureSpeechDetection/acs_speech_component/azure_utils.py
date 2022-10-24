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

# Dict of conversions from ISO639-3 language codes to BCP-47 codes. The first
#  BCP-47 code in each value list will be used (the rest are primarily for
#  reference in case of later changes or customization)
ISO6393_TO_BCP47 = dict(
    AMH=["am-ET"],
    ARA=["ar-EG", "ar-SA", "ar-IQ", "ar-IL", "ar-AE", "ar-SY", "ar-LY", "ar-DZ",
         "ar-BH", "ar-JO", "ar-KW", "ar-LB", "ar-MA", "ar-OM", "ar-PS", "ar-QA",
         "ar-TN", "ar-YE"],
    # AZE=["az-AZ"],
    # BEL=["be-BY"],
    # BEN=["bn-BD", "bn-IN"],
    # BOD=["bo"], # Deprecated
    BUL=["bg-BG"],
    # CEB=["ceb"],
    CES=["cs-CZ"],
    CMN=["zh-CN"],
    ELL=["el-GR"],
    ENG=["en-US", "en-CA", "en-GB", "en-AU", "en-GH", "en-HK", "en-IN", "en-IE",
         "en-KE", "en-NZ", "en-NG", "en-PH", "en-SG", "en-ZA", "en-TZ"],
    FRA=["fr-FR", "fr-CA", "fr-CH"],
    # GUG=["gn"], # Deprecated
    # HAT=[],
    # HAU=["ha"], # Deprecated
    # HBS=["sh"], # Deprecated
    HIN=["hi-IN"],
    # HYE=["hy"],
    IND=["id-ID"],
    JAV=["jv-ID"],
    JPN=["ja-JP"],
    # KAT=["ka-GE"],
    # KAZ=["kk-KZ"],
    # KIR=["ky-KG"],
    KOR=["ko-KR"],
    # KUR=["ku"], # Deprecated
    LAO=["lo-LA"],
    LIT=["lt-LT"],
    # LUO=[],
    MKD=["mk-MK"],
    MYA=["my-MM"],
    NAN=["zh-TW", "nan-TW"],
    # NDE=["nd"],
    # ORM=["om"],
    # PAN=["pa-IN"],
    PES=["fa-IR"],
    POL=["pl-PL"],
    POR=["pt-BR", "pt-PT"],
    # PRS=["prs-AF"],
    # PUS=["pa-AF"],
    RON=["ro-RO", "ro-MD"],
    # RUN=[],
    RUS=["ru-RU"],
    SLK=["sk-SK"],
    # SNA=["sn"],
    # SOM=["so-SO"],
    SPA=["es-MX", "es-US", "es-AR", "es-BO", "es-CL", "es-CO", "es-CR", "es-CU",
         "es-DO", "es-EC", "es-SV", "es-GQ", "es-GT", "es-HN", "es-NI", "es-PA",
         "es-PY", "es-PE", "es-PR", "es-ES", "es-UY", "es-VE"],
    # SQI=["sq-AL"],
    SWA=["sw-KE", "sw-TZ"],
    TAM=["ta-IN"],
    # TAT=[],
    # TGK=["tg-TJ"],
    TGL=["fil-PH", "tl-PH"],
    THA=["th-TH"],
    # TIR=[],
    # TPI=["tpi-PG"],
    TUR=["tr-TR"],
    UKR=["uk-UA"],
    # URD=["ur-IN"],
    UZB=["uz-UZ"],
    VIE=["vi-VN"],
    YUE=["zh-HK", "yue-CN"],
    ZUL=["zu-ZA"]
)