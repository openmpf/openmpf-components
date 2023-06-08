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

# Dict of conversions from ISO639-3 language codes to BCP-47 codes. The first
#  BCP-47 code in each value list will be used (the rest are primarily for
#  reference in case of later changes or customization)
ISO6393_TO_BCP47 = dict(
    amh=["am-ET"],
    ara=["ar-EG", "ar-SA", "ar-IQ", "ar-IL", "ar-AE", "ar-SY", "ar-LY", "ar-DZ",
         "ar-BH", "ar-JO", "ar-KW", "ar-LB", "ar-MA", "ar-OM", "ar-PS", "ar-QA",
         "ar-TN", "ar-YE"],
    aze=["az-AZ"],
    bel=["be-BY"],
    ben=["bn-BD", "bn-IN"],
    # bod=["bo"], # Deprecated
    bul=["bg-BG"],
    # ceb=["ceb"],
    ces=["cs-CZ"],
    cmn=["zh-CN"],
    ell=["el-GR"],
    eng=["en-US", "en-CA", "en-GB", "en-AU", "en-GH", "en-HK", "en-IN", "en-IE",
         "en-KE", "en-NZ", "en-NG", "en-PH", "en-SG", "en-ZA", "en-TZ"],
    fra=["fr-FR", "fr-CA", "fr-CH"],
    # gug=["gn"], # Deprecated
    # hat=[],
    # hau=["ha"], # Deprecated
    # hbs=["sh"], # Deprecated
    hin=["hi-IN"],
    # hye=["hy"],
    ind=["id-ID"],
    jav=["jv-ID"],
    jpn=["ja-JP"],
    kat=["ka-GE"],
    kaz=["kk-KZ"],
    kir=["ky-KG"],
    kor=["ko-KR"],
    # kur=["ku"], # Deprecated
    lao=["lo-LA"],
    lit=["lt-LT"],
    # luo=[],
    mkd=["mk-MK"],
    mya=["my-MM"],
    nan=["zh-TW", "nan-TW"],
    # nde=["nd"],
    # orm=["om"],
    pan=["pa-IN"],
    pes=["fa-IR"],
    pol=["pl-PL"],
    por=["pt-BR", "pt-PT"],
    prs=["prs-AF"],
    pus=["pa-AF"],
    ron=["ro-RO", "ro-MD"],
    # run=[],
    rus=["ru-RU"],
    slk=["sk-SK"],
    # sna=["sn"],
    som=["so-SO"],
    spa=["es-MX", "es-US", "es-AR", "es-BO", "es-CL", "es-CO", "es-CR", "es-CU",
         "es-DO", "es-EC", "es-SV", "es-GQ", "es-GT", "es-HN", "es-NI", "es-PA",
         "es-PY", "es-PE", "es-PR", "es-ES", "es-UY", "es-VE"],
    sqi=["sq-AL"],
    swa=["sw-KE", "sw-TZ"],
    tam=["ta-IN"],
    # tat=[],
    tgk=["tg-TJ"],
    tgl=["fil-PH", "tl-PH"],
    tha=["th-TH"],
    # tir=[],
    tpi=["tpi-PG"],
    tur=["tr-TR"],
    ukr=["uk-UA"],
    urd=["ur-IN"],
    uzb=["uz-UZ"],
    vie=["vi-VN"],
    yue=["zh-HK", "yue-CN"],
    zul=["zu-ZA"]
)