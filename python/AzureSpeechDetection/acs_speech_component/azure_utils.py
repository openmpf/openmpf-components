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
    afr=["af-ZA"],
    amh=["am-ET"],
    ara=["ar-EG", "ar-SA", "ar-IQ", "ar-IL", "ar-AE", "ar-SY", "ar-LY", "ar-DZ",
         "ar-BH", "ar-JO", "ar-KW", "ar-LB", "ar-MA", "ar-OM", "ar-PS", "ar-QA",
         "ar-TN", "ar-YE"],
    aze=["az-AZ"],
    # bel=["be-BY"], # Depreciated
    # ben=["bn-BD", "bn-IN"], # Bengali-Bangladesh has been depreciated
    ben=["bn-IN"],
    # bod=["bo"], # Deprecated
    bul=["bg-BG"],
    bos=["bs-BA"],
    cat=["ca-ES"],
    # ceb=["ceb"],
    ces=["cs-CZ"],
    cym=["cy-GB"],
    cze=["cs-CZ"],
    dan=["da-DK"],
    deu=["de-DE", "de-AT", "de-CH"],
    ell=["el-GR"],
    eng=["en-US", "en-CA", "en-GB", "en-AU", "en-GH", "en-HK", "en-IN", "en-IE",
         "en-KE", "en-NZ", "en-NG", "en-PH", "en-SG", "en-ZA", "en-TZ"],
    est=["et-EE"], # Estonian (Inclusive)
    ekk=["et-EE"], # Standard Estonian
    #vro=["et-EE"], Voro, doesn't seem to be direct match
    eus=["eu-ES"],
    fas=["fa-IR"],
    fin=["fi-FI"],
    fil=["fil-PH"],
    fra=["fr-FR", "fr-BE", "fr-CA", "fr-CH"],
    gle=["ga-IE"],
    glg=["gl-ES"],
    guj=["gu-IN"],
    heb=["he-IL"],
    hin=["hi-IN"],
    hrv=["hr-HR"],
    hun=["hu-HU"],
    #ohu=["hu-HU"], # Note: Old-Hungarian, might not fully work with modern "hu-HU"
    # gug=["gn"], # Deprecated
    # hat=[],
    # hau=["ha"], # Deprecated
    # hbs=["sh"], # Deprecated
    # hye=["hy"],
    ita=["it-IT", "it-CH"],
    ind=["id-ID"],
    ice=["is-IS"],
    isl=["is-IS"],
    jav=["jv-ID"],
    jpn=["ja-JP"],
    kat=["ka-GE"],
    kaz=["kk-KZ"],
    khm=["km-KH"],
    kxm=["km-KH"], # Northern Khmer, might not work as well.
    kan=["kn-IN"],
    # kir=["ky-KG"], # Deprecated
    kor=["ko-KR"],
    # kur=["ku"], # Deprecated
    lao=["lo-LA"],
    lit=["lt-LT"],
    lav=["lv-LV"],
    lvs=["lv-LV"], # Standard Latvian
    # luo=[],
    mkd=["mk-MK"],
    mya=["my-MM"],
    mal=["ml-IN"],
    mon=["mn-MN"], # Mongolian (Inclusive)
    khk=["mn-MN"], # Khalkha Mongolian (Predominant)
    mvf=["mn-MN"], # Peripheral Mongolian (Part)
    mar=["mr-IN"],
    zsm=["ms-MY"],
    mlt=["mt-MT"],
    nob=["nb-NO"],
    nep=["ne-NP"], # Nepali (Macrolanguage)
    npi=["ne-NP"], # Nepali
    nld=["nl-NL", "nl-BE"], # Netherlands and Belgium
    # omr=["mr-IN"], # Old Maranthi, might not work
    # nde=["nd"],
    # orm=["om"],
    pan=["pa-IN"],
    pes=["fa-IR"],
    pol=["pl-PL"],
    por=["pt-BR", "pt-PT"], # pt-BR = Portuguese Brazil, pt-PT = Portuguese Portugal
    pus=["ps-AF"], # Pashto, Pushto (Inclusive)
    pbu=["ps-AF"], # Northern Pahsto
    pst=["ps-AF"], # Central Pahsto
    pbt=["ps-AF"], # Southern Pahsto
    sin=["si-LK"],
    #prs=["prs-AF"], # Deprecated
    #pus=["pa-AF"], # Deprecated
    #ron=["ro-RO", "ro-MD"], # ro-MD depreciated
    ron=["ro-RO"],
    # run=[],
    rus=["ru-RU"],
    slk=["sk-SK"],
    slv=["sl-SI"],
    # sna=["sn"],
    som=["so-SO"],
    spa=["es-MX", "es-US", "es-AR", "es-BO", "es-CL", "es-CO", "es-CR", "es-CU",
         "es-DO", "es-EC", "es-SV", "es-GQ", "es-GT", "es-HN", "es-NI", "es-PA",
         "es-PY", "es-PE", "es-PR", "es-ES", "es-UY", "es-VE"],

    sqi=["sq-AL"],
    swa=["sw-KE", "sw-TZ"],
    swe=["sv-SE"],
    srp=["sr-RS"],
    tam=["ta-IN"],
    tel=["te-IN"],
    # wbq = ["te-IN"], Waddar/Vadari is related to Telugu.
    # tat=[],
    #tgk=["tg-TJ"], # Depreciated
    #tgl=["fil-PH", "tl-PH"], # "tl-PH" Depreciated
    tha=["th-TH"],
    # tir=[],
    #tpi=["tpi-PG"], # Depreciated
    tur=["tr-TR"],
    ukr=["uk-UA"],
    urd=["ur-IN"],
    uzb=["uz-UZ"],
    vie=["vi-VN"],
    cmn=["zh-CN", "zh-CN-shandong", "zh-CN-sichuan", "zh-HK", "zh-TW"],
    zho=["zh-CN", "zh-CN-shandong", "zh-CN-sichuan", "zh-HK", "zh-TW"],
    yue=["yue-CN", "zh-HK"], # Cantonese
    wuu=["wuu-CN"],
    # nan=["zh-TW", "nan-TW"], # nan-TW depreciated
    nan=["zh-TW"], # Note, Taiwanese has one standard + one major dialect,
    # not sure which is covered better by Azure.
    zul=["zu-ZA"]
)