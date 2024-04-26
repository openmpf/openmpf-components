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

# This mapping is also used to generate `ISO_LANGUAGE` codes after
# Speech-to-Text conversion. The last ISO code listed in the key list will be used.

# Supported languages can be found here:
# https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=stt
ISO6393_TO_BCP47 = dict(
    afr=["af-ZA"], # Afrikaans
    amh=["am-ET"], # Amharic
    ara=["ar-EG", "ar-SA", "ar-IQ", "ar-IL", "ar-AE", "ar-SY", "ar-LY", "ar-DZ",
         "ar-BH", "ar-JO", "ar-KW", "ar-LB", "ar-MA", "ar-OM", "ar-PS", "ar-QA",
         "ar-TN", "ar-YE"], # Arabic
    azj=["az-AZ"], # North Azerbaijani
    azb=["az-AZ"], # South Azerbaijani
    aze=["az-AZ"], # Azerbaijani (Inclusive)
    # bel=["be-BY"], # Deprecated
    ben=["bn-IN"], # "bn-BD" Bengali-Bangladesh has been deprecated
    # bod=["bo"], # Deprecated
    bul=["bg-BG"], # Bulgarian
    bos=["bs-BA"], # Bosnian
    cat=["ca-ES"], # Catalan, Valencian
    # ceb=["ceb"],
    cze=["cs-CZ"], # Czech SO-639-2 Variant
    ces=["cs-CZ"], # Czech
    wel=["cy-GB"], # Welsh ISO-639-2 Variant
    cym=["cy-GB"], # Welsh
    dan=["da-DK"], # Note: There is a related dialect JUT - Jutlandic
    jut=["da-DK"], # Upon further research, Jutlantic is present in Denmark
                   # but declining over time.
    ger=["de-DE", "de-AT", "de-CH"], # German ISO-2 Variant
    deu=["de-DE", "de-AT", "de-CH"], # German
    # Many other forms of German exist
    gsw=["de-CH"], # Swiss German
    bar=["de-AT"], # Bavarian / Upper German variant common in most of Austria
    gre=["el-GR"], # Greek ISO-639-2 Variant
    ell=["el-GR"], # Greek
    eng=["en-US", "en-CA", "en-GB", "en-AU", "en-GH", "en-HK", "en-IN", "en-IE",
         "en-KE", "en-NZ", "en-NG", "en-PH", "en-SG", "en-ZA", "en-TZ"],
    ekk=["et-EE"], # Standard Estonian
    est=["et-EE"], # Estonian (Inclusive)
    # vro=["et-EE"], Voro, doesn't seem to be direct match
    baq=["eu-ES"], # Basque ISO-639-2 Variant
    eus=["eu-ES"], # Basque
    fin=["fi-FI"], # Finnish
    fre=["fr-FR", "fr-BE", "fr-CA", "fr-CH"], # French ISO-639-2 Variant
    fra=["fr-FR", "fr-BE", "fr-CA", "fr-CH"], # French
    gle=["ga-IE"], # Irish
    glg=["gl-ES"], # Galician
    guj=["gu-IN"], # Gujarati
    heb=["he-IL"], # Hebrew
    hin=["hi-IN"], # Hindi
    hrv=["hr-HR"], # Croatian
    hun=["hu-HU"], # Hungarian
    # ohu=["hu-HU"], # Note: Old-Hungarian, might not fully work with modern "hu-HU"
    # gug=["gn"], # Deprecated
    # hat=[],
    # hau=["ha"], # Deprecated
    # hbs=["sh"], # Deprecated
    arm=['hy-AM'], # Armenian ISO-639-2 Variant
    hye=['hy-AM'], # Armenian
    ita=["it-IT", "it-CH"], # Italian
    ind=["id-ID"], # Indonesian
    ice=["is-IS"], # Icelandic ISO-639-2 Variant
    isl=["is-IS"], # Icelandic
    jav=["jv-ID"], # Javanese
    jpn=["ja-JP"], # Japanese
    geo=["ka-GE"], # Georgian ISO-639-2 Variant
    kat=["ka-GE"], # Georgian
    kaz=["kk-KZ"], # Kazakh
    kxm=["km-KH"], # Northern Khmer, might not work as well.
    khm=["km-KH"], # Central Khmer
    kan=["kn-IN"], # Kannada
    # kir=["ky-KG"], # Deprecated
    kor=["ko-KR"], # Korean
    # kur=["ku"], # Deprecated
    lao=["lo-LA"], # Lao
    lit=["lt-LT"], # Lithuanian
    lvs=["lv-LV"], # Standard Latvian
    lav=["lv-LV"], # Latvian (Inclusive)
    # luo=[],
    mac=["mk-MK"], # Macedonian ISO 639-2 Variant
    mkd=["mk-MK"], # Macedonian
    bur=["my-MM"], # Burmese / Myanmar ISO 639-2 Variant
    mya=["my-MM"], # Burmese / Myanmar
    mal=["ml-IN"], # Malayalam
    khk=["mn-MN"], # Khalkha Mongolian (Predominant)
    mvf=["mn-MN"], # Peripheral Mongolian (Part)
    mon=["mn-MN"], # Mongolian (Inclusive)
    mar=["mr-IN"], # Marathi
    may=["ms-MY"], # Malay ISO 639-2 Variant
    msa=["ms-MY"], # Malay (Inclusive, Macrolanguage)
    zsm=["ms-MY"], # Standard Malay (Malaysian Malay)
    # In this case, the ms-MY code indicates ZSM (Standard Malay)
    mlt=["mt-MT"], # Maltese
    nob=["nb-NO"], # Norwegian Bokmål (Norway)
    npi=["ne-NP"], # Nepali
    nep=["ne-NP"], # Nepali (Inclusive, Macrolanguage)
    dut=["nl-NL", "nl-BE"], # Dutch ISO 639-2 Variant
    nld=["nl-NL", "nl-BE"], # Dutch - Netherlands and Belgium
    # omr=["mr-IN"], # Old Maranthi, might not work
    # nde=["nd"],
    # orm=["om"],
    pan=["pa-IN"], # Punjabi, Panjabi
    per=["fa-IR"], # Persian ISO-639-2 Variant
    fas=["fa-IR"], # Persian
    pes=["fa-IR"], # Iranian Persian
    pol=["pl-PL"], # Polish
    por=["pt-BR", "pt-PT"], # pt-BR = Portuguese Brazil, pt-PT = Portuguese Portugal
    pbu=["ps-AF"], # Northern Pahsto
    pst=["ps-AF"], # Central Pahsto
    pbt=["ps-AF"], # Southern Pahsto
    pus=["ps-AF"], # Pashto, Pushto (Inclusive)
    sin=["si-LK"], # Sinhala, Sinhalese
    # prs=["prs-AF"], # Deprecated
    # pus=["pa-AF"], # Deprecated
    rum=["ro-RO"], # Romanian ISO-639-2 Variant
    ron=["ro-RO"], # Romanian, Moldavian, Moldovan
    # ro-MD deprecated
    # run=[],
    rus=["ru-RU"], # Russian
    slo=["sk-SK"], # Slovak ISO-639-2 Variant
    slk=["sk-SK"], # Slovak
    slv=["sl-SI"], # Slovenian
    # sna=["sn"],
    som=["so-SO"], # Somali
    spa=["es-MX", "es-US", "es-AR", "es-BO", "es-CL", "es-CO", "es-CR", "es-CU",
         "es-DO", "es-EC", "es-SV", "es-GQ", "es-GT", "es-HN", "es-NI", "es-PA",
         "es-PY", "es-PE", "es-PR", "es-ES", "es-UY", "es-VE"], # Spanish
    alb=["sq-AL"], # Albanian ISO-639-2 Variant
    sqi=["sq-AL"], # Albanian
    swa=["sw-KE", "sw-TZ"], # Swahili
    swe=["sv-SE"], # Swedish
    srp=["sr-RS"], # Serbian
    tam=["ta-IN"], # Tamil
    tel=["te-IN"], # Telugu
    # wbq = ["te-IN"], Waddar/Vadari is related to Telugu.
    # tat=[],
    # tgk=["tg-TJ"], # Deprecated
    tgl=["fil-PH"], # "tl-PH" deprecated, Tagalog
    fil=["fil-PH"], # Filipino (Standardized form of Tagalog)
    tha=["th-TH"], # Thai
    # tir=[],
    # tpi=["tpi-PG"], # Deprecated
    tur=["tr-TR"], # Turkish
    ukr=["uk-UA"], # Ukrainian
    urd=["ur-IN"], # Urdu
    uzb=["uz-UZ"], # Uzbek
    vie=["vi-VN"], # Vietnamese
    cmn=["zh-CN", "zh-CN-shandong", "zh-CN-sichuan", "zh-HK", "zh-TW"], # Mandarin Chinese
    zho=["zh-CN", "zh-CN-shandong", "zh-CN-sichuan", "zh-HK", "zh-TW"], # Chinese
    yue=["yue-CN", "zh-HK"], # Cantonese
    wuu=["wuu-CN"], # Chinese (Wu, Simplified)
    nan=["zh-TW"], # nan-TW deprecated
    # Note, Taiwanese has one standard + one major dialect,
    # not sure which is covered better by Azure.
    zul=["zu-ZA"] # Zulu
)