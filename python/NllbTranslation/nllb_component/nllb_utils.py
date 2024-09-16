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

class NllbLanguageMapper:

    # double nested dictionary to convert ISO-639-3 language and ISO-15924 script into Flores-200
    _iso_to_flores200: dict[str, dict[str, str]] = {
    'ace' : {'Arab': 'ace_Arab',    # Acehnese Arabic
             'Latn': 'ace_Latn'},   # Acehnese Latin
    'acm' : {'Arab': 'acm_Arab'},   # Mesopotamian Arabic
    'acq' : {'Arab': 'acq_Arab'},   # Ta’izzi-Adeni Arabic
    'aeb' : {'Arab': 'aeb_Arab'},   # Tunisian Arabic
    'afr' : {'Latn': 'afr_Latn'},   # Afrikaans
    'ajp' : {'Arab': 'ajp_Arab'},   # South Levantine Arabic
    'aka' : {'Latn': 'aka_Latn'},   # Akan
    'amh' : {'Ethi': 'amh_Ethi'},   # Amharic
    'apc' : {'Arab': 'apc_Arab'},   # North Levantine Arabic
    'arb' : {'Arab': 'arb_Arab',    # Modern Standard Arabic
             'Latn': 'arb_Latn'},   # Modern Standard Arabic (Romanized)
    'ars' : {'Arab': 'ars_Arab'},   # Najdi Arabic
    'ary' : {'Arab': 'ary_Arab'},   # Moroccan Arabic
    'arz' : {'Arab': 'arz_Arab'},   # Egyptian Arabic
    'asm' : {'Beng': 'asm_Beng'},   # Assamese
    'ast' : {'Latn': 'ast_Latn'},   # Asturian
    'awa' : {'Deva': 'awa_Deva'},	# Awadhi
    'ayr' : {'Latn': 'ayr_Latn'},	# Central Aymara
    'azb' : {'Arab': 'azb_Arab'},	# South Azerbaijani
    'azj' : {'Latn': 'azj_Latn'},	# North Azerbaijani
    'bak' : {'Cyrl': 'bak_Cyrl'},	# Bashkir
    'bam' : {'Latn': 'bam_Latn'},	# Bambara
    'ban' : {'Latn': 'ban_Latn'},	# Balinese
    'bel' : {'Cyrl': 'bel_Cyrl'},   # Belarusian
    'bem' : {'Latn': 'bem_Latn'},	# Bemba
    'ben' : {'Beng': 'ben_Beng'},	# Bengali
    'bho' : {'Deva': 'bho_Deva'},	# Bhojpuri
    'bjn' : {'Arab': 'bjn_Arab',	# Banjar (Arabic script)
             'Latn': 'bjn_Latn'},	# Banjar (Latin script)
    'bod' : {'Tibt': 'bod_Tibt'},	# Standard Tibetan
    'bos' : {'Latn': 'bos_Latn'},	# Bosnian
    'bug' : {'Latn': 'bug_Latn'},	# Buginese
    'bul' : {'Cyrl': 'bul_Cyrl'},	# Bulgarian
    'cat' : {'Latn': 'cat_Latn'},	# Catalan
    'ceb' : {'Latn': 'ceb_Latn'},	# Cebuano
    'ces' : {'Latn': 'ces_Latn'},	# Czech
    'cjk' : {'Latn': 'cjk_Latn'},	# Chokwe
    'ckb' : {'Arab': 'ckb_Arab'},	# Central Kurdish
    'crh' : {'Latn': 'crh_Latn'},	# Crimean Tatar
    'cym' : {'Latn': 'cym_Latn'},	# Welsh
    'dan' : {'Latn': 'dan_Latn'},	# Danish
    'deu' : {'Latn': 'deu_Latn'},	# German
    'dik' : {'Latn': 'dik_Latn'},	# Southwestern Dinka
    'dyu' : {'Latn': 'dyu_Latn'},	# Dyula
    'dzo' : {'Tibt': 'dzo_Tibt'},	# Dzongkha
    'ell' : {'Grek': 'ell_Grek'},	# Greek
    'eng' : {'Latn': 'eng_Latn'},	# English
    'epo' : {'Latn': 'epo_Latn'},	# Esperanto
    'est' : {'Latn': 'est_Latn'},	# Estonian
    'eus' : {'Latn': 'eus_Latn'},	# Basque
    'ewe' : {'Latn': 'ewe_Latn'},	# Ewe
    'fao' : {'Latn': 'fao_Latn'},	# Faroese
    'fij' : {'Latn': 'fij_Latn'},	# Fijian
    'fin' : {'Latn': 'fin_Latn'},	# Finnish
    'fon' : {'Latn': 'fon_Latn'},	# Fon
    'fra' : {'Latn': 'fra_Latn'},	# French
    'fur' : {'Latn': 'fur_Latn'},	# Friulian
    'fuv' : {'Latn': 'fuv_Latn'},	# Nigerian Fulfulde
    'gla' : {'Latn': 'gla_Latn'},	# Scottish Gaelic
    'gle' : {'Latn': 'gle_Latn'},	# Irish
    'glg' : {'Latn': 'glg_Latn'},	# Galician
    'grn' : {'Latn': 'grn_Latn'},	# Guarani
    'guj' : {'Gujr': 'guj_Gujr'},	# Gujarati
    'hat' : {'Latn': 'hat_Latn'},	# Haitian Creole
    'hau' : {'Latn': 'hau_Latn'},	# Hausa
    'heb' : {'Hebr': 'heb_Hebr'},	# Hebrew
    'hin' : {'Deva': 'hin_Deva'},	# Hindi
    'hne' : {'Deva': 'hne_Deva'},	# Chhattisgarhi
    'hrv' : {'Latn': 'hrv_Latn'},	# Croatian
    'hun' : {'Latn': 'hun_Latn'},	# Hungarian
    'hye' : {'Armn': 'hye_Armn'},	# Armenian
    'ibo' : {'Latn': 'ibo_Latn'},	# Igbo
    'ilo' : {'Latn': 'ilo_Latn'},	# Ilocano
    'ind' : {'Latn': 'ind_Latn'},	# Indonesian
    'isl' : {'Latn': 'isl_Latn'},	# Icelandic
    'ita' : {'Latn': 'ita_Latn'},	# Italian
    'jav' : {'Latn': 'jav_Latn'},	# Javanese
    'jpn' : {'Jpan': 'jpn_Jpan'},	# Japanese
    'kab' : {'Latn': 'kab_Latn'},	# Kabyle
    'kac' : {'Latn': 'kac_Latn'},	# Jingpho
    'kam' : {'Latn': 'kam_Latn'},	# Kamba
    'kan' : {'Knda': 'kan_Knda'},	# Kannada
    'kas' : {'Arab': 'kas_Arab',	# Kashmiri (Arabic script)
             'Deva': 'kas_Deva'},	# Kashmiri (Devanagari script)
    'kat' : {'Geor': 'kat_Geor'},	# Georgian
    'knc' : {'Arab': 'knc_Arab',	# Central Kanuri (Arabic script)
             'Latn': 'knc_Latn'},	# Central Kanuri (Latin script)
    'kaz' : {'Cyrl': 'kaz_Cyrl'},	# Kazakh
    'kbp' : {'Latn': 'kbp_Latn'},	# Kabiyè
    'kea' : {'Latn': 'kea_Latn'},	# Kabuverdianu
    'khm' : {'Khmr': 'khm_Khmr'},	# Khmer
    'kik' : {'Latn': 'kik_Latn'},	# Kikuyu
    'kin' : {'Latn': 'kin_Latn'},	# Kinyarwanda
    'kir' : {'Cyrl': 'kir_Cyrl'},	# Kyrgyz
    'kmb' : {'Latn': 'kmb_Latn'},	# Kimbundu
    'kmr' : {'Latn': 'kmr_Latn'},	# Northern Kurdish
    'kon' : {'Latn': 'kon_Latn'},	# Kikongo
    'kor' : {'Hang': 'kor_Hang'},	# Korean
    'lao' : {'Laoo': 'lao_Laoo'},	# Lao
    'lij' : {'Latn': 'lij_Latn'},	# Ligurian
    'lim' : {'Latn': 'lim_Latn'},	# Limburgish
    'lin' : {'Latn': 'lin_Latn'},	# Lingala
    'lit' : {'Latn': 'lit_Latn'},	# Lithuanian
    'lmo' : {'Latn': 'lmo_Latn'},	# Lombard
    'ltg' : {'Latn': 'ltg_Latn'},	# Latgalian
    'ltz' : {'Latn': 'ltz_Latn'},	# Luxembourgish
    'lua' : {'Latn': 'lua_Latn'},	# Luba-Kasai
    'lug' : {'Latn': 'lug_Latn'},	# Ganda
    'luo' : {'Latn': 'luo_Latn'},	# Luo
    'lus' : {'Latn': 'lus_Latn'},	# Mizo
    'lvs' : {'Latn': 'lvs_Latn'},	# Standard Latvian
    'mag' : {'Deva': 'mag_Deva'},	# Magahi
    'mai' : {'Deva': 'mai_Deva'},	# Maithili
    'mal' : {'Mlym': 'mal_Mlym'},	# Malayalam
    'mar' : {'Deva': 'mar_Deva'},	# Marathi
    'min' : {'Arab': 'min_Arab',	# Minangkabau (Arabic script)
             'Latn': 'min_Latn'},	# Minangkabau (Latin script)
    'mkd' : {'Cyrl': 'mkd_Cyrl'},	# Macedonian
    'plt' : {'Latn': 'plt_Latn'},	# Plateau Malagasy
    'mlt' : {'Latn': 'mlt_Latn'},	# Maltese
    'mni' : {'Beng': 'mni_Beng'},	# Meitei (Bengali script)
    'khk' : {'Cyrl': 'khk_Cyrl'},	# Halh Mongolian
    'mos' : {'Latn': 'mos_Latn'},	# Mossi
    'mri' : {'Latn': 'mri_Latn'},	# Maori
    'mya' : {'Mymr': 'mya_Mymr'},	# Burmese
    'nld' : {'Latn': 'nld_Latn'},	# Dutch
    'nno' : {'Latn': 'nno_Latn'},	# Norwegian Nynorsk
    'nob' : {'Latn': 'nob_Latn'},	# Norwegian Bokmål
    'npi' : {'Deva': 'npi_Deva'},	# Nepali
    'nso' : {'Latn': 'nso_Latn'},	# Northern Sotho
    'nus' : {'Latn': 'nus_Latn'},	# Nuer
    'nya' : {'Latn': 'nya_Latn'},	# Nyanja
    'oci' : {'Latn': 'oci_Latn'},	# Occitan
    'gaz' : {'Latn': 'gaz_Latn'},	# West Central Oromo
    'ory' : {'Orya': 'ory_Orya'},	# Odia
    'pag' : {'Latn': 'pag_Latn'},	# Pangasinan
    'pan' : {'Guru': 'pan_Guru'},	# Eastern Panjabi
    'pap' : {'Latn': 'pap_Latn'},	# Papiamento
    'pes' : {'Arab': 'pes_Arab'},	# Western Persian
    'pol' : {'Latn': 'pol_Latn'},	# Polish
    'por' : {'Latn': 'por_Latn'},	# Portuguese
    'prs' : {'Arab': 'prs_Arab'},	# Dari
    'pbt' : {'Arab': 'pbt_Arab'},	# Southern Pashto
    'quy' : {'Latn': 'quy_Latn'},	# Ayacucho Quechua
    'ron' : {'Latn': 'ron_Latn'},	# Romanian
    'run' : {'Latn': 'run_Latn'},	# Rundi
    'rus' : {'Cyrl': 'rus_Cyrl'},	# Russian
    'sag' : {'Latn': 'sag_Latn'},	# Sango
    'san' : {'Deva': 'san_Deva'},	# Sanskrit
    'sat' : {'Olck': 'sat_Olck'},	# Santali
    'scn' : {'Latn': 'scn_Latn'},	# Sicilian
    'shn' : {'Mymr': 'shn_Mymr'},	# Shan
    'sin' : {'Sinh': 'sin_Sinh'},	# Sinhala
    'slk' : {'Latn': 'slk_Latn'},	# Slovak
    'slv' : {'Latn': 'slv_Latn'},	# Slovenian
    'smo' : {'Latn': 'smo_Latn'},	# Samoan
    'sna' : {'Latn': 'sna_Latn'},	# Shona
    'snd' : {'Arab': 'snd_Arab'},	# Sindhi
    'som' : {'Latn': 'som_Latn'},	# Somali
    'sot' : {'Latn': 'sot_Latn'},	# Southern Sotho
    'spa' : {'Latn': 'spa_Latn'},	# Spanish
    'als' : {'Latn': 'als_Latn'},	# Tosk Albanian
    'srd' : {'Latn': 'srd_Latn'},	# Sardinian
    'srp' : {'Cyrl': 'srp_Cyrl'},	# Serbian
    'ssw' : {'Latn': 'ssw_Latn'},	# Swati
    'sun' : {'Latn': 'sun_Latn'},	# Sundanese
    'swe' : {'Latn': 'swe_Latn'},	# Swedish
    'swh' : {'Latn': 'swh_Latn'},	# Swahili
    'szl' : {'Latn': 'szl_Latn'},	# Silesian
    'tam' : {'Taml': 'tam_Taml'},	# Tamil
    'tat' : {'Cyrl': 'tat_Cyrl'},	# Tatar
    'tel' : {'Telu': 'tel_Telu'},	# Telugu
    'tgk' : {'Cyrl': 'tgk_Cyrl'},	# Tajik
    'tgl' : {'Latn': 'tgl_Latn'},	# Tagalog
    'tha' : {'Thai': 'tha_Thai'},	# Thai
    'tir' : {'Ethi': 'tir_Ethi'},	# Tigrinya
    'taq' : {'Latn': 'taq_Latn',	# Tamasheq (Latin script)
             'Tfng': 'taq_Tfng'},	# Tamasheq (Tifinagh script)
    'tpi' : {'Latn': 'tpi_Latn'},	# Tok Pisin
    'tsn' : {'Latn': 'tsn_Latn'},	# Tswana
    'tso' : {'Latn': 'tso_Latn'},	# Tsonga
    'tuk' : {'Latn': 'tuk_Latn'},	# Turkmen
    'tum' : {'Latn': 'tum_Latn'},	# Tumbuka
    'tur' : {'Latn': 'tur_Latn'},	# Turkish
    'twi' : {'Latn': 'twi_Latn'},	# Twi
    'tzm' : {'Tfng': 'tzm_Tfng'},	# Central Atlas Tamazight
    'uig' : {'Arab': 'uig_Arab'},	# Uyghur
    'ukr' : {'Cyrl': 'ukr_Cyrl'},	# Ukrainian
    'umb' : {'Latn': 'umb_Latn'},	# Umbundu
    'urd' : {'Arab': 'urd_Arab'},	# Urdu
    'uzn' : {'Latn': 'uzn_Latn'},	# Northern Uzbek
    'vec' : {'Latn': 'vec_Latn'},	# Venetian
    'vie' : {'Latn': 'vie_Latn'},	# Vietnamese
    'war' : {'Latn': 'war_Latn'},	# Waray
    'wol' : {'Latn': 'wol_Latn'},	# Wolof
    'xho' : {'Latn': 'xho_Latn'},	# Xhosa
    'ydd' : {'Hebr': 'ydd_Hebr'},	# Eastern Yiddish
    'yor' : {'Latn': 'yor_Latn'},	# Yoruba
    'yue' : {'Hant': 'yue_Hant'},	# Yue Chinese
    'zho' : {'Hans': 'zho_Hans',	# Chinese (Simplified)
             'Hant': 'zho_Hant'},	# Chinese (Traditional)
    'zsm' : {'Latn': 'zsm_Latn'},	# Standard Malay
    'zul' : {'Latn': 'zul_Latn'}}	# Zulu

    # default a script to use if only language is provided
    _iso_default_script_flores200: dict[str, str] = {
    'ace' : 'ace_Latn', # Acehnese Latin
    'acm' : 'acm_Arab', # Mesopotamian Arabic
    'acq' : 'acq_Arab', # Ta’izzi-Adeni Arabic
    'aeb' : 'aeb_Arab', # Tunisian Arabic
    'afr' : 'afr_Latn', # Afrikaans
    'ajp' : 'ajp_Arab', # South Levantine Arabic
    'aka' : 'aka_Latn', # Akan
    'amh' : 'amh_Ethi', # Amharic
    'apc' : 'apc_Arab', # North Levantine Arabic
    'arb' : 'arb_Arab', # Modern Standard Arabic
    'ars' : 'ars_Arab', # Najdi Arabic
    'ary' : 'ary_Arab', # Moroccan Arabic
    'arz' : 'arz_Arab', # Egyptian Arabic
    'asm' : 'asm_Beng', # Assamese
    'ast' : 'ast_Latn', # Asturian
    'awa' : 'awa_Deva',	# Awadhi
    'ayr' : 'ayr_Latn',	# Central Aymara
    'azb' : 'azb_Arab',	# South Azerbaijani
    'azj' : 'azj_Latn',	# North Azerbaijani
    'bak' : 'bak_Cyrl',	# Bashkir
    'bam' : 'bam_Latn',	# Bambara
    'ban' : 'ban_Latn',	# Balinese
    'bel' : 'bel_Cyrl', # Belarusian
    'bem' : 'bem_Latn',	# Bemba
    'ben' : 'ben_Beng',	# Bengali
    'bho' : 'bho_Deva',	# Bhojpuri
    'bjn' : 'bjn_Latn',	# Banjar (Latin script)
    'bod' : 'bod_Tibt',	# Standard Tibetan
    'bos' : 'bos_Latn',	# Bosnian
    'bug' : 'bug_Latn',	# Buginese
    'bul' : 'bul_Cyrl',	# Bulgarian
    'cat' : 'cat_Latn',	# Catalan
    'ceb' : 'ceb_Latn',	# Cebuano
    'ces' : 'ces_Latn',	# Czech
    'cjk' : 'cjk_Latn',	# Chokwe
    'ckb' : 'ckb_Arab',	# Central Kurdish
    'crh' : 'crh_Latn',	# Crimean Tatar
    'cym' : 'cym_Latn',	# Welsh
    'dan' : 'dan_Latn',	# Danish
    'deu' : 'deu_Latn',	# German
    'dik' : 'dik_Latn',	# Southwestern Dinka
    'dyu' : 'dyu_Latn',	# Dyula
    'dzo' : 'dzo_Tibt',	# Dzongkha
    'ell' : 'ell_Grek',	# Greek
    'eng' : 'eng_Latn',	# English
    'epo' : 'epo_Latn',	# Esperanto
    'est' : 'est_Latn',	# Estonian
    'eus' : 'eus_Latn',	# Basque
    'ewe' : 'ewe_Latn',	# Ewe
    'fao' : 'fao_Latn',	# Faroese
    'fij' : 'fij_Latn',	# Fijian
    'fin' : 'fin_Latn',	# Finnish
    'fon' : 'fon_Latn',	# Fon
    'fra' : 'fra_Latn',	# French
    'fur' : 'fur_Latn',	# Friulian
    'fuv' : 'fuv_Latn',	# Nigerian Fulfulde
    'gla' : 'gla_Latn',	# Scottish Gaelic
    'gle' : 'gle_Latn',	# Irish
    'glg' : 'glg_Latn',	# Galician
    'grn' : 'grn_Latn',	# Guarani
    'guj' : 'guj_Gujr',	# Gujarati
    'hat' : 'hat_Latn',	# Haitian Creole
    'hau' : 'hau_Latn',	# Hausa
    'heb' : 'heb_Hebr',	# Hebrew
    'hin' : 'hin_Deva',	# Hindi
    'hne' : 'hne_Deva',	# Chhattisgarhi
    'hrv' : 'hrv_Latn',	# Croatian
    'hun' : 'hun_Latn',	# Hungarian
    'hye' : 'hye_Armn',	# Armenian
    'ibo' : 'ibo_Latn',	# Igbo
    'ilo' : 'ilo_Latn',	# Ilocano
    'ind' : 'ind_Latn',	# Indonesian
    'isl' : 'isl_Latn',	# Icelandic
    'ita' : 'ita_Latn',	# Italian
    'jav' : 'jav_Latn',	# Javanese
    'jpn' : 'jpn_Jpan',	# Japanese
    'kab' : 'kab_Latn',	# Kabyle
    'kac' : 'kac_Latn',	# Jingpho
    'kam' : 'kam_Latn',	# Kamba
    'kan' : 'kan_Knda',	# Kannada
    'kas' : 'kas_Deva',	# Kashmiri (Devanagari script)
    'kat' : 'kat_Geor',	# Georgian
    'knc' : 'knc_Latn',	# Central Kanuri (Latin script)
    'kaz' : 'kaz_Cyrl',	# Kazakh
    'kbp' : 'kbp_Latn',	# Kabiyè
    'kea' : 'kea_Latn',	# Kabuverdianu
    'khm' : 'khm_Khmr',	# Khmer
    'kik' : 'kik_Latn',	# Kikuyu
    'kin' : 'kin_Latn',	# Kinyarwanda
    'kir' : 'kir_Cyrl',	# Kyrgyz
    'kmb' : 'kmb_Latn',	# Kimbundu
    'kmr' : 'kmr_Latn',	# Northern Kurdish
    'kon' : 'kon_Latn',	# Kikongo
    'kor' : 'kor_Hang',	# Korean
    'lao' : 'lao_Laoo',	# Lao
    'lij' : 'lij_Latn',	# Ligurian
    'lim' : 'lim_Latn',	# Limburgish
    'lin' : 'lin_Latn',	# Lingala
    'lit' : 'lit_Latn',	# Lithuanian
    'lmo' : 'lmo_Latn',	# Lombard
    'ltg' : 'ltg_Latn',	# Latgalian
    'ltz' : 'ltz_Latn',	# Luxembourgish
    'lua' : 'lua_Latn',	# Luba-Kasai
    'lug' : 'lug_Latn',	# Ganda
    'luo' : 'luo_Latn',	# Luo
    'lus' : 'lus_Latn',	# Mizo
    'lvs' : 'lvs_Latn',	# Standard Latvian
    'mag' : 'mag_Deva',	# Magahi
    'mai' : 'mai_Deva',	# Maithili
    'mal' : 'mal_Mlym',	# Malayalam
    'mar' : 'mar_Deva',	# Marathi
    'min' : 'min_Latn',	# Minangkabau (Latin script)
    'mkd' : 'mkd_Cyrl',	# Macedonian
    'plt' : 'plt_Latn',	# Plateau Malagasy
    'mlt' : 'mlt_Latn',	# Maltese
    'mni' : 'mni_Beng',	# Meitei (Bengali script)
    'khk' : 'khk_Cyrl',	# Halh Mongolian
    'mos' : 'mos_Latn',	# Mossi
    'mri' : 'mri_Latn',	# Maori
    'mya' : 'mya_Mymr',	# Burmese
    'nld' : 'nld_Latn',	# Dutch
    'nno' : 'nno_Latn',	# Norwegian Nynorsk
    'nob' : 'nob_Latn',	# Norwegian Bokmål
    'npi' : 'npi_Deva',	# Nepali
    'nso' : 'nso_Latn',	# Northern Sotho
    'nus' : 'nus_Latn',	# Nuer
    'nya' : 'nya_Latn',	# Nyanja
    'oci' : 'oci_Latn',	# Occitan
    'gaz' : 'gaz_Latn',	# West Central Oromo
    'ory' : 'ory_Orya',	# Odia
    'pag' : 'pag_Latn',	# Pangasinan
    'pan' : 'pan_Guru',	# Eastern Panjabi
    'pap' : 'pap_Latn',	# Papiamento
    'pes' : 'pes_Arab',	# Western Persian
    'pol' : 'pol_Latn',	# Polish
    'por' : 'por_Latn',	# Portuguese
    'prs' : 'prs_Arab',	# Dari
    'pbt' : 'pbt_Arab',	# Southern Pashto
    'quy' : 'quy_Latn',	# Ayacucho Quechua
    'ron' : 'ron_Latn',	# Romanian
    'run' : 'run_Latn',	# Rundi
    'rus' : 'rus_Cyrl',	# Russian
    'sag' : 'sag_Latn',	# Sango
    'san' : 'san_Deva',	# Sanskrit
    'sat' : 'sat_Olck',	# Santali
    'scn' : 'scn_Latn',	# Sicilian
    'shn' : 'shn_Mymr',	# Shan
    'sin' : 'sin_Sinh',	# Sinhala
    'slk' : 'slk_Latn',	# Slovak
    'slv' : 'slv_Latn',	# Slovenian
    'smo' : 'smo_Latn',	# Samoan
    'sna' : 'sna_Latn',	# Shona
    'snd' : 'snd_Arab',	# Sindhi
    'som' : 'som_Latn',	# Somali
    'sot' : 'sot_Latn',	# Southern Sotho
    'spa' : 'spa_Latn',	# Spanish
    'als' : 'als_Latn',	# Tosk Albanian
    'srd' : 'srd_Latn',	# Sardinian
    'srp' : 'srp_Cyrl',	# Serbian
    'ssw' : 'ssw_Latn',	# Swati
    'sun' : 'sun_Latn',	# Sundanese
    'swe' : 'swe_Latn',	# Swedish
    'swh' : 'swh_Latn',	# Swahili
    'szl' : 'szl_Latn',	# Silesian
    'tam' : 'tam_Taml',	# Tamil
    'tat' : 'tat_Cyrl',	# Tatar
    'tel' : 'tel_Telu',	# Telugu
    'tgk' : 'tgk_Cyrl',	# Tajik
    'tgl' : 'tgl_Latn',	# Tagalog
    'tha' : 'tha_Thai',	# Thai
    'tir' : 'tir_Ethi',	# Tigrinya
    'taq' : 'taq_Latn',	# Tamasheq (Latin script)
    'tpi' : 'tpi_Latn',	# Tok Pisin
    'tsn' : 'tsn_Latn',	# Tswana
    'tso' : 'tso_Latn',	# Tsonga
    'tuk' : 'tuk_Latn',	# Turkmen
    'tum' : 'tum_Latn',	# Tumbuka
    'tur' : 'tur_Latn',	# Turkish
    'twi' : 'twi_Latn',	# Twi
    'tzm' : 'tzm_Tfng',	# Central Atlas Tamazight
    'uig' : 'uig_Arab',	# Uyghur
    'ukr' : 'ukr_Cyrl',	# Ukrainian
    'umb' : 'umb_Latn',	# Umbundu
    'urd' : 'urd_Arab',	# Urdu
    'uzn' : 'uzn_Latn',	# Northern Uzbek
    'vec' : 'vec_Latn',	# Venetian
    'vie' : 'vie_Latn',	# Vietnamese
    'war' : 'war_Latn',	# Waray
    'wol' : 'wol_Latn',	# Wolof
    'xho' : 'xho_Latn',	# Xhosa
    'ydd' : 'ydd_Hebr',	# Eastern Yiddish
    'yor' : 'yor_Latn',	# Yoruba
    'yue' : 'yue_Hant',	# Yue Chinese
    'zho' : 'zho_Hans',	# Chinese (Simplified)
    'zsm' : 'zsm_Latn',	# Standard Malay
    'zul' : 'zul_Latn'	# Zulu
    }

    @classmethod
    def get_code(cls, lang : str, script : str):
        if script and lang in cls._iso_to_flores200:
            return cls._iso_to_flores200[lang][script]
        return cls._iso_default_script_flores200.get(lang)