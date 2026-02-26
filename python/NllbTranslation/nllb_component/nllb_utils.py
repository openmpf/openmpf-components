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

from __future__ import annotations
import mpf_component_api as mpf

class NllbLanguageMapper:

    # double nested dictionary to convert ISO-639-3 language and ISO-15924 script into Flores-200
    _iso_to_flores200: dict[str, dict[str, str]] = {
    'ace' : {'arab': 'ace_Arab',    # Acehnese Arabic
             'latn': 'ace_Latn'},   # Acehnese Latin
    'acm' : {'arab': 'acm_Arab'},   # Mesopotamian Arabic
    'acq' : {'arab': 'acq_Arab'},   # Ta’izzi-Adeni Arabic
    'aeb' : {'arab': 'aeb_Arab'},   # Tunisian Arabic
    'afr' : {'latn': 'afr_Latn'},   # Afrikaans
    'ajp' : {'arab': 'ajp_Arab'},   # South Levantine Arabic
    'aka' : {'latn': 'aka_Latn'},   # Akan
    'amh' : {'ethi': 'amh_Ethi'},   # Amharic
    'apc' : {'arab': 'apc_Arab'},   # North Levantine Arabic
    'arb' : {'arab': 'arb_Arab',    # Modern Standard Arabic
             'latn': 'arb_Latn'},   # Modern Standard Arabic (Romanized)
    'ars' : {'arab': 'ars_Arab'},   # Najdi Arabic
    'ary' : {'arab': 'ary_Arab'},   # Moroccan Arabic
    'arz' : {'arab': 'arz_Arab'},   # Egyptian Arabic
    'asm' : {'beng': 'asm_Beng'},   # Assamese
    'ast' : {'latn': 'ast_Latn'},   # Asturian
    'awa' : {'deva': 'awa_Deva'},	# Awadhi
    'ayr' : {'latn': 'ayr_Latn'},	# Central Aymara
    'azb' : {'arab': 'azb_Arab'},	# South Azerbaijani
    'azj' : {'latn': 'azj_Latn'},	# North Azerbaijani
    'bak' : {'cyrl': 'bak_Cyrl'},	# Bashkir
    'bam' : {'latn': 'bam_Latn'},	# Bambara
    'ban' : {'latn': 'ban_Latn'},	# Balinese
    'bel' : {'cyrl': 'bel_Cyrl'},   # Belarusian
    'bem' : {'latn': 'bem_Latn'},	# Bemba
    'ben' : {'beng': 'ben_Beng'},	# Bengali
    'bho' : {'deva': 'bho_Deva'},	# Bhojpuri
    'bjn' : {'arab': 'bjn_Arab',	# Banjar (Arabic script)
             'latn': 'bjn_Latn'},	# Banjar (Latin script)
    'bod' : {'tibt': 'bod_Tibt'},	# Standard Tibetan
    'bos' : {'latn': 'bos_Latn'},	# Bosnian
    'bug' : {'latn': 'bug_Latn'},	# Buginese
    'bul' : {'cyrl': 'bul_Cyrl'},	# Bulgarian
    'cat' : {'latn': 'cat_Latn'},	# Catalan
    'ceb' : {'latn': 'ceb_Latn'},	# Cebuano
    'ces' : {'latn': 'ces_Latn'},	# Czech
    'cjk' : {'latn': 'cjk_Latn'},	# Chokwe
    'ckb' : {'arab': 'ckb_Arab'},	# Central Kurdish
    'crh' : {'latn': 'crh_Latn'},	# Crimean Tatar
    'cym' : {'latn': 'cym_Latn'},	# Welsh
    'dan' : {'latn': 'dan_Latn'},	# Danish
    'deu' : {'latn': 'deu_Latn'},	# German
    'dik' : {'latn': 'dik_Latn'},	# Southwestern Dinka
    'dyu' : {'latn': 'dyu_Latn'},	# Dyula
    'dzo' : {'tibt': 'dzo_Tibt'},	# Dzongkha
    'ell' : {'grek': 'ell_Grek'},	# Greek
    'eng' : {'latn': 'eng_Latn'},	# English
    'epo' : {'latn': 'epo_Latn'},	# Esperanto
    'est' : {'latn': 'est_Latn'},	# Estonian
    'eus' : {'latn': 'eus_Latn'},	# Basque
    'ewe' : {'latn': 'ewe_Latn'},	# Ewe
    'fao' : {'latn': 'fao_Latn'},	# Faroese
    'fij' : {'latn': 'fij_Latn'},	# Fijian
    'fin' : {'latn': 'fin_Latn'},	# Finnish
    'fon' : {'latn': 'fon_Latn'},	# Fon
    'fra' : {'latn': 'fra_Latn'},	# French
    'fur' : {'latn': 'fur_Latn'},	# Friulian
    'fuv' : {'latn': 'fuv_Latn'},	# Nigerian Fulfulde
    'gla' : {'latn': 'gla_Latn'},	# Scottish Gaelic
    'gle' : {'latn': 'gle_Latn'},	# Irish
    'glg' : {'latn': 'glg_Latn'},	# Galician
    'grn' : {'latn': 'grn_Latn'},	# Guarani
    'guj' : {'gujr': 'guj_Gujr'},	# Gujarati
    'hat' : {'latn': 'hat_Latn'},	# Haitian Creole
    'hau' : {'latn': 'hau_Latn'},	# Hausa
    'heb' : {'hebr': 'heb_Hebr'},	# Hebrew
    'hin' : {'deva': 'hin_Deva'},	# Hindi
    'hne' : {'deva': 'hne_Deva'},	# Chhattisgarhi
    'hrv' : {'latn': 'hrv_Latn'},	# Croatian
    'hun' : {'latn': 'hun_Latn'},	# Hungarian
    'hye' : {'armn': 'hye_Armn'},	# Armenian
    'ibo' : {'latn': 'ibo_Latn'},	# Igbo
    'ilo' : {'latn': 'ilo_Latn'},	# Ilocano
    'ind' : {'latn': 'ind_Latn'},	# Indonesian
    'isl' : {'latn': 'isl_Latn'},	# Icelandic
    'ita' : {'latn': 'ita_Latn'},	# Italian
    'jav' : {'latn': 'jav_Latn'},	# Javanese
    'jpn' : {'jpan': 'jpn_Jpan'},	# Japanese
    'kab' : {'latn': 'kab_Latn'},	# Kabyle
    'kac' : {'latn': 'kac_Latn'},	# Jingpho
    'kam' : {'latn': 'kam_Latn'},	# Kamba
    'kan' : {'knda': 'kan_Knda'},	# Kannada
    'kas' : {'arab': 'kas_Arab',	# Kashmiri (Arabic script)
             'deva': 'kas_Deva'},	# Kashmiri (Devanagari script)
    'kat' : {'geor': 'kat_Geor'},	# Georgian
    'knc' : {'arab': 'knc_Arab',	# Central Kanuri (Arabic script)
             'latn': 'knc_Latn'},	# Central Kanuri (Latin script)
    'kaz' : {'cyrl': 'kaz_Cyrl'},	# Kazakh
    'kbp' : {'latn': 'kbp_Latn'},	# Kabiyè
    'kea' : {'latn': 'kea_Latn'},	# Kabuverdianu
    'khm' : {'khmr': 'khm_Khmr'},	# Khmer
    'kik' : {'latn': 'kik_Latn'},	# Kikuyu
    'kin' : {'latn': 'kin_Latn'},	# Kinyarwanda
    'kir' : {'cyrl': 'kir_Cyrl'},	# Kyrgyz
    'kmb' : {'latn': 'kmb_Latn'},	# Kimbundu
    'kmr' : {'latn': 'kmr_Latn'},	# Northern Kurdish
    'kon' : {'latn': 'kon_Latn'},	# Kikongo
    'kor' : {'hang': 'kor_Hang'},	# Korean
    'lao' : {'laoo': 'lao_Laoo'},	# Lao
    'lij' : {'latn': 'lij_Latn'},	# Ligurian
    'lim' : {'latn': 'lim_Latn'},	# Limburgish
    'lin' : {'latn': 'lin_Latn'},	# Lingala
    'lit' : {'latn': 'lit_Latn'},	# Lithuanian
    'lmo' : {'latn': 'lmo_Latn'},	# Lombard
    'ltg' : {'latn': 'ltg_Latn'},	# Latgalian
    'ltz' : {'latn': 'ltz_Latn'},	# Luxembourgish
    'lua' : {'latn': 'lua_Latn'},	# Luba-Kasai
    'lug' : {'latn': 'lug_Latn'},	# Ganda
    'luo' : {'latn': 'luo_Latn'},	# Luo
    'lus' : {'latn': 'lus_Latn'},	# Mizo
    'lvs' : {'latn': 'lvs_Latn'},	# Standard Latvian
    'mag' : {'deva': 'mag_Deva'},	# Magahi
    'mai' : {'deva': 'mai_Deva'},	# Maithili
    'mal' : {'mlym': 'mal_Mlym'},	# Malayalam
    'mar' : {'deva': 'mar_Deva'},	# Marathi
    'min' : {'arab': 'min_Arab',	# Minangkabau (Arabic script)
             'latn': 'min_Latn'},	# Minangkabau (Latin script)
    'mkd' : {'cyrl': 'mkd_Cyrl'},	# Macedonian
    'plt' : {'latn': 'plt_Latn'},	# Plateau Malagasy
    'mlt' : {'latn': 'mlt_Latn'},	# Maltese
    'mni' : {'beng': 'mni_Beng'},	# Meitei (Bengali script)
    'khk' : {'cyrl': 'khk_Cyrl'},	# Halh Mongolian
    'mos' : {'latn': 'mos_Latn'},	# Mossi
    'mri' : {'latn': 'mri_Latn'},	# Maori
    'mya' : {'mymr': 'mya_Mymr'},	# Burmese
    'nld' : {'latn': 'nld_Latn'},	# Dutch
    'nno' : {'latn': 'nno_Latn'},	# Norwegian Nynorsk
    'nob' : {'latn': 'nob_Latn'},	# Norwegian Bokmål
    'npi' : {'deva': 'npi_Deva'},	# Nepali
    'nso' : {'latn': 'nso_Latn'},	# Northern Sotho
    'nus' : {'latn': 'nus_Latn'},	# Nuer
    'nya' : {'latn': 'nya_Latn'},	# Nyanja
    'oci' : {'latn': 'oci_Latn'},	# Occitan
    'gaz' : {'latn': 'gaz_Latn'},	# West Central Oromo
    'ory' : {'orya': 'ory_Orya'},	# Odia
    'pag' : {'latn': 'pag_Latn'},	# Pangasinan
    'pan' : {'guru': 'pan_Guru'},	# Eastern Panjabi
    'pap' : {'latn': 'pap_Latn'},	# Papiamento
    'pes' : {'arab': 'pes_Arab'},	# Western Persian
    'pol' : {'latn': 'pol_Latn'},	# Polish
    'por' : {'latn': 'por_Latn'},	# Portuguese
    'prs' : {'arab': 'prs_Arab'},	# Dari
    'pbt' : {'arab': 'pbt_Arab'},	# Southern Pashto
    'quy' : {'latn': 'quy_Latn'},	# Ayacucho Quechua
    'ron' : {'latn': 'ron_Latn'},	# Romanian
    'run' : {'latn': 'run_Latn'},	# Rundi
    'rus' : {'cyrl': 'rus_Cyrl'},	# Russian
    'sag' : {'latn': 'sag_Latn'},	# Sango
    'san' : {'deva': 'san_Deva'},	# Sanskrit
    'sat' : {'olck': 'sat_Olck'},	# Santali
    'scn' : {'latn': 'scn_Latn'},	# Sicilian
    'shn' : {'mymr': 'shn_Mymr'},	# Shan
    'sin' : {'sinh': 'sin_Sinh'},	# Sinhala
    'slk' : {'latn': 'slk_Latn'},	# Slovak
    'slv' : {'latn': 'slv_Latn'},	# Slovenian
    'smo' : {'latn': 'smo_Latn'},	# Samoan
    'sna' : {'latn': 'sna_Latn'},	# Shona
    'snd' : {'arab': 'snd_Arab'},	# Sindhi
    'som' : {'latn': 'som_Latn'},	# Somali
    'sot' : {'latn': 'sot_Latn'},	# Southern Sotho
    'spa' : {'latn': 'spa_Latn'},	# Spanish
    'als' : {'latn': 'als_Latn'},	# Tosk Albanian
    'srd' : {'latn': 'srd_Latn'},	# Sardinian
    'srp' : {'cyrl': 'srp_Cyrl'},	# Serbian
    'ssw' : {'latn': 'ssw_Latn'},	# Swati
    'sun' : {'latn': 'sun_Latn'},	# Sundanese
    'swe' : {'latn': 'swe_Latn'},	# Swedish
    'swh' : {'latn': 'swh_Latn'},	# Swahili
    'szl' : {'latn': 'szl_Latn'},	# Silesian
    'tam' : {'taml': 'tam_Taml'},	# Tamil
    'tat' : {'cyrl': 'tat_Cyrl'},	# Tatar
    'tel' : {'telu': 'tel_Telu'},	# Telugu
    'tgk' : {'cyrl': 'tgk_Cyrl'},	# Tajik
    'tgl' : {'latn': 'tgl_Latn'},	# Tagalog
    'tha' : {'thai': 'tha_Thai'},	# Thai
    'tir' : {'ethi': 'tir_Ethi'},	# Tigrinya
    'taq' : {'latn': 'taq_Latn',	# Tamasheq (Latin script)
             'tfng': 'taq_Tfng'},	# Tamasheq (Tifinagh script)
    'tpi' : {'latn': 'tpi_Latn'},	# Tok Pisin
    'tsn' : {'latn': 'tsn_Latn'},	# Tswana
    'tso' : {'latn': 'tso_Latn'},	# Tsonga
    'tuk' : {'latn': 'tuk_Latn'},	# Turkmen
    'tum' : {'latn': 'tum_Latn'},	# Tumbuka
    'tur' : {'latn': 'tur_Latn'},	# Turkish
    'twi' : {'latn': 'twi_Latn'},	# Twi
    'tzm' : {'tfng': 'tzm_Tfng'},	# Central Atlas Tamazight
    'uig' : {'arab': 'uig_Arab'},	# Uyghur
    'ukr' : {'cyrl': 'ukr_Cyrl'},	# Ukrainian
    'umb' : {'latn': 'umb_Latn'},	# Umbundu
    'urd' : {'arab': 'urd_Arab'},	# Urdu
    'uzn' : {'latn': 'uzn_Latn'},	# Northern Uzbek
    'vec' : {'latn': 'vec_Latn'},	# Venetian
    'vie' : {'latn': 'vie_Latn'},	# Vietnamese
    'war' : {'latn': 'war_Latn'},	# Waray
    'wol' : {'latn': 'wol_Latn'},	# Wolof
    'xho' : {'latn': 'xho_Latn'},	# Xhosa
    'ydd' : {'hebr': 'ydd_Hebr'},	# Eastern Yiddish
    'yor' : {'latn': 'yor_Latn'},	# Yoruba
    'yue' : {'hant': 'yue_Hant'},	# Yue Chinese
    'zho' : {'hans': 'zho_Hans',	# Chinese (Simplified)
             'hant': 'zho_Hant'},	# Chinese (Traditional)
    'zsm' : {'latn': 'zsm_Latn'},	# Standard Malay
    'zul' : {'latn': 'zul_Latn'}}	# Zulu


    @classmethod
    def get_code(cls, lang : str, script : str):
        if script and lang.lower() in cls._iso_to_flores200:
            if script.lower() in cls._iso_to_flores200[lang.lower()]:
                return cls._iso_to_flores200[lang.lower()][script.lower()]
            else:
                raise mpf.DetectionException(
                    f'Language/script combination ({lang}_{script}) is invalid or not supported',
                mpf.DetectionError.INVALID_PROPERTY)
        return cls._iso_default_script_flores200.get(lang.lower())
