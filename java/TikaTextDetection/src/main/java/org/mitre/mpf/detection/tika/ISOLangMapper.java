/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
 *                                                                            *
 * Licensed under the Apache License, Version 2.0 (the "License");            *
 * you may not use this file except in compliance with the License.           *
 * You may obtain a copy of the License at                                    *
 *                                                                            *
 *    http://www.apache.org/licenses/LICENSE-2.0                              *
 *                                                                            *
 * Unless required by applicable law or agreed to in writing, software        *
 * distributed under the License is distributed on an "AS IS" BASIS,          *
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
 * See the License for the specific language governing permissions and        *
 * limitations under the License.                                             *
 ******************************************************************************/

package org.mitre.mpf.detection.tika;
import java.util.*;

public final class ISOLangMapper {
    public static final Map<String, String> langMap, isoMap;

    static {
        langMap = Collections.unmodifiableMap(initLangMap());
        isoMap = Collections.unmodifiableMap(initISO639Map());
    }

    // Map for translating from ISO 639-2 code to english description.
    private static Map<String, String> initLangMap() {
        Map<String, String> map = new HashMap<>();
        map.put("af", "Afrikaans");
        map.put("an", "Aragonese");
        map.put("ar", "Arabic");
        map.put("be", "Belarusian");
        map.put("bg", "Bulgarian");
        map.put("bn", "Bengali");
        map.put("br", "Breton");
        map.put("ca", "Catalan");
        map.put("cs", "Czech");
        map.put("cy", "Welsh");
        map.put("da", "Danish");
        map.put("de", "German");
        map.put("el", "Modern Greek");
        map.put("en", "English");
        map.put("eo", "Esperanto");
        map.put("es", "Spanish");
        map.put("et", "Estonian");
        map.put("eu", "Basque");
        map.put("fa", "Persian");
        map.put("fi", "Finnish");
        map.put("fr", "French");
        map.put("ga", "Irish");
        map.put("gl", "Galician");
        map.put("gu", "Gujarati");
        map.put("he", "Hebrew");
        map.put("hi", "Hindi");
        map.put("hr", "Croatian");
        map.put("ht", "Haitian");
        map.put("hu", "Hungarian");
        map.put("id", "Indonesian");
        map.put("is", "Icelandic");
        map.put("it", "Italian");
        map.put("ja", "Japanese");
        map.put("km", "Khmer");
        map.put("kn", "Kannada");
        map.put("ko", "Korean");
        map.put("lt", "Lithuanian");
        map.put("lv", "Latvian");
        map.put("mk", "Macedonian");
        map.put("ml", "Malayalam");
        map.put("mr", "Marathi");
        map.put("ms", "Malay");
        map.put("mt", "Maltese");
        map.put("ne", "Nepali");
        map.put("nl", "Dutch");
        map.put("no", "Norwegian");
        map.put("oc", "Occitan");
        map.put("pa", "Punjabi");
        map.put("pl", "Polish");
        map.put("pt", "Portuguese");
        map.put("ro", "Romanian");
        map.put("ru", "Russian");
        map.put("sk", "Slovak");
        map.put("sl", "Slovenian");
        map.put("so", "Somali");
        map.put("sq", "Albanian");
        map.put("sr", "Serbian");
        map.put("sv", "Swedish");
        map.put("sw", "Swahili");
        map.put("ta", "Tamil");
        map.put("te", "Telugu");
        map.put("th", "Thai");
        map.put("tl", "Tagalog");
        map.put("tr", "Turkish");
        map.put("uk", "Ukrainian");
        map.put("ur", "Urdu");
        map.put("vi", "Vietnamese");
        map.put("wa", "Walloon");
        map.put("yi", "Yiddish");
        map.put("zh-cn", "Simplified Chinese");
        map.put("zh-tw", "Traditional Chinese");

        map.put("afr", "Afrikaans");
        map.put("amh", "Amharic");
        map.put("ara", "Arabic");
        map.put("asm", "Assamese");
        map.put("ast", "Asturian");
        map.put("aze", "Azerbaijani");
        map.put("bak", "Bashkir");
        map.put("ban", "Balinese");
        map.put("bel", "Belarusian");
        map.put("ben", "Bengali");
        map.put("bih", "Bihari languages");
        map.put("bos", "Bosnian");
        map.put("bre", "Breton");
        map.put("bul", "Bulgarian");
        map.put("cat", "Catalan");
        map.put("ceb", "Cebuano");
        map.put("ces", "Czech");
        map.put("che", "Chechen");
        map.put("ckb", "Central Kurdish");
        map.put("cmn", "Mandarin Chinese");
        map.put("cym", "Welsh");
        map.put("dan", "Danish");
        map.put("deu", "German");
        map.put("div", "Dhivehi");
        map.put("ekk", "Standard Estonian");
        map.put("ell", "Modern Greek");
        map.put("eng", "English");
        map.put("epo", "Esperanto");
        map.put("est", "Estonian");
        map.put("eus", "Basque");
        map.put("fao", "Faroese");
        map.put("fas", "Persian");
        map.put("fin", "Finnish");
        map.put("fra", "French");
        map.put("fry", "Western Frisian");
        map.put("ful", "Fulah");
        map.put("gla", "Scottish Gaelic");
        map.put("gle", "Irish");
        map.put("glg", "Galician");
        map.put("gom", "Goan Konkani");
        map.put("gsw", "Swiss German");
        map.put("gug", "Paraguayan Guaraní");
        map.put("guj", "Gujarati");
        map.put("hat", "Haitian");
        map.put("hau", "Hausa");
        map.put("heb", "Hebrew");
        map.put("hin", "Hindi");
        map.put("hrv", "Croatian");
        map.put("hun", "Hungarian");
        map.put("hye", "Armenian");
        map.put("ibo", "Igbo");
        map.put("ind", "Indonesian");
        map.put("isl", "Icelandic");
        map.put("ita", "Italian");
        map.put("jav", "Javanese");
        map.put("jpn", "Japanese");
        map.put("kan", "Kannada");
        map.put("kat", "Georgian");
        map.put("kaz", "Kazakh");
        map.put("khm", "Khmer");
        map.put("kin", "Kinyarwanda");
        map.put("kir", "Kirghiz");
        map.put("knn", "Konkani");
        map.put("kor", "Korean");
        map.put("kur", "Kurdish");
        map.put("lao", "Lao");
        map.put("lat", "Latin");
        map.put("lav", "Latvian");
        map.put("lim", "Limburgan");
        map.put("lin", "Lingala");
        map.put("lit", "Lithuanian");
        map.put("ltz", "Luxembourgish");
        map.put("lug", "Ganda");
        map.put("lvs", "Standard Latvian");
        map.put("mal", "Malayalam");
        map.put("mar", "Marathi");
        map.put("mhr", "Eastern Mari");
        map.put("min", "Minangkabau");
        map.put("mkd", "Macedonian");
        map.put("mlg", "Malagasy");
        map.put("mlt", "Maltese");
        map.put("mon", "Mongolian");
        map.put("mri", "Maori");
        map.put("msa", "Malay");
        map.put("mya", "Burmese");
        map.put("nan", "Min Nan Chinese");
        map.put("nds", "Low German");
        map.put("nep", "Nepali");
        map.put("new", "Newari");
        map.put("nld", "Dutch");
        map.put("nno", "Norwegian Nynorsk");
        map.put("nob", "Norwegian Bokmål");
        map.put("nso", "Pedi");
        map.put("oci", "Occitan");
        map.put("ori", "Oriya");
        map.put("orm", "Oromo");
        map.put("pan", "Panjabi");
        map.put("pes", "Iranian Persian");
        map.put("pnb", "Western Punjabi");
        map.put("pol", "Polish");
        map.put("por", "Portuguese");
        map.put("pus", "Pushto");
        map.put("quz", "Cusco Quechua");
        map.put("roh", "Romansh");
        map.put("ron", "Romanian");
        map.put("rus", "Russian");
        map.put("san", "Sanskrit");
        map.put("sin", "Sinhala");
        map.put("slk", "Slovak");
        map.put("slv", "Slovenian");
        map.put("snd", "Sindhi");
        map.put("som", "Somali");
        map.put("spa", "Spanish");
        map.put("sqi", "Albanian");
        map.put("srd", "Sardinian");
        map.put("srp", "Serbian");
        map.put("ssw", "Swati");
        map.put("swa", "Swahili");
        map.put("swe", "Swedish");
        map.put("tam", "Tamil");
        map.put("tat", "Tatar");
        map.put("tel", "Telugu");
        map.put("tgk", "Tajik");
        map.put("tgl", "Tagalog");
        map.put("tha", "Thai");
        map.put("tsn", "Tswana");
        map.put("tuk", "Turkmen");
        map.put("tur", "Turkish");
        map.put("uig", "Uighur");
        map.put("ukr", "Ukrainian");
        map.put("urd", "Urdu");
        map.put("uzb", "Uzbek");
        map.put("vie", "Vietnamese");
        map.put("vol", "Volapük");
        map.put("war", "Waray");
        map.put("wol", "Wolof");
        map.put("xho", "Xhosa");
        map.put("yid", "Yiddish");
        map.put("yor", "Yoruba");
        map.put("zul", "Zulu");

        map.put("ben-rom", "Romanized Bengali");
        map.put("hin-rom", "Romanized Hindi");
        map.put("mya-zaw", "Burmese (Zawgyi)");
        map.put("tam-rom", "Romanized Tamil");
        map.put("tel-rom", "Romanized Telugu");
        map.put("urd-rom", "Romanized Urdu");
        map.put("zho-simp", "Simplified Chinese");
        map.put("zho-trad", "Traditional Chinese");
        return map;
    }

    // Maps individual, native language inputs to ISO639.
    private static Map<String, String> initISO639Map() {
        Map<String, String> map = new HashMap<>();
        map.put("af", "AFR");
        map.put("an", "ARG");
        map.put("ar", "ARA");
        map.put("be", "BEL");
        map.put("bg", "BUL");
        map.put("bn", "BEN");
        map.put("br", "BRE");
        map.put("ca", "CAT");
        map.put("cs", "CES");
        map.put("cy", "CYM");
        map.put("da", "DAN");
        map.put("de", "DEU");
        map.put("el", "ELL");
        map.put("en", "ENG");
        map.put("eo", "EPO");
        map.put("es", "SPA");
        map.put("et", "EST");
        map.put("eu", "EUS");
        map.put("fa", "FAS");
        map.put("fi", "FIN");
        map.put("fr", "FRA");
        map.put("ga", "GLE");
        map.put("gl", "GLG");
        map.put("gu", "GUJ");
        map.put("he", "HEB");
        map.put("hi", "HIN");
        map.put("hr", "HRV");
        map.put("ht", "HAT");
        map.put("hu", "HUN");
        map.put("id", "IND");
        map.put("is", "ISL");
        map.put("it", "ITA");
        map.put("ja", "JPN");
        map.put("km", "KHM");
        map.put("kn", "KAN");
        map.put("ko", "KOR");
        map.put("lt", "LIT");
        map.put("lv", "LAV");
        map.put("mk", "MKD");
        map.put("ml", "MAL");
        map.put("mr", "MAR");
        map.put("ms", "MSA");
        map.put("mt", "MLT");
        map.put("ne", "NEP");
        map.put("nl", "NLD");
        map.put("no", "NOR");
        map.put("oc", "OCI");
        map.put("pa", "PAN");
        map.put("pl", "POL");
        map.put("pt", "POR");
        map.put("ro", "RON");
        map.put("ru", "RUS");
        map.put("sk", "SLK");
        map.put("sl", "SLV");
        map.put("so", "SOM");
        map.put("sq", "SQI");
        map.put("sr", "SRP");
        map.put("sv", "SWE");
        map.put("sw", "SWA");
        map.put("ta", "TAM");
        map.put("te", "TEL");
        map.put("th", "THA");
        map.put("tl", "TGL");
        map.put("tr", "TUR");
        map.put("uk", "UKR");
        map.put("ur", "URD");
        map.put("vi", "VIE");
        map.put("wa", "WLN");
        map.put("yi", "YID");

        // ISO-693-3 does not differentiate between simplified and traditional Chinese script.
        // Include ISO-15924 code to differentiate, if possible.
        map.put("zh", "ZHO");
        map.put("zho", "ZHO");
        map.put("zh-cn", "ZHO-HANS");
        map.put("zh-tw", "ZHO-HANT");
        map.put("zho-simp", "ZHO-HANS");
        map.put("zho-trad", "ZHO-HANT");

        return map;
    }
}
