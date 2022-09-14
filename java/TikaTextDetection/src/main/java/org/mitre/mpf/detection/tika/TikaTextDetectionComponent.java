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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.collections4.MapUtils;
import org.apache.tika.exception.TikaException;
import org.apache.tika.language.detect.LanguageDetector;
import org.apache.tika.language.detect.LanguageConfidence;
import org.apache.tika.langdetect.optimaize.OptimaizeLangDetector;
import org.apache.tika.langdetect.opennlp.OpenNLPDetector;
import org.apache.tika.langdetect.tika.TikaLanguageDetector;

import org.apache.tika.language.detect.LanguageResult;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.mitre.mpf.component.api.detection.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.IndexOutOfBoundsException;
import java.util.*;

public class TikaTextDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger log = LoggerFactory.getLogger(TikaTextDetectionComponent.class);
    private static final Map<String, String> langMap, isoMap;
    private static final List<String> supportedLangDetectors = Arrays.asList("opennlp", "optimaize", "tika");
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private final LanguageDetector nlpDetector, optDetector, tikaDetector;

    static {
        langMap = Collections.unmodifiableMap(initLangMap());
        isoMap = Collections.unmodifiableMap(initISO639Map());
    }

    public TikaTextDetectionComponent() throws MPFComponentDetectionError {
        try {
            optDetector = new OptimaizeLangDetector();
            optDetector.loadModels();
        } catch (IOException e) {
            String errorMsg = "Error loading Optimaize Language Model.";
            log.error(errorMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_NOT_INITIALIZED, errorMsg);
        }
        try {
            tikaDetector = new TikaLanguageDetector();
            tikaDetector.loadModels();
        } catch (IOException e) {
            String errorMsg = "Error loading Tika Language Model.";
            log.error(errorMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_NOT_INITIALIZED, errorMsg);
        }
        nlpDetector = new OpenNLPDetector();
    }

    // Handles the case where the media is a generic type.
    public List<MPFGenericTrack> getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        log.info("[{}] Starting job.", mpfGenericJob.getJobName());
        log.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
            mpfGenericJob.getJobName(), mpfGenericJob.getDataUri(),
            mpfGenericJob.getJobProperties().size(), mpfGenericJob.getMediaProperties().size());

        // Specify filename for tika parsers here.
        File file = new File(mpfGenericJob.getDataUri());

        Map<Integer, List<StringBuilder>> pageToSections;
        Metadata metadata = new Metadata();
        try (FileInputStream inputstream = new FileInputStream(file)) {
            // Init parser with custom content handler for parsing text per page (PDF/PPTX).
            Parser parser = new AutoDetectParser();
            TextExtractionContentHandler handler = new TextExtractionContentHandler();
            ParseContext context = new ParseContext();
            // Parse file.
            // If the format is .pdf or .pptx, output will be divided by page/slide.
            parser.parse(inputstream, handler, metadata, context);
            pageToSections = handler.getPages();

        } catch (IOException | TikaException | SAXException e) {
            String errorMsg = String.format("Error parsing file. Filepath = %s", file);
            log.error(errorMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_MEDIA, errorMsg, e);
        }

        float confidence = -1.0f;
        List<MPFGenericTrack> tracks = new LinkedList<>();

        String contentType = metadata.get("Content-Type");
        boolean supportsPageNumbers =
                contentType.equals("application/pdf") ||
                contentType.startsWith("application/vnd.openxmlformats-officedocument.presentationml");

        Map<String,String> properties = mpfGenericJob.getJobProperties();

        // Set language filtering limit.
        int charLimit = MapUtils.getIntValue(properties, "MIN_CHARS_FOR_LANGUAGE_DETECTION", 0);
        int maxLang = MapUtils.getIntValue(properties, "MAX_REASONABLE_LANGUAGES", -1);
        int minLang = MapUtils.getIntValue(properties, "MIN_LANGUAGES", 2);

        // Store metadata as a unique track.
        // Disabled by default for format consistency.
        if (MapUtils.getBooleanValue(properties, "STORE_METADATA")) {
            Map<String, String> genericDetectionProperties = new HashMap<>();
            Map<String, String> metadataMap = new HashMap<>();

            String[] metadataKeys = metadata.names();
            for (String s: metadataKeys) {
                metadataMap.put(s, metadata.get(s));
            }

            String metadataOutput;
            try {
                metadataOutput = objectMapper.writeValueAsString(metadataMap);
            } catch (JsonProcessingException e) {
                String errorMsg = "Error writing metadata as json string.";
                log.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
            }
            genericDetectionProperties.put("METADATA", metadataOutput);
            MPFGenericTrack metadataTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
            tracks.add(metadataTrack);
        }

        boolean listAllPages = MapUtils.getBooleanValue(properties, "LIST_ALL_PAGES", false);
        String langOption = properties.getOrDefault("LANGUAGE_DETECTOR", "opennlp");

        if (!supportedLangDetectors.contains(langOption))
        {
            log.warn(langOption + " language detection option not supported. Defaulting to `opennlp` language detector.");
            langOption = "opennlp";
        }
        LanguageDetector identifier;
        if (langOption == "optimaize") {
            identifier = optDetector;
        }
        else if(langOption == "tika") {
            //TODO: It appears that the legacy tika detector has several issues processing short sections of text.
            //TODO: If not resolved soon, we should avoid adding this detector. Want to run more tests to confirm.
            identifier = tikaDetector;
        }
        else {
            // Default option.
            // Note: loadModels() call is not needed, OpenNLP models are static.
            identifier = nlpDetector;
        }

        // Separate all output into separate pages. Tag each page by detected language.
        for (int p = 0; p < pageToSections.size(); p++) {
            var sections = pageToSections.get(p);

            if (sections.size() == 1 && sections.get(0).toString().isBlank()) {
                // If LIST_ALL_PAGES is true, create empty tracks for empty pages.
                if (listAllPages) {
                    Map<String, String> genericDetectionProperties = new HashMap<>();
                    genericDetectionProperties.put("TEXT", "");
                    genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                    genericDetectionProperties.put("ISO_LANGUAGE", "UNKNOWN");
                    // Moving forward, leave new properties blank if language detection fails.
                    genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", "");
                    genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGES", "");
                    genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGE_CONFIDENCES", "");

                    if (supportsPageNumbers) {
                        genericDetectionProperties.put("PAGE_NUM", String.format("%d", p + 1));
                    } else {
                        genericDetectionProperties.put("PAGE_NUM", "-1");
                    }
                    genericDetectionProperties.put("SECTION_NUM", String.format("%d", 1));
                    tracks.add(new MPFGenericTrack(confidence, genericDetectionProperties));
                }
                continue;
            }

            for (int s = 0; s < sections.size(); s++) {

                // By default, trim out detected text.
                String text = sections.get(s).toString().trim();
                if (text.isEmpty()) {
                    continue;
                }

                Map<String, String> genericDetectionProperties = new HashMap<>();
                genericDetectionProperties.put("TEXT", text);

                // Process text languages.
                if (text.length() >= charLimit) {
                    List<LanguageResult> langResultList;
                    langResultList = identifier.detectAll(text);

                    List<String> langList = new ArrayList<String>();
                    List<String> isoList = new ArrayList<String>();
                    List<String> confidenceList = new ArrayList<String>();

                    int maxTrackLang = (maxLang > langResultList.size() || maxLang < 1) ? langResultList.size(): maxLang;
                    int minTrackLang = (minLang > langResultList.size()) ? langResultList.size() : minLang;

                    minTrackLang = (minTrackLang <= 0) ? 0 : minTrackLang;
                    maxTrackLang = (maxTrackLang < minTrackLang) ? minTrackLang: maxTrackLang;

                    for (int i=0; i < maxTrackLang; i++) {
                        LanguageResult langResult = langResultList.get(i);

                        String language = langResult.getLanguage();
                        String isoCode = language;
                        LanguageConfidence langConfidence = langResult.getConfidence();

                        if (langMap.containsKey(language)) {
                            language = langMap.get(language);
                        }
                        if (isoMap.containsKey(isoCode)){
                            isoCode = isoMap.get(isoCode);
                        }
                        if (i >= minTrackLang && !langResult.isReasonablyCertain()) {
                            break;
                        }
                        if (language == null) {
                            break;
                        }
                        langList.add(language);
                        isoList.add(isoCode);
                        confidenceList.add(langConfidence.toString());
                    }

                    if(langList.size() < 1) {
                        genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                        genericDetectionProperties.put("ISO_LANGUAGE", "UNKNOWN");
                        // Moving forward, leave new properties blank if language detection is inconclusive.
                        genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", "");
                        genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGES", "");
                        genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGE_CONFIDENCES", "");
                    } else if(langList.size() == 1) {
                        genericDetectionProperties.put("TEXT_LANGUAGE", langList.get(0));
                        genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", confidenceList.get(0));
                        genericDetectionProperties.put("ISO_LANGUAGE", isoList.get(0));
                        genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGES", "");
                        genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGE_CONFIDENCES", "");
                    } else {
                        genericDetectionProperties.put("TEXT_LANGUAGE", langList.remove(0));
                        genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", confidenceList.remove(0));
                        genericDetectionProperties.put("ISO_LANGUAGE", isoList.get(0));
                        genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGES", String.join(", ", langList));
                        genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGE_CONFIDENCES",
                                                        String.join(", ", confidenceList));
                    }
                } else {
                    genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                    genericDetectionProperties.put("ISO_LANGUAGE", "UNKNOWN");
                    genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", "");
                    genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGES", "");
                    genericDetectionProperties.put("SECONDARY_TEXT_LANGUAGE_CONFIDENCES", "");
                }
                if (supportsPageNumbers) {
                    genericDetectionProperties.put("PAGE_NUM", String.format("%d", p + 1));
                } else {
                    genericDetectionProperties.put("PAGE_NUM", "-1");
                }
                genericDetectionProperties.put("SECTION_NUM", String.format("%d", s + 1));
                tracks.add(new MPFGenericTrack(confidence, genericDetectionProperties));
            }
        }

        // If entire document is empty, generate a single track reporting no detections.
        if (tracks.isEmpty()) {
            log.warn("Empty or invalid document. No extracted text.");
        }

        log.info("[{}] Processing complete. Generated {} generic tracks.", mpfGenericJob.getJobName(), tracks.size());

        return tracks;
    }

    // Map for translating from ISO 639-2 code to english description.
    private static Map<String,String> initLangMap() {
        Map<String,String> map = new HashMap<>();
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
    private static Map<String,String> initISO639Map() {
        Map<String,String> map = new HashMap<>();
        map.put("af", "afr");
        map.put("an", "arg");
        map.put("ar", "ara");
        map.put("be", "bel");
        map.put("bg", "bul");
        map.put("bn", "ben");
        map.put("br", "bre");
        map.put("ca", "cat");
        map.put("cs", "ces");
        map.put("cy", "cym");
        map.put("da", "dan");
        map.put("de", "deu");
        map.put("el", "ell");
        map.put("en", "eng");
        map.put("eo", "epo");
        map.put("es", "spa");
        map.put("et", "est");
        map.put("eu", "eus");
        map.put("fa", "fas");
        map.put("fi", "fin");
        map.put("fr", "fra");
        map.put("ga", "gle");
        map.put("gl", "glg");
        map.put("gu", "guj");
        map.put("he", "heb");
        map.put("hi", "hin");
        map.put("hr", "hrv");
        map.put("ht", "hat");
        map.put("hu", "hun");
        map.put("id", "ind");
        map.put("is", "isl");
        map.put("it", "ita");
        map.put("ja", "jpn");
        map.put("km", "khm");
        map.put("kn", "Kan");
        map.put("ko", "kor");
        map.put("lt", "lit");
        map.put("lv", "lav");
        map.put("mk", "mkd");
        map.put("ml", "mal");
        map.put("mr", "mar");
        map.put("ms", "msa");
        map.put("mt", "mlt");
        map.put("ne", "nep");
        map.put("nl", "nld");
        map.put("no", "nor");
        map.put("oc", "oci");
        map.put("pa", "pan");
        map.put("pl", "pol");
        map.put("pt", "por");
        map.put("ro", "ron");
        map.put("ru", "rus");
        map.put("sk", "slk");
        map.put("sl", "slv");
        map.put("so", "som");
        map.put("sq", "sqi");
        map.put("sr", "srp");
        map.put("sv", "swe");
        map.put("sw", "swa");
        map.put("ta", "tam");
        map.put("te", "tel");
        map.put("th", "tha");
        map.put("tl", "tgl");
        map.put("tr", "tur");
        map.put("uk", "ukr");
        map.put("ur", "urd");
        map.put("vi", "vie");
        map.put("wa", "wln");
        map.put("yi", "yid");

        // ISO-693-3 does not differentiate between simplified and traditional Chinese script.
        // Preserving the `-simp` and `-trad` tags.
        map.put("zh-cn", "zho-simp");
        map.put("zh-tw", "zho-trad");
        map.put("zho-simp", "zho-simp");
        map.put("zho-trad", "zho-trad");
        return map;
    }

    // The TikeDetection component supports generic file types (pdfs, documents, txt, etc.).
    public boolean supports(MPFDataType mpfDataType) {
        return MPFDataType.UNKNOWN.equals(mpfDataType);
    }

    public String getDetectionType() {
        return "TEXT";
    }

    public List<MPFImageLocation> getDetections(MPFImageJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE, "Image detection not supported.");
    }

    public List<MPFVideoTrack> getDetections(MPFVideoJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE, "Video detection not supported.");
    }

    public List<MPFAudioTrack> getDetections(MPFAudioJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE, "Audio detection not supported.");
    }
}
