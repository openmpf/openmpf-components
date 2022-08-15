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

    private static final Map<String, String> langMap;
    private static final List<String> supportedLangDetectors = Arrays.asList("opennlp", "optimaize", "tika");
    private static final ObjectMapper objectMapper = new ObjectMapper();

    static {
         langMap = Collections.unmodifiableMap(convertISO639());
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
        String langOption = properties.getOrDefault("LANG_DETECTOR", "opennlp");
        boolean filterReasonableDetections = MapUtils.getBooleanValue(properties, "LANG_DETECTOR_FILTER", false);

        if (!supportedLangDetectors.contains(langOption))
        {
            log.warn(langOption + " language option not supported. Defaulting to `opennlp` language detector.");
        }
        LanguageDetector identifier;
        if (langOption == "optimaize") {
            identifier = new OptimaizeLangDetector();
            try {
                identifier.loadModels();
            } catch (IOException e) {
                String errorMsg = "Error loading Optimaize Language Model.";
                log.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_OTHER_DETECTION_ERROR_TYPE, errorMsg);
            }
        }
        else if(langOption == "tika") {
            //TODO: It appears that the legaacy tika detector has several issues procssing short sections of text.
            //TODO: If not resolved soon, we should avoid adding this detector. Want to run more tests to confirm.
            identifier = new TikaLanguageDetector();
            try {
                identifier.loadModels();
            } catch (IOException e) {
                String errorMsg = "Error loading Tika Language Model.";
                log.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_OTHER_DETECTION_ERROR_TYPE, errorMsg);
            }
        }
        else {
            // Default option.
            // Note: loadModels() call is not needed, OpenNLP models are static.
            identifier = new OpenNLPDetector();
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
                    if (supportsPageNumbers) {
                        genericDetectionProperties.put("PAGE_NUM", String.format("%d", p + 1));
                    } else {
                        genericDetectionProperties.put("PAGE_NUM", "-1");
                    }
                    genericDetectionProperties.put("SECTION_NUM", String.format("%d", 1));
                    MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                    tracks.add(genericTrack);
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
                    LanguageResult langResult;
                    String language;
                    try{
                        langResult = identifier.detect(text);
                        language = langResult.getLanguage();
                        if (langMap.containsKey(language)) {
                            language = langMap.get(language);
                        }
                        if (!langResult.isReasonablyCertain() && filterReasonableDetections) {
                            language = null;
                        }
                    } catch (IndexOutOfBoundsException e) {
                        // TODO: Resolve if possible.
                        // Right now, the legacy Tika detector is throwing index errors with short snippets of text.
                        // the other two detectors are able to detect text properly and do not throw this error.
                        language = "Unknown";
                        String errorMsg = "Error detecting language using " + langOption + " detector.";
                        log.error(errorMsg, e);
                    }
                    if (language != null && language.length() > 0) {
                        genericDetectionProperties.put("TEXT_LANGUAGE", language);
                    } else {
                        genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                    }
                } else {
                    genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                }

                if (supportsPageNumbers) {
                    genericDetectionProperties.put("PAGE_NUM", String.format("%d", p + 1));
                } else {
                    genericDetectionProperties.put("PAGE_NUM", "-1");
                }
                genericDetectionProperties.put("SECTION_NUM", String.format("%d", s + 1));
                MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                tracks.add(genericTrack);
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
    private static Map<String,String> convertISO639() {
        Map<String,String> map = new HashMap<>();
        map.put("af", "Afrikaans");
        map.put("an", "Aragonese");
        map.put("ar", "Arabic");
        map.put("ast", "Asturian");
        map.put("be", "Belarusian");
        map.put("br", "Breton");
        map.put("bg", "Bulgarian");
        map.put("bn", "Bengali");
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

        map.put("bih", "Bihari languages");
        map.put("hun", "Hungarian");
        map.put("heb", "Hebrew");
        map.put("fin", "Finnish");
        map.put("gsw", "Swiss German");
        map.put("tuk", "Turkmen");
        map.put("bak", "Bashkir");
        map.put("mon", "Mongolian");
        map.put("ban", "Balinese");
        map.put("pus", "Pushto");
        map.put("tur", "Turkish");
        map.put("vie", "Vietnamese");
        map.put("fra", "French");
        map.put("oci", "Occitan");
        map.put("bre", "Breton");
        map.put("fao", "Faroese");
        map.put("deu", "German");
        map.put("ssw", "Swati");
        map.put("fas", "Persian");
        map.put("tel", "Telugu");
        map.put("nan", "Min Nan Chinese");
        map.put("yor", "Yoruba");
        map.put("gla", "Scottish Gaelic");
        map.put("jav", "Javanese");
        map.put("pes", "Iranian Persian");
        map.put("gle", "Irish");
        map.put("glg", "Galician");
        map.put("epo", "Esperanto");
        map.put("pnb", "Western Punjabi");
        map.put("lvs", "Standard Latvian");
        map.put("fry", "Western Frisian");
        map.put("slk", "Slovak");
        map.put("mya", "Burmese");
        map.put("mhr", "Eastern Mari");
        map.put("gug", "Paraguayan Guaraní");
        map.put("slv", "Slovenian");
        map.put("guj", "Gujarati");
        map.put("ceb", "Cebuano");
        map.put("cmn", "Mandarin Chinese");
        map.put("kur", "Kurdish");
        map.put("nso", "Pedi");
        map.put("pol", "Polish");
        map.put("aze", "Azerbaijani");
        map.put("ces", "Czech");
        map.put("ara", "Arabic");
        map.put("uig", "Uighur");
        map.put("por", "Portuguese");
        map.put("min", "Minangkabau");
        map.put("yid", "Yiddish");
        map.put("tgl", "Tagalog");
        map.put("tgk", "Tajik");
        map.put("mal", "Malayalam");
        map.put("uzb", "Uzbek");
        map.put("mar", "Marathi");
        map.put("mri", "Maori");
        map.put("urd", "Urdu");
        map.put("nld", "Dutch");
        map.put("snd", "Sindhi");
        map.put("knn", "Konkani");
        map.put("tha", "Thai");
        map.put("hye", "Armenian");
        map.put("ibo", "Igbo");
        map.put("bul", "Bulgarian");
        map.put("asm", "Assamese");
        map.put("msa", "Malay");
        map.put("nds", "Low German");
        map.put("ful", "Fulah");
        map.put("swa", "Swahili");
        map.put("xho", "Xhosa");
        map.put("swe", "Swedish");
        map.put("isl", "Icelandic");
        map.put("ast", "Asturian");
        map.put("ekk", "Standard Estonian");
        map.put("gom", "Goan Konkani");
        map.put("est", "Estonian");
        map.put("mkd", "Macedonian");
        map.put("bel", "Belarusian");
        map.put("ben", "Bengali");
        map.put("hin", "Hindi");
        map.put("kor", "Korean");
        map.put("dan", "Danish");
        map.put("lin", "Lingala");
        map.put("div", "Dhivehi");
        map.put("som", "Somali");
        map.put("zul", "Zulu");
        map.put("rus", "Russian");
        map.put("lim", "Limburgan");
        map.put("wol", "Wolof");
        map.put("lit", "Lithuanian");
        map.put("ita", "Italian");
        map.put("nep", "Nepali");
        map.put("hat", "Haitian");
        map.put("lao", "Lao");
        map.put("pan", "Panjabi");
        map.put("ukr", "Ukrainian");
        map.put("hau", "Hausa");
        map.put("lat", "Latin");
        map.put("lav", "Latvian");
        map.put("tam", "Tamil");
        map.put("che", "Chechen");
        map.put("new", "Newari");
        map.put("ell", "Modern Greek");
        map.put("spa", "Spanish");
        map.put("tat", "Tatar");
        map.put("mlg", "Malagasy");
        map.put("hrv", "Croatian");
        map.put("nno", "Norwegian Nynorsk");
        map.put("khm", "Khmer");
        map.put("mlt", "Maltese");
        map.put("cym", "Welsh");
        map.put("amh", "Amharic");
        map.put("nob", "Norwegian Bokmål");
        map.put("eus", "Basque");
        map.put("bos", "Bosnian");
        map.put("roh", "Romansh");
        map.put("sqi", "Albanian");
        map.put("tsn", "Tswana");
        map.put("ron", "Romanian");
        map.put("kin", "Kinyarwanda");
        map.put("vol", "Volapük");
        map.put("kir", "Kirghiz");
        map.put("cat", "Catalan");
        map.put("quz", "Cusco Quechua");
        map.put("sin", "Sinhala");
        map.put("kan", "Kannada");
        map.put("ind", "Indonesian");
        map.put("eng", "English");
        map.put("kat", "Georgian");
        map.put("san", "Sanskrit");
        map.put("srd", "Sardinian");
        map.put("kaz", "Kazakh");
        map.put("ori", "Oriya");
        map.put("jpn", "Japanese");
        map.put("war", "Waray");
        map.put("orm", "Oromo");
        map.put("afr", "Afrikaans");
        map.put("srp", "Serbian");
        map.put("ltz", "Luxembourgish");
        map.put("ckb", "Central Kurdish");
        map.put("lug", "Ganda");
        map.put("ben-rom", "Romanized Bengali");
        map.put("urd-rom", "Romanized Urdu");
        map.put("zho-simp", "Simplified Chinese");
        map.put("hin-rom", "Romanized Hindi");
        map.put("mya-zaw", "Burmese (Zawgyi)");
        map.put("zho-trad", "Traditional Chinese");
        map.put("tel-rom", "Romanized Telugu");
        map.put("tam-rom", "Romanized Tamil");

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
