/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2021 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2021 The MITRE Corporation                                       *
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
import org.apache.tika.langdetect.OptimaizeLangDetector;
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
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

public class TikaTextDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger log = LoggerFactory.getLogger(TikaTextDetectionComponent.class);
    private static final Map<String, String> langMap;
    private static final ObjectMapper objectMapper = new ObjectMapper();

    static {
         langMap = Collections.unmodifiableMap(initLangMap());
    }

    // Handles the case where the media is a generic type.
    public List<MPFGenericTrack>  getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
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

        Map<String,String> properties = mpfGenericJob.getJobProperties();


        // Set language filtering limit.
        int charLimit = MapUtils.getIntValue(properties, "MIN_CHARS_FOR_LANGUAGE_DETECTION", 0);

        // Store metadata as a unique track.
        // Disabled by default for format consistency.
        if (MapUtils.getBooleanValue(properties, "STORE_METADATA")) { // TODO: Test this, odp, odt and OpenDoc calc
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
        // If output exists, separate all output into separate pages.
        // Tag each page by detected language.
        if (!pageToSections.isEmpty()) {
            // Load language identifier.
            OptimaizeLangDetector identifier = new OptimaizeLangDetector();

            identifier.loadModels();

            var pageNums = pageToSections.keySet();
            int maxPage = Collections.max(pageNums);

            int maxIdLength = (int) (Math.log10(maxPage)) + 1;

            int maxSectionsOnPage = pageToSections.values().stream().mapToInt(List::size).max().getAsInt();
            int sectionIdLength = (int) (Math.log10(maxSectionsOnPage)) + 1;

            if (sectionIdLength > maxIdLength) {
                maxIdLength = sectionIdLength;
            }
            for (int p = 0; p < pageToSections.size(); p++) {
                var sections = pageToSections.get(p);

                if (sections.size() == 1 && sections.get(0).toString().isBlank()) {
                    // If LIST_ALL_PAGES is true, create empty tracks for empty pages.
                    if (listAllPages) {
                        Map<String, String> genericDetectionProperties = new HashMap<>();
                        genericDetectionProperties.put("TEXT", "");
                        genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                        genericDetectionProperties.put("PAGE_NUM", String.format("%0" + maxIdLength + "d", p + 1));
                        genericDetectionProperties.put("SECTION_NUM", String.format("%0" + maxIdLength + "d", 1));
                        MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                        tracks.add(genericTrack);
                    }
                    continue;
                }

                for (int s = 0; s < sections.size(); s++) {

                    Map<String, String> genericDetectionProperties = new HashMap<>();

                    try {
                        String text = sections.get(s).toString();

                        // By default, trim out detected text.
                        text = text.trim();
                        if (text.isEmpty()) {
                            continue;
                        }

                        genericDetectionProperties.put("TEXT", text);

                        // Process text languages.
                        if (text.length() >= charLimit) {
                            LanguageResult langResult = identifier.detect(text);
                            String language = langResult.getLanguage();

                            if (langMap.containsKey(language)) {
                                language = langMap.get(language);
                            }
                            if (!langResult.isReasonablyCertain()) {
                                language = null;
                            }
                            if (language != null && language.length() > 0) {
                                genericDetectionProperties.put("TEXT_LANGUAGE", language);
                            } else {
                                genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                            }
                        } else {
                            genericDetectionProperties.put("TEXT_LANGUAGE", "Unknown");
                        }

                    } catch (Exception e) {
                        String errorMsg = "Failed to process text detections.";
                        log.error(errorMsg, e);
                        throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_FAILED, errorMsg);
                    }

                    genericDetectionProperties.put("PAGE_NUM", String.format("%0" + maxIdLength + "d", p + 1));
                    genericDetectionProperties.put("SECTION_NUM", String.format("%0" + maxIdLength + "d", s + 1));
                    MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                    tracks.add(genericTrack);
                }
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
        map.put("el", "Greek");
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
