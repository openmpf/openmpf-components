/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
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
import com.fasterxml.jackson.databind.JsonNode;
import org.apache.commons.collections4.MapUtils;
import org.apache.tika.langdetect.OptimaizeLangDetector;
import org.apache.tika.language.detect.LanguageResult;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.ObjectReader;
import com.fasterxml.jackson.core.type.TypeReference;


import org.mitre.mpf.component.api.detection.*;
import org.mitre.mpf.component.api.detection.util.MPFEnvironmentVariablePathExpander;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.*;
import java.util.*;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class TikaTextDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger LOG = LoggerFactory.getLogger(TikaTextDetectionComponent.class);
    private static final Map<String, String> langMap;
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private static final ObjectReader objectReader = objectMapper.readerFor(new TypeReference<Map<String, List<JsonNode>>>(){});

    static {
         langMap = Collections.unmodifiableMap(initLangMap());
    }

    // Handles the case where the media is a generic type.
    public List<MPFGenericTrack>  getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        LOG.info("[{}] Starting job.", mpfGenericJob.getJobName());
        LOG.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
            mpfGenericJob.getJobName(), mpfGenericJob.getDataUri(),
            mpfGenericJob.getJobProperties().size(), mpfGenericJob.getMediaProperties().size());

        // =========================
        // Tika Detection
        // =========================

        // Specify filename for tika parsers here.
        File file = new File(mpfGenericJob.getDataUri());


        List<StringBuilder> pageOutput = new ArrayList<StringBuilder>();
        Metadata metadata = new Metadata();
        try (FileInputStream inputstream = new FileInputStream(file)) {
            // Init parser with custom content handler for parsing text per page (PDF/PPTX).
            Parser parser = new AutoDetectParser();
            TextExtractionContentHandler handler = new TextExtractionContentHandler();
            ParseContext context = new ParseContext();
            // Parse file.
            // If the format is .pdf or .pptx, output will be divided by page/slide.
            parser.parse(inputstream, handler, metadata, context);
            pageOutput = handler.getPages();

        } catch (Exception e) {
            String errorMsg = String.format("Error parsing file. Filepath = %s", file.toString());
            LOG.error(errorMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
        }

        float confidence = -1.0f;
        List<MPFGenericTrack> tracks = new LinkedList<MPFGenericTrack>();

        Map<String,String> properties = mpfGenericJob.getJobProperties();

        // Acquire tag file.
        String tagFile = MapUtils.getString(properties, "TAGGING_FILE");
        if (tagFile == null) {
            String runDirectory = this.getRunDirectory();
            if (runDirectory == null || runDirectory.length() < 1) {
                runDirectory = "../plugins";
            }
            tagFile = runDirectory + "/TikaDetection/config/text-tags.json";
        }
        tagFile = MPFEnvironmentVariablePathExpander.expand(tagFile);

        // Set language filtering limit.
        int charLimit = MapUtils.getIntValue(properties, "MIN_CHARS_FOR_LANGUAGE_DETECTION", 0);

        // Store metadata as a unique track.
        // Disabled by default for format consistency.
        if (MapUtils.getBooleanValue(properties, "STORE_METADATA")) {
            Map<String, String> genericDetectionProperties = new HashMap<String, String>();
            Map<String, String> metadataMap = new HashMap<String, String>();

            String[] metadataKeys = metadata.names();
            for (String s: metadataKeys) {
                metadataMap.put(s, metadata.get(s));
            }

            String metadataOutput;
            try {
                metadataOutput = objectMapper.writeValueAsString(metadataMap);
            } catch (JsonProcessingException e) {
                String errorMsg = "Error writing metadata as json string.";
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
            }
            genericDetectionProperties.put("METADATA", metadataOutput);
            MPFGenericTrack metadataTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
            tracks.add(metadataTrack);
        }

        // If output exists, separate all output into separate pages.
        // Tag each page by detected language.
        if (pageOutput.size() >= 1) {
            // Load language identifier.
            OptimaizeLangDetector identifier = new OptimaizeLangDetector();
            Map<String, List<JsonNode>>  regexTags;

            try {
                identifier.loadModels();
            } catch (IOException e) {
                String errorMsg = "Failed to load language models.";
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_FAILED, errorMsg, e);
            }

            // Parse Tag File.
            try{
                JsonNode jsonRoot = objectMapper.readTree(new File(tagFile));
                JsonNode regexNode = jsonRoot.get("TAGS_BY_REGEX");

                regexTags = objectReader.readValue(regexNode);

            } catch (FileNotFoundException e) {
                String errorMsg = String.format("Tag file %s not found!", tagFile);
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
            } catch (JsonParseException e) {
                String errorMsg = String.format("Error parsing tag file.");
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
            } catch (IOException e) {
                String errorMsg = String.format("I/O Error.");
                LOG.error(errorMsg, e);
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_DATAFILE, errorMsg);
            }

            int pageIDLen = (int) (java.lang.Math.log10(pageOutput.size())) + 1;
            for (int i = 0; i < pageOutput.size(); i++) {

                Map<String, String> genericDetectionProperties = new HashMap<String, String>();
                List<String> tagsList = new ArrayList<String>();
                Map<String, Set<String>> triggerWordsMap = new LinkedHashMap<String, Set<String>>();

                // Split out non-alphanumeric characters.
                String pageText = pageOutput.get(i).toString();
                String pageTextLower = pageText.toLowerCase();
                char[] pageTextChar = pageText.toCharArray();

                try {

                    for (Map.Entry<String, List<JsonNode>> entry : regexTags.entrySet()) {
                        String key = entry.getKey();
                        List<JsonNode> tagList = entry.getValue();
                        boolean keyFound = false;
                        for (JsonNode regexEntry: tagList) {
                            String regex = "";
                            boolean caseSensitive = false;
                            if (regexEntry.isTextual()) {
                                // Legacy JSON processing.
                                // Legacy regex patterns in the JSON tags file are listed as follows:
                                //
                                // "TAGS_BY_REGEX": {
                                //    "vehicle-tag-legacy-format": [
                                //        "auto",
                                //        "car"
                                //    ],
                                //  ...
                                // }

                                regex = regexEntry.textValue();
                            } else {
                                // Standard JSON format processing.
                                // Standard JSON regex patterns are listed as follows:
                                //
                                // "TAGS_BY_REGEX": {
                                //    "vehicle-tag-standard-format": [
                                //      {"pattern": "auto"},
                                //      {"pattern": "car"}
                                //    ],
                                //  ...
                                // }

                                JsonNode pattern = regexEntry.get("pattern");
                                JsonNode caseSens = regexEntry.get("caseSensitive");
                                if (pattern != null && pattern.isTextual()) {
                                    regex = pattern.textValue();
                                }
                                if (caseSens != null && caseSens.isBoolean()) {
                                    caseSensitive = caseSens.asBoolean();
                                }
                            }
                            Matcher m;
                            Pattern p;
                            if (caseSensitive) {
                                p = Pattern.compile(regex);
                            } else {
                                p = Pattern.compile(regex, Pattern.CASE_INSENSITIVE);
                            }
                            m = p.matcher(pageText);

                            while (m.find()) {
                                keyFound = true;
                                Set<String> triggerWordOffsets;
                                int start = m.start(), end = m.end();

                                // Trim trigger words of whitespace.
                                for (int c = start; c < end; c++) {
                                    if (!Character.isWhitespace(pageTextChar[c])) {
                                        break;
                                    } else {
                                        start++;
                                    }
                                }
                                for (int c = end; c > start; c--) {
                                    if (!Character.isWhitespace(pageTextChar[c - 1])) {
                                        break;
                                    } else {
                                        end--;
                                    }
                                }

                                String triggerWord = pageText.substring(start, end).replaceAll(";", "[;]");
                                String offset;
                                if ( start != (end - 1)) {
                                    // Offset for words and phrases.
                                    offset = String.format("%d-%d", start, end - 1);
                                } else {
                                    // Offset for a single character.
                                    offset = String.format("%d", start);
                                }

                                if (triggerWordsMap.containsKey(triggerWord)){
                                    triggerWordOffsets = triggerWordsMap.get(triggerWord);
                                } else {
                                    triggerWordOffsets = new TreeSet<String>();
                                    triggerWordsMap.put(triggerWord, triggerWordOffsets);
                                }
                                triggerWordOffsets.add(offset);
                                if (!MapUtils.getBooleanValue(properties, "FULL_REGEX_SEARCH", true)) {
                                    break;
                                }
                            }
                            if (!MapUtils.getBooleanValue(properties, "FULL_REGEX_SEARCH", true) && keyFound) {
                                break;
                            }
                        }
                        if (keyFound) {
                            tagsList.add(key);
                        }
                    }

                    String tagsListResult = String.join("; ", tagsList);
                    Set<String> triggerWordsSet = triggerWordsMap.keySet();
                    String triggerWordsResult = String.join("; ", triggerWordsSet);
                    
                    ArrayList<String> offsetsList = new ArrayList();
                    for (String key: triggerWordsSet) {
                        offsetsList.add(String.join(", ",triggerWordsMap.get(key)));
                    }

                    String triggerOffsetsResult = String.join("; ", offsetsList);
                    String textDetect = pageOutput.get(i).toString();

                    // By default, trim out detected text.
                    textDetect = textDetect.trim();


                    if (textDetect.length() > 0) {
                        genericDetectionProperties.put("TEXT", textDetect);
                    }
                    else{
                        if (!MapUtils.getBooleanValue(properties, "LIST_ALL_PAGES", false)) {
                            continue;
                        }
                        genericDetectionProperties.put("TEXT", "");
                        genericDetectionProperties.put("TEXT_LANGUAGE",  "Unknown");
                        genericDetectionProperties.put("TAGS",  "");
                        genericDetectionProperties.put("TRIGGER_WORDS", "");
                        genericDetectionProperties.put("TRIGGER_WORDS_OFFSET", "");
                    }

                    // Process text languages.
                    if (textDetect.length() >= charLimit) {
                        LanguageResult langResult = identifier.detect(textDetect);
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
                        genericDetectionProperties.put("TEXT_LANGUAGE",  "Unknown");
                    }

                    if (tagsListResult.length() > 0) {
                        genericDetectionProperties.put("TAGS", tagsListResult);
                        genericDetectionProperties.put("TRIGGER_WORDS", triggerWordsResult);
                        genericDetectionProperties.put("TRIGGER_WORDS_OFFSET", triggerOffsetsResult);
                    } else {
                        genericDetectionProperties.put("TAGS", "");
                        genericDetectionProperties.put("TRIGGER_WORDS", "");
                        genericDetectionProperties.put("TRIGGER_WORDS_OFFSET", "");
                    }
                } catch (Exception e) {
                    String errorMsg = String.format("Failed to process text detections.");
                    LOG.error(errorMsg, e);
                    throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_FAILED, errorMsg);
                }



                genericDetectionProperties.put("PAGE_NUM",String.format("%0" + String.valueOf(pageIDLen) + "d", i + 1));
                MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                tracks.add(genericTrack);
            }
        }
        // If entire document is empty, generate a single track reporting no detections.
        if (tracks.size() == 0) {
            LOG.warn("Empty or invalid document. No extracted text.");
        }

        LOG.info("[{}] Processing complete. Generated {} generic tracks.", mpfGenericJob.getJobName(), tracks.size());

        return tracks;
    }

    // Map for translating from ISO 639-2 code to english description.
    private static Map<String,String> initLangMap() {
        Map<String,String> map = new HashMap<String,String>();
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
