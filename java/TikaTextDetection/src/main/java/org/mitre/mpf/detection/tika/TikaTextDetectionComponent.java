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
    private static final List<String> supportedLangDetectors = Arrays.asList("opennlp", "optimaize");
    private static final ObjectMapper objectMapper = new ObjectMapper();
    private LanguageDetector langDetector;
    private String langDetectorSetting = "empty";

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

        if (!langDetectorSetting.equals(langOption))
        {
            langDetectorSetting = langOption;

            if (langOption.equals("optimaize")) {
                try {
                    langDetector = new OptimaizeLangDetector();
                    langDetector.loadModels();
                } catch (IOException e) {
                    String errorMsg = "Error loading Optimaize Language Model.";
                    log.error(errorMsg, e);
                    throw new MPFComponentDetectionError(MPFDetectionError.MPF_DETECTION_NOT_INITIALIZED, errorMsg);
                }
            } else {
                langDetector = new OpenNLPDetector();
            }
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
                    genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", "-1");

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
                    List<LanguageResult> langResultList = langDetector.detectAll(text);

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
                        Float langConfidence = langResult.getRawScore();

                        if (ISOLangMapper.langMap.containsKey(language)) {
                            language = ISOLangMapper.langMap.get(language);
                        }
                        if (ISOLangMapper.isoMap.containsKey(isoCode)){
                            isoCode = ISOLangMapper.isoMap.get(isoCode);
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
                        genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", "-1");

                    } else if(langList.size() == 1) {
                        genericDetectionProperties.put("TEXT_LANGUAGE", langList.get(0));
                        genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", confidenceList.get(0));
                        genericDetectionProperties.put("ISO_LANGUAGE", isoList.get(0));

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
                    genericDetectionProperties.put("TEXT_LANGUAGE_CONFIDENCE", "-1");
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
