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
import org.apache.tika.langdetect.optimaize.OptimaizeLangDetector;
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
import java.util.*;

public class TikaTextDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger log = LoggerFactory.getLogger(TikaTextDetectionComponent.class);
    private static final ObjectMapper objectMapper = new ObjectMapper();
    public static OptimaizeLangDetector identifier = new OptimaizeLangDetector();
    static {
        // Load language models.
        identifier.loadModels();
    }

    // Handles the case where the media is a generic type.
    public List<MPFGenericTrack> getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        logJobInfo(mpfGenericJob);
        float confidence = -1.0f;

        List<MPFGenericTrack> tracks = new LinkedList<MPFGenericTrack>();
        Map<String,String> properties = mpfGenericJob.getJobProperties();

        // Set language filtering limit.
        int charLimit = MapUtils.getIntValue(properties, "MIN_CHARS_FOR_LANGUAGE_DETECTION", 0);
        boolean processFF = MapUtils.getBooleanValue(properties, "LANGUAGE_ID_ONLY", false);
        String ffProps = MapUtils.getString(properties, "FEED_FORWARD_PROP_TO_PROCESS", "TEXT,TRANSCRIPT");

        MPFGenericTrack feedForwardTrack = mpfGenericJob.getFeedForwardTrack();

        if (feedForwardTrack == null) {
            if (processFF) {
                throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                        "No feed forward tracks were provided. Please set `LANGUAGE_ID_ONLY` to `false` to process both" +
                                "feed forward and generic media.");
            }
        } else {
            Map<String, String> detectionProperties =  feedForwardTrack.getDetectionProperties();
            getLanguageDetection(ffProps, detectionProperties, charLimit);
            tracks.add(feedForwardTrack);

            // If the generic media was already processed by another component and we wish to skip text extraction.
            if (processFF) {
                return tracks;
            }
        }

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

        String contentType = metadata.get("Content-Type");
        boolean supportsPageNumbers =
                contentType.equals("application/pdf") ||
                contentType.startsWith("application/vnd.openxmlformats-officedocument.presentationml");

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
                getLanguageDetection("TEXT", genericDetectionProperties, charLimit);
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

    // The TikeDetection component supports generic file types (pdfs, documents, txt, etc.).
    public boolean supports(MPFDataType mpfDataType) {
        return MPFDataType.UNKNOWN.equals(mpfDataType);
    }

    public String getDetectionType() {
        return "TEXT";
    }

    private void logJobInfo(MPFJob job) {
        log.info("[{}] Starting job.", job.getJobName());
        log.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
                job.getJobName(), job.getDataUri(),
                job.getJobProperties().size(), job.getMediaProperties().size());
    }

    private void getLanguageDetection(String checkPropertiesSetting, Map<String, String> detectionProperties, int charLimit) {
        List<String> rawProperties = Arrays.asList(checkPropertiesSetting.split("\\s*,\\s*"));
        for (String rawProp: rawProperties) {
            String prop = rawProp.trim();
            if (detectionProperties.containsKey(prop)) {
                String text = detectionProperties.get(prop);
                // Process text languages.
                if (text.length() >= charLimit) {
                    LanguageResult langResult = identifier.detect(text);
                    String language = langResult.getLanguage();
                    String isoCode = language;

                    if (ISOLangMapper.langMap.containsKey(language)) {
                        language = ISOLangMapper.langMap.get(language);
                    }
                    if (ISOLangMapper.isoMap.containsKey(isoCode)){
                        isoCode = ISOLangMapper.isoMap.get(isoCode);
                        isoCode = isoCode.toUpperCase();
                    }
                    if (!langResult.isReasonablyCertain()) {
                        language = null;
                        isoCode = null;
                    }
                    if (language != null && language.length() > 0) {
                        detectionProperties.put("TEXT_LANGUAGE", language);
                        detectionProperties.put("ISO_LANGUAGE", isoCode);
                    } else {
                        detectionProperties.put("TEXT_LANGUAGE", "Unknown");
                        detectionProperties.put("ISO_LANGUAGE", "UNKNOWN");
                    }
                } else {
                    detectionProperties.put("TEXT_LANGUAGE", "Unknown");
                    detectionProperties.put("ISO_LANGUAGE", "UNKNOWN");
                }
                break;
            }
        }
    }

    private <T> List<T> processFFJob(MPFJob job, T feedForwardTrack, Map<String, String> detectionProperties) throws MPFComponentDetectionError {
        List<T> tracks = new LinkedList<T>();
        Map<String,String> properties = job.getJobProperties();
        int charLimit = MapUtils.getIntValue(properties, "MIN_CHARS_FOR_LANGUAGE_DETECTION", 0);
        String ffProps = properties.getOrDefault("FEED_FORWARD_PROP_TO_PROCESS", "TEXT,TRANSCRIPT");

        if (feedForwardTrack == null) {
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                    "This component can process feed-forward jobs for non-generic media, " +
                            "but no feed forward tracks were provided.");
        }

        //Map<String, String> detectionProperties = feedForwardTrack.getDetectionProperties();
        getLanguageDetection(ffProps, detectionProperties, charLimit);
        tracks.add(feedForwardTrack);
        return tracks;
    }

    public List<MPFImageLocation> getDetections(MPFImageJob job) throws MPFComponentDetectionError {
        logJobInfo(job);
        MPFImageLocation feedForwardTrack = job.getFeedForwardLocation();
        return processFFJob(job, feedForwardTrack, feedForwardTrack.getDetectionProperties());
    }

    public List<MPFVideoTrack> getDetections(MPFVideoJob job) throws MPFComponentDetectionError {
        logJobInfo(job);
        MPFVideoTrack feedForwardTrack = job.getFeedForwardTrack();
        return processFFJob(job, feedForwardTrack, feedForwardTrack.getDetectionProperties());
    }

    public List<MPFAudioTrack> getDetections(MPFAudioJob job) throws MPFComponentDetectionError {
        logJobInfo(job);
        MPFAudioTrack feedForwardTrack = job.getFeedForwardTrack();
        return processFFJob(job, feedForwardTrack, feedForwardTrack.getDetectionProperties());
    }
}
