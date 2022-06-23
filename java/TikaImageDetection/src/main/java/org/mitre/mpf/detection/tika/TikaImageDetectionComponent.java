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

import org.apache.tika.config.TikaConfig;
import org.apache.tika.exception.TikaException;
import org.apache.tika.extractor.EmbeddedDocumentExtractor;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.pdf.PDFParserConfig;
import org.mitre.mpf.component.api.detection.*;
import org.mitre.mpf.component.api.detection.util.MPFEnvironmentVariablePathExpander;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;

import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

public class TikaImageDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger log = LoggerFactory.getLogger(TikaImageDetectionComponent.class);

    private String configDirectory;

    private static void setPdfConfig(ParseContext context) {

        PDFParserConfig pdfConfig = new PDFParserConfig();
        pdfConfig.setExtractInlineImages(true);
        pdfConfig.setExtractUniqueInlineImagesOnly(false);
        context.set(PDFParserConfig.class, pdfConfig);
    }

    public void setConfigDirectory(String configDirectory) {
        this.configDirectory = configDirectory;
    }

    public void init() {
        if (configDirectory == null) {
            configDirectory = getRunDirectory() + "/TikaImageDetection/config";
        }
    }

    private LinkedHashMap<String, TreeSet<String>> parseDocument(final String input, final String docPath,
                                                                 final boolean separatePages)
            throws MPFComponentDetectionError {
        TikaConfig config;
        String configPath = configDirectory + "/tika-config.xml";

        try {
            config = new TikaConfig(configPath);
        } catch (SAXException | IOException | TikaException e) {
            String errMsg = "Failed to load tika config file: " + configPath;
            log.error(errMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_OPEN_DATAFILE, errMsg);
        }

        AutoDetectParser parser = new AutoDetectParser(config);
        ContentHandler handler = new ImageExtractionContentHandler();
        Metadata metadata = new Metadata();
        ParseContext context = new ParseContext();
        EmbeddedDocumentExtractor embeddedDocumentExtractor = new EmbeddedContentExtractor(docPath, separatePages);
        String errMsg = ((EmbeddedContentExtractor) embeddedDocumentExtractor).init(log);

        if( errMsg.length() > 0) {
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_FILE_WRITE_ERROR, errMsg);
        }

        context.set(EmbeddedDocumentExtractor.class, embeddedDocumentExtractor);
        context.set(AutoDetectParser.class, parser);
        setPdfConfig(context);
        try {
            InputStream stream = new FileInputStream(input);
            parser.parse(stream, handler, metadata, context);
        } catch (FileNotFoundException e) {
            errMsg = "Error opening file at : " + docPath;
            log.error(errMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_OPEN_MEDIA, errMsg);
        } catch (SAXException | IOException | TikaException e) {
            errMsg = "Error processing file at : " + docPath;
            log.error(errMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_MEDIA,
                    "Error processing file at : " + docPath);
        }
        EmbeddedContentExtractor ex = ((EmbeddedContentExtractor) embeddedDocumentExtractor);
        return ex.getImageMap();
    }

    // Handles the case where the media is a generic type.
    public List<MPFGenericTrack>  getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        log.info("[{}] Starting job.", mpfGenericJob.getJobName());
        log.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
                mpfGenericJob.getJobName(), mpfGenericJob.getDataUri(),
                mpfGenericJob.getJobProperties().size(), mpfGenericJob.getMediaProperties().size());

        // =========================
        // Tika Detection
        // =========================

        String defaultSavePath = "$MPF_HOME/share/tmp/derivative-media";

        Map<String,String> properties = mpfGenericJob.getJobProperties();
        boolean separatePages = false;

        if (properties.get("SAVE_PATH") != null) {
            defaultSavePath = properties.get("SAVE_PATH");
        }
        defaultSavePath = MPFEnvironmentVariablePathExpander.expand(defaultSavePath);

        if (properties.get("ORGANIZE_BY_PAGE") != null) {
            separatePages = Boolean.valueOf(properties.get("ORGANIZE_BY_PAGE"));
        }

        if (mpfGenericJob.getJobName().length() != 0) {
            String jobId = mpfGenericJob.getJobName().split(":")[0];
            jobId = Pattern.compile("job"
                    , Pattern.LITERAL | Pattern.CASE_INSENSITIVE).matcher(jobId).replaceAll("");
            jobId = jobId.trim();
            defaultSavePath += "/" + jobId;
        }

        List tracks = new LinkedList<MPFGenericTrack>();
        float confidence = -1.0f;


        LinkedHashMap<String, TreeSet<String>> imageMap =
                parseDocument(mpfGenericJob.getDataUri(), defaultSavePath, separatePages);
        if (imageMap.size() > 0){
            for(Map.Entry<String, TreeSet<String>> entry : imageMap.entrySet()) {
            Map genericDetectionProperties = new HashMap<String, String>();
                genericDetectionProperties.put("DERIVATIVE_MEDIA_TEMP_PATH", entry.getKey());
                genericDetectionProperties.put("PAGE_NUM", String.join("; ", entry.getValue()));

                MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
                tracks.add(genericTrack);
            }
        }
        else {
            // If no images were found at all, wipe out empty tracks.
            log.info("No images detected in document");
            tracks.clear();
        }
        deleteEmptySubDirectories(new File(defaultSavePath + "/tika-extracted"));

        log.info("[{}] Processing complete. Generated {} generic tracks.",
                mpfGenericJob.getJobName(),
                tracks.size());

        return tracks;
    }

    // Recursively walks through the given directory and clears out empty subdirectories
    // If all subdirectories are empty and nothing else is contained, the given directory is also deleted.
    private void deleteEmptySubDirectories(File path){
        if (path.isDirectory()) {
            for(File file: path.listFiles()) {
                if (file.isDirectory()) {
                    deleteEmptySubDirectories(file);
                }
            }
            if (path.list().length == 0) {
                path.delete();
            }
        }
    }

    // The TikeDetection component supports generic file types (pdfs, documents, txt, etc.).
    public boolean supports(MPFDataType mpfDataType) {
        return mpfDataType != null && MPFDataType.UNKNOWN.equals(mpfDataType);
    }

    public String getDetectionType() {
        return "MEDIA";
    }

    public List<MPFImageLocation> getDetections(MPFImageJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                "Image detection not supported.");
    }

    public List<MPFVideoTrack> getDetections(MPFVideoJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                "Video detection not supported.");
    }

    public List<MPFAudioTrack> getDetections(MPFAudioJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                "Audio detection not supported.");
    }
}
