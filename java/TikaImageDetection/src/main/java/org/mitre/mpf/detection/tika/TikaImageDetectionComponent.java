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
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileSystemException;
import java.nio.file.Path;
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class TikaImageDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger LOG = LoggerFactory.getLogger(TikaImageDetectionComponent.class);

    private Path _configDirectory;

    private static void setPdfConfig(ParseContext context) {
        PDFParserConfig pdfConfig = new PDFParserConfig();
        pdfConfig.setExtractInlineImages(true);
        pdfConfig.setExtractUniqueInlineImagesOnly(false);
        context.set(PDFParserConfig.class, pdfConfig);
    }

    public void setConfigDirectory(String configDirectory) {
        _configDirectory = Path.of(configDirectory);
    }

    @Override
    public void init() {
        if (_configDirectory == null) {
            _configDirectory = Path.of(getRunDirectory(), "TikaImageDetection/config");
        }
    }

    private Map<Path, SortedSet<Integer>> parseDocument(String input, String docPath,
                                                        boolean separatePages, Metadata metadata)
            throws MPFComponentDetectionError {
        TikaConfig config;
        String configPath = _configDirectory + "/tika-config.xml";

        try {
            config = new TikaConfig(configPath);
        } catch (SAXException | IOException | TikaException e) {
            String errMsg = "Failed to load tika config file: " + configPath;
            LOG.error(errMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_OPEN_DATAFILE, errMsg);
        }

        AutoDetectParser parser = new AutoDetectParser(config);
        ContentHandler handler = new PageNumberExtractionContentHandler();
        ParseContext context = new ParseContext();

        var embeddedDocumentExtractor = separatePages
                ? new SeparateDirImageExtractor(docPath)
                : new SameDirImageExtractor(docPath);

        context.set(EmbeddedDocumentExtractor.class, embeddedDocumentExtractor);
        context.set(AutoDetectParser.class, parser);
        setPdfConfig(context);
        try {
            InputStream stream = new FileInputStream(input);
            parser.parse(stream, handler, metadata, context);
        } catch (FileAlreadyExistsException e) {
            throw new MPFComponentDetectionError(
                MPFDetectionError.MPF_FILE_WRITE_ERROR,
                String.format("Failed to write file \"%s\" because that file already existed", e.getFile()),
                e);
        } catch (FileNotFoundException e) {
            var errMsg = "Error opening file at : " + docPath;
            LOG.error(errMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_OPEN_MEDIA, errMsg);
        } catch (SAXException | IOException | TikaException e) {
            if (e.getCause() instanceof FileAlreadyExistsException) {
                var alreadyExistedFile = ((FileSystemException) e.getCause()).getFile();
                var errMsg = String.format(
                    "Error processing file at \"%s\": Couldn't write new file to \"%s\" because it already existed.",
                    docPath, alreadyExistedFile);
                LOG.error(errMsg, e);
                throw new MPFComponentDetectionError(
                        MPFDetectionError.MPF_FILE_WRITE_ERROR, errMsg, e);
            }
            var errMsg = "Error processing file at : " + docPath;
            LOG.error(errMsg, e);
            throw new MPFComponentDetectionError(MPFDetectionError.MPF_COULD_NOT_READ_MEDIA,
                    "Error processing file at : " + docPath, e);
        }
        return embeddedDocumentExtractor.getImageMap();
    }

    // Handles the case where the media is a generic type.
    @Override
    public List<MPFGenericTrack> getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        LOG.info("[{}] Starting job.", mpfGenericJob.getJobName());
        LOG.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
                mpfGenericJob.getJobName(), mpfGenericJob.getDataUri(),
                mpfGenericJob.getJobProperties().size(), mpfGenericJob.getMediaProperties().size());

        // =========================
        // Tika Detection
        // =========================

        String defaultSavePath = "$MPF_HOME/share/tmp/derivative-media";

        Map<String, String> properties = mpfGenericJob.getJobProperties();
        boolean separatePages = Boolean.parseBoolean(properties.get("ORGANIZE_BY_PAGE"));

        if (properties.get("SAVE_PATH") != null) {
            defaultSavePath = properties.get("SAVE_PATH");
        }
        defaultSavePath = MPFEnvironmentVariablePathExpander.expand(defaultSavePath);

        if (!mpfGenericJob.getJobName().isEmpty()) {
            String jobId = mpfGenericJob.getJobName().split(":")[0];
            jobId = Pattern.compile("job", Pattern.LITERAL | Pattern.CASE_INSENSITIVE).matcher(jobId).replaceAll("");
            jobId = jobId.trim();
            defaultSavePath += '/' + jobId;
        }

        var tracks = new ArrayList<MPFGenericTrack>();

        Metadata metadata = new Metadata();
        Map<Path, SortedSet<Integer>> imageMap =
                parseDocument(mpfGenericJob.getDataUri(), defaultSavePath, separatePages, metadata);
        if (imageMap.isEmpty()) {
            LOG.info("No images detected in document");
        }
        else {
            String contentType = metadata.get("Content-Type");
            boolean supportsPageNumbers = contentType.equals("application/pdf");

            for (Map.Entry<Path, SortedSet<Integer>> entry : imageMap.entrySet()) {
                var genericDetectionProperties = new HashMap<String, String>();
                genericDetectionProperties.put("DERIVATIVE_MEDIA_TEMP_PATH", entry.getKey().toString());

                if (supportsPageNumbers) {
                    var pageNumsStr = entry.getValue()
                            .stream()
                            .map(String::valueOf)
                            .collect(Collectors.joining("; "));
                    genericDetectionProperties.put("PAGE_NUM", pageNumsStr);
                } else {
                    genericDetectionProperties.put("PAGE_NUM", "-1");
                }

                MPFGenericTrack genericTrack = new MPFGenericTrack(-1, genericDetectionProperties);
                tracks.add(genericTrack);
            }
        }

        deleteEmptySubDirectories(new File(defaultSavePath + "/tika-extracted"));

        LOG.info("[{}] Processing complete. Generated {} generic tracks.",
                mpfGenericJob.getJobName(),
                tracks.size());

        return tracks;
    }

    // Recursively walks through the given directory and clears out empty subdirectories
    // If all subdirectories are empty and nothing else is contained, the given directory is also deleted.
    private static void deleteEmptySubDirectories(File path){
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
    @Override
    public boolean supports(MPFDataType mpfDataType) {
        return MPFDataType.UNKNOWN == mpfDataType;
    }

    @Override
    public String getDetectionType() {
        return "MEDIA";
    }

    @Override
    public List<MPFImageLocation> getDetections(MPFImageJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                "Image detection not supported.");
    }

    @Override
    public List<MPFVideoTrack> getDetections(MPFVideoJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                "Video detection not supported.");
    }

    @Override
    public List<MPFAudioTrack> getDetections(MPFAudioJob job) throws MPFComponentDetectionError {
        throw new MPFComponentDetectionError(MPFDetectionError.MPF_UNSUPPORTED_DATA_TYPE,
                "Audio detection not supported.");
    }
}
