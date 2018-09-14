/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2018 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2018 The MITRE Corporation                                       *
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

import org.mitre.mpf.component.api.detection.*;
import org.mitre.mpf.component.api.detection.util.MPFEnvironmentVariablePathExpander;

import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.LinkedList;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

import org.apache.tika.metadata.Metadata;

import java.lang.StringBuilder;

import org.apache.tika.parser.AutoDetectParser;
import org.apache.tika.parser.ParseContext;
import org.apache.tika.parser.Parser;
import org.apache.tika.config.TikaConfig;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Iterator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.StringWriter;
import java.io.PrintWriter;


import org.apache.tika.parser.pdf.PDFParserConfig;
import org.xml.sax.ContentHandler;

import java.io.InputStream;
import org.apache.tika.extractor.EmbeddedDocumentExtractor;
import org.apache.tika.exception.TikaException;
import org.xml.sax.SAXException;

public class TikaImageDetectionComponent extends MPFDetectionComponentBase {

    private static final Logger LOG = LoggerFactory.getLogger(TikaImageDetectionComponent.class);

    private static void setPdfConfig(ParseContext context) {
            PDFParserConfig pdfConfig = new PDFParserConfig();
            pdfConfig.setExtractInlineImages(true);
            pdfConfig.setExtractUniqueInlineImagesOnly(false);
            context.set(PDFParserConfig.class, pdfConfig);
    }

    private static ArrayList<StringBuilder> parseDocument(final String input, final String path, final boolean separatePages) {
        String xhtmlContents = "";
        TikaConfig config = TikaConfig.getDefaultConfig() ;
        try {
            config = new TikaConfig("plugin-files/config/tika-config.xml");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (TikaException e) {
            e.printStackTrace();
        }

        AutoDetectParser parser = new AutoDetectParser(config);
        ContentHandler handler = new ImageExtractionContentHandler();
        Metadata metadata = new Metadata();
        ParseContext context = new ParseContext();
        EmbeddedDocumentExtractor embeddedDocumentExtractor = new EmbeddedContentExtractor(path, separatePages);

        context.set(EmbeddedDocumentExtractor.class, embeddedDocumentExtractor);
        context.set(AutoDetectParser.class, parser);
        setPdfConfig(context);
        try {
            InputStream stream = new FileInputStream(input);
            parser.parse(stream, handler, metadata, context);
            xhtmlContents = handler.toString();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        } catch (TikaException e) {
            e.printStackTrace();
        }
        EmbeddedContentExtractor x = ((EmbeddedContentExtractor) embeddedDocumentExtractor);
        return x.getImageList();
    }

    // Handles the case where the media is a generic type.
    public List<MPFGenericTrack>  getDetections(MPFGenericJob mpfGenericJob) throws MPFComponentDetectionError {
        LOG.debug("jobName = {}, dataUri = {}, size of jobProperties = {}, size of mediaProperties = {}",
            mpfGenericJob.getJobName(), mpfGenericJob.getDataUri(),
            mpfGenericJob.getJobProperties().size(), mpfGenericJob.getMediaProperties().size());

        // =========================
        // Tika Detection
        // =========================

        String default_save_path = mpfGenericJob.getDataUri();
        Map<String,String> properties = mpfGenericJob.getJobProperties();
        boolean separatePages = false;

        if (properties.get("SAVE_PATH") != null) {
            default_save_path = properties.get("SAVE_PATH");
            default_save_path = MPFEnvironmentVariablePathExpander.expand(default_save_path);
        }

        if (properties.get("ORGANIZE_BY_PAGE") != null) {
            separatePages = Boolean.valueOf(properties.get("ORGANIZE_BY_PAGE"));
        }

        if (mpfGenericJob.getJobName().length() != 0) {
            default_save_path += "/"+mpfGenericJob.getJobName().split(":")[0];
        }

        String text_output = "";
        String metadata_output = "";
        String context_output = "";
        List<StringBuilder> page_output = new ArrayList<StringBuilder>();
        List<MPFGenericTrack> tracks = new LinkedList<MPFGenericTrack>();
        Integer page = 0;
        float confidence = -1.0f;
        for (StringBuilder im_list: parseDocument(mpfGenericJob.getDataUri(), default_save_path, separatePages)) {
            Map<String, String> genericDetectionProperties = new HashMap<String, String>();
            genericDetectionProperties.put("PAGE",page.toString());
            if (im_list.toString().length() > 0) {
                genericDetectionProperties.put("IMAGE_FILES",im_list.toString());
            }
            MPFGenericTrack genericTrack = new MPFGenericTrack(confidence, genericDetectionProperties);
            tracks.add(genericTrack);
            page++;
        }

        LOG.debug(
                String.format("[%s] Processing complete. Generated %d generic tracks.",
                        mpfGenericJob.getJobName(),
                        tracks.size()));

        return tracks;
    }

    // Extract stack trace as a string.
    public static String extractStackTrace(Exception e) {
        StringWriter sw = new StringWriter();
        PrintWriter pw = new PrintWriter(sw);
        e.printStackTrace(pw);
        return sw.toString();
    }

    // The TikeDetection component supports generic file types (pdfs, documents, txt, etc.).
    public boolean supports(MPFDataType mpfDataType) {
        return mpfDataType != null && MPFDataType.UNKNOWN.equals(mpfDataType);
    }

    public String getDetectionType() {
        return "IMAGE";
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
