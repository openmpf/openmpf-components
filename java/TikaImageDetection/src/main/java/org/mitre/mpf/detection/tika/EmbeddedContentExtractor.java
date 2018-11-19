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
import org.apache.tika.extractor.EmbeddedDocumentExtractor;
import org.apache.tika.metadata.Metadata;


import org.slf4j.Logger;
import org.xml.sax.ContentHandler;

import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import java.lang.StringBuilder;

import java.util.ArrayList;
import java.util.HashMap;



public class EmbeddedContentExtractor implements EmbeddedDocumentExtractor {
    private String path;
    private HashMap<String, String> imagesFound, imagesIndex;
    private ArrayList<String> commonImages;
    private boolean separatePages, repeatImages;
    private int id, pagenum;
    private ArrayList<StringBuilder> imageMap;
    private StringBuilder current;
    private Path outputDir;
    private Path commonImgDir;
    private Logger LOG;
    public EmbeddedContentExtractor(String savePath, boolean separate, boolean repeat) {
        path = savePath;
        pagenum = -1;
        imagesFound = new HashMap<String, String>();
        imagesIndex = new HashMap<String, String>();
        commonImages = new ArrayList<String>();
        separatePages = separate;
        repeatImages = repeat;
        id = 0;
        imageMap = new ArrayList<StringBuilder>();
        current = new StringBuilder();

        outputDir = (new File(path + "/tika-extracted")).toPath();
        commonImgDir =  (new File(path + "/tika-extracted/common")).toPath();

    }
    public String init(Logger log) {
        LOG = log;
        String errMsg = "";
        try {
            if (!Files.exists(outputDir)) {
                Files.createDirectories(outputDir);
            }
            if (repeatImages && !Files.exists(commonImgDir)) {
                Files.createDirectories(commonImgDir);
            }
        } catch (IOException e) {
            e.printStackTrace();
            errMsg = String.format("Unable to create the following directories: \n%s\n%s",
                    outputDir.toAbsolutePath(), commonImgDir.toAbsolutePath());
            LOG.error(errMsg, e);
        }
        return errMsg;

    }

    public ArrayList<StringBuilder> getImageList() {
        return imageMap;
    }

    public boolean shouldParseEmbedded(Metadata metadata) {
        return true;
    }

    public void parseEmbedded(InputStream stream, ContentHandler imHandler, Metadata metadata, boolean outputHtml)
            throws IOException {
        //Create a new page
        int nextpage = Integer.parseInt(imHandler.toString()) - 1;
        while (pagenum < nextpage) {
            pagenum++;
            current = new StringBuilder();
            imageMap.add(current);
            if (separatePages) {
                outputDir = (new File(path + "/page-" + String.valueOf(pagenum))).toPath();
            }
            Files.createDirectories(outputDir);
        }


        String cosID = metadata.get(Metadata.EMBEDDED_RELATIONSHIP_ID);
        String filename = "image" + String.valueOf(id) + "." + metadata.get(Metadata.CONTENT_TYPE);
        if (imagesFound.containsKey(cosID) ) {
            if (repeatImages && !commonImages.contains(cosID)) {
                // For images already encountered, save into a common images folder.
                // Save each image only once in the common images folder.
                commonImages.add(cosID);
                String imageFile = imagesIndex.get(cosID);
                File outputFPath = new File(commonImgDir.toString() + "/" + imageFile);
                Path outputPath = outputFPath.toPath();



                if (Files.exists(outputPath)) {
                    // Warn users that folder already contains images being overwritten
                    // This only happens if the same jobID was used on separate runs.
                    LOG.warn("File {} already exists and will be overwritten", outputPath.toAbsolutePath());
                    Files.deleteIfExists(outputPath);
                }
                Files.copy(stream, outputPath);

                filename = outputFPath.getAbsolutePath();
                imagesFound.put(cosID, filename);
            } else {
                filename = imagesFound.get(cosID);
            }
        } else {
            imagesIndex.put(cosID, filename);
            File outputFPath = new File(outputDir.toString() + "/" + filename);
            Path outputPath = outputFPath.toPath();

            if (Files.exists(outputPath)) {
                // Warn users that folder already contains images being overwritten
                // This only happens if the same jobID was used on separate runs.
                LOG.warn("File {} already exists and will be overwritten", outputPath.toAbsolutePath());
                Files.deleteIfExists(outputPath);
            }
            Files.copy(stream, outputPath);
            filename = outputFPath.getAbsolutePath();
            imagesFound.put(cosID, filename);
            id++;
        }

        if (current.length() == 0) {
            current.append(filename);
        } else {
            current.append(";" + filename);
        }

    }
}
