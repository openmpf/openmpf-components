/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2020 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2020 The MITRE Corporation                                       *
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
import java.nio.file.Paths;


import java.util.ArrayList;
import java.util.HashMap;



public class EmbeddedContentExtractor implements EmbeddedDocumentExtractor {
    private String path;
    private HashMap<String, String> imagesFound, imagesIndex;
    private ArrayList<String> commonImages;
    private boolean separatePages;
    private int id, pagenum;
    private ArrayList<ArrayList<String>> imageMap;
    private ArrayList<String> current;
    private Path outputDir;
    private Path commonImgDir;
    private Logger log;

    public EmbeddedContentExtractor(String savePath, boolean separate) {
        path = savePath;
        pagenum = -1;
        imagesFound = new HashMap<String, String>();
        imagesIndex = new HashMap<String, String>();
        commonImages = new ArrayList<String>();
        separatePages = separate;
        id = 0;
        imageMap = new ArrayList<ArrayList<String>>();
        current = new ArrayList<String>();

        outputDir = Paths.get(path + "/tika-extracted");
        commonImgDir = Paths.get(path + "/tika-extracted/common");

    }

    public String init(Logger mainLog) {
        log = mainLog;
        String errMsg = "";
        try {
            if (!Files.exists(outputDir)) {
                Files.createDirectories(outputDir);
            }
            if (separatePages && !Files.exists(commonImgDir)) {
                Files.createDirectories(commonImgDir);
            }
        } catch (IOException e) {
            errMsg = String.format("Unable to create the following directories: \n%s\n%s",
                    outputDir.toAbsolutePath(), commonImgDir.toAbsolutePath());
            log.error(errMsg, e);
        }
        return errMsg;

    }

    public ArrayList<String> getImageList() {
        ArrayList<String> results = new ArrayList<String>();
        for (ArrayList<String> pageResults: imageMap) {
            results.add(String.join("; ", pageResults));
        }
        return results;
    }

    public boolean shouldParseEmbedded(Metadata metadata) {
        return true;
    }

    public void updateFileLocation(String originalLocation, String newLocation) {
        for (ArrayList<String> pageList: imageMap) {
            if (pageList.indexOf(originalLocation) > -1) {
                pageList.set(pageList.indexOf(originalLocation), newLocation);
            }
        }
    }

    public void parseEmbedded(InputStream stream, ContentHandler imHandler, Metadata metadata, boolean outputHtml)
            throws IOException {
        //Create a new page
        int nextpage = Integer.parseInt(imHandler.toString()) - 1;
        while (pagenum < nextpage) {
            pagenum++;
            current = new ArrayList<String>();
            imageMap.add(current);
            if (separatePages) {
                outputDir = Paths.get(path + "/tika-extracted/page-" + String.valueOf(pagenum + 1));
                Files.createDirectories(outputDir);
            }
        }

        String cosID = metadata.get(Metadata.EMBEDDED_RELATIONSHIP_ID);
        String filename = "image" + String.valueOf(id) + "." + metadata.get(Metadata.CONTENT_TYPE);
        if (imagesFound.containsKey(cosID) ) {
            if (separatePages && !commonImages.contains(cosID)) {
                // For images already encountered, save into a common images folder.
                // Save each image only once in the common images folder.
                commonImages.add(cosID);
                String imageFile = imagesIndex.get(cosID);
                Path originalFile = Paths.get(imagesFound.get(cosID));
                Path outputPath = Paths.get(commonImgDir.toString() + "/" + imageFile);


                if (Files.exists(outputPath)) {
                    // Warn users that folder already contains images being overwritten
                    // This only happens if the same jobID was used on separate runs.
                    log.warn("File {} already exists. Can't write file.", outputPath.toAbsolutePath());
                    throw new IOException(String.format("File %s already exists. Can't write new file.", outputPath.toAbsolutePath()));
                }

                Files.move(originalFile, outputPath);
                updateFileLocation(originalFile.toString(), outputPath.toString());

                filename = outputPath.toAbsolutePath().toString();
                imagesFound.put(cosID, filename);
            } else {
                filename = imagesFound.get(cosID);
            }
        } else {
            imagesIndex.put(cosID, filename);
            Path outputPath = Paths.get(outputDir.toString() + "/" + filename);

            if (Files.exists(outputPath)) {
                // Warn users that folder already contains images being overwritten
                // This only happens if the same jobID was used on separate runs.
                log.warn("File {} already exists. Can't write new file.", outputPath.toAbsolutePath());
                throw new IOException(String.format("File %s already exists. Can't write file.", outputPath.toAbsolutePath()));
            }
            Files.copy(stream, outputPath);
            filename = outputPath.toAbsolutePath().toString();
            imagesFound.put(cosID, filename);
            id++;
        }
        current.add(filename);
    }
}
