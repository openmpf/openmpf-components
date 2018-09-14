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

import org.xml.sax.ContentHandler;
import org.xml.sax.SAXException;
import org.xml.sax.SAXParseException;
import org.xml.sax.Attributes;

import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import java.lang.StringBuilder;

import java.util.ArrayList;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;



public class EmbeddedContentExtractor implements EmbeddedDocumentExtractor {
    private String path;
    private HashMap<String, String> images_found;
    private boolean separatePages;
    private int id,pagenum;
    private ArrayList<StringBuilder> image_map;
    private StringBuilder current;
    private Path outputPageDir;
    public EmbeddedContentExtractor(String save_path, boolean separate){
        path = save_path;
        pagenum = 0;
        images_found = new HashMap<String,String>();
        separatePages = separate;
        id = 0;
        image_map = new ArrayList<StringBuilder>();
        current = new StringBuilder();
        image_map.add(current);
        if(separatePages)
            outputPageDir = (new File(path + "/PAGE_" + String.valueOf(pagenum-1))).toPath();

        Path outputDir = (new File(path)).toPath();
        outputPageDir = outputDir;
        try {
            if (!Files.exists(outputDir))
                Files.createDirectories(outputDir);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public ArrayList<StringBuilder> getImageList() {
        return image_map;
    }

    public boolean shouldParseEmbedded(Metadata metadata) {
        return true;
    }

    public void parseEmbedded(InputStream stream, ContentHandler im_handler, Metadata metadata, boolean outputHtml)
            throws SAXException, IOException {
        //Create a new page
        int nextpage = Integer.parseInt(im_handler.toString()) - 1;
        while (pagenum < nextpage) {
            pagenum ++;
            current = new StringBuilder();
            image_map.add(current);
            if (separatePages) {
                outputPageDir = (new File(path + "/PAGE_" + String.valueOf(pagenum))).toPath();
            }
            Files.createDirectories(outputPageDir);
        }


        String cosID = metadata.get(Metadata.EMBEDDED_RELATIONSHIP_ID);
        if (images_found.containsKey(cosID)) {
            //Skip saving the file but append the file information.
        } else {
            images_found.put(cosID,"image" + String.valueOf(id) + "." + metadata.get(Metadata.CONTENT_TYPE));
            File outputFPath = new File(outputPageDir.toString() + "/image" + String.valueOf(id) + "." + metadata.get(Metadata.CONTENT_TYPE));
            Path outputPath = outputFPath.toPath();
            Files.deleteIfExists(outputPath);
            boolean check = Files.exists(outputPath);
            Files.copy(stream, outputPath);
            id++;
        }

        String filename = images_found.get(cosID);
        if (current.length() == 0) {
            current.append(filename);
        } else {
            current.append(", " + filename);
        }
    }
}
