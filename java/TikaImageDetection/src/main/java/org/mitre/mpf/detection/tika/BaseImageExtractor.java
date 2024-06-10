/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2024 The MITRE Corporation                                       *
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
import org.apache.tika.metadata.TikaCoreProperties;
import org.mitre.mpf.component.api.detection.MPFComponentDetectionError;
import org.mitre.mpf.component.api.detection.MPFDetectionError;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.ContentHandler;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public abstract class BaseImageExtractor implements EmbeddedDocumentExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(TikaImageDetectionComponent.class);

    private final Map<String, Path> _imgIdToAbsPath = new HashMap<>();

    private final Map<Path, SortedSet<Integer>> _imgPathToSrcPages = new LinkedHashMap<>();

    private int _id;


    protected abstract Path getOutputDir(int page) throws IOException;

    protected abstract Path processDuplicate(String imgId, Path existingPath, int page) throws IOException;


    @Override
    public boolean shouldParseEmbedded(Metadata metadata) {
        return true;
    }

    @Override
    public void parseEmbedded(InputStream stream, ContentHandler imHandler,
                              Metadata metadata, boolean outputHtml) throws IOException {
        String imgId = metadata.get(TikaCoreProperties.EMBEDDED_RELATIONSHIP_ID);
        if (imgId == null) {
            // Certain formats (i.e. odp) will not set a relationship ID for embedded files.
            // Assign a unique ID for each null ID.
            imgId = UUID.randomUUID().toString();
        }
        int page = Integer.parseInt(imHandler.toString());
        var existingPath = _imgIdToAbsPath.get(imgId);
        if (existingPath == null) {
            var pageOutputDir = getOutputDir(page);
            var imgPath = writeNewImage(pageOutputDir, stream, metadata);
            _imgIdToAbsPath.put(imgId, imgPath);
            _imgPathToSrcPages.put(imgPath, new TreeSet<>(List.of(page)));
            return;
        }

        var newPath = processDuplicate(imgId, existingPath, page);
        if (!newPath.equals(existingPath)) {
            _imgIdToAbsPath.put(imgId, newPath);
            updateImgPath(existingPath, newPath);
        }
        _imgPathToSrcPages.get(newPath).add(page);
    }

    private String getExtension(Metadata metadata) {
        String contentType = metadata.get(Metadata.CONTENT_TYPE);
        if (contentType != null) {
            int lastSlashPos = contentType.lastIndexOf('/'); // handle things like "image/png"
            if (lastSlashPos != -1) {
                return contentType.substring(lastSlashPos + 1);
            }
            return contentType;
        }
        String resourceName = metadata.get(TikaCoreProperties.RESOURCE_NAME_KEY);
        if (resourceName != null) {
            int lastDotPos = resourceName.lastIndexOf('.');
            if (lastDotPos != -1) {
                return resourceName.substring(lastDotPos + 1);
            }
        }
        return "";
    }

    private Path writeNewImage(Path dir, InputStream stream, Metadata metadata) throws IOException {
        String extension = getExtension(metadata);
        String fileName = extension.isBlank() ?
                String.format("image%s", _id) :
                String.format("image%s.%s", _id, extension);
        var outputPath = dir.resolve(fileName).toAbsolutePath();
        Files.copy(stream, outputPath);
        _id++;
        return outputPath;
    }

    private void updateImgPath(Path existingPath, Path newPath) {
        _imgPathToSrcPages.put(newPath, _imgPathToSrcPages.remove(existingPath));
    }


    protected static void createInitialDirectory(Path path) throws MPFComponentDetectionError {
        try {
            Files.createDirectories(path);
        }
        catch (IOException e) {
            var errorMsg = String.format(
                    "Unable to create the following directory %s",
                    path.toAbsolutePath());
            LOG.error(errorMsg, e);
            throw new MPFComponentDetectionError(
                    MPFDetectionError.MPF_FILE_WRITE_ERROR, errorMsg, e);
        }
    }

    public Map<Path, SortedSet<Integer>> getImageMap() {
        return _imgPathToSrcPages;
    }
}
