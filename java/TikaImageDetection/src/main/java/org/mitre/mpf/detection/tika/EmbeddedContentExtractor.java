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

import org.apache.tika.extractor.EmbeddedDocumentExtractor;
import org.apache.tika.metadata.Metadata;
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


public class EmbeddedContentExtractor implements EmbeddedDocumentExtractor {

    private static final Logger LOG = LoggerFactory.getLogger(TikaImageDetectionComponent.class);

    private final Map<String, Path> _imgIdToAbsPath = new HashMap<>();

    private final Set<String> _commonImages = new HashSet<>();

    private final boolean _separatePages;

    private int _id;

    private int _pageNum = -1;

    // Maps the path of an extracted image to the page numbers it appeared on.
    private final Map<Path, SortedSet<Integer>> _imgPathToSrcPages = new LinkedHashMap<>();

    private final Path _baseOutputDir;
    private final Path _commonImgDir;

    public EmbeddedContentExtractor(String savePath, boolean separate) throws MPFComponentDetectionError {
        _separatePages = separate;

        var uniqueId = UUID.randomUUID().toString();
        _baseOutputDir = Path.of(savePath, "tika-extracted", uniqueId);
        _commonImgDir = _baseOutputDir.resolve("common");

        try {
            Files.createDirectories(_baseOutputDir);
            if (_separatePages) {
                Files.createDirectories(_commonImgDir);
            }
        }
        catch (IOException e) {
            var errorMsg = String.format(
                    "Unable to create the following directories: \n%s\n%s",
                    _baseOutputDir.toAbsolutePath(), _commonImgDir.toAbsolutePath());
            LOG.error(errorMsg, e);
            throw new MPFComponentDetectionError(
                    MPFDetectionError.MPF_FILE_WRITE_ERROR, errorMsg, e);
        }
    }


    public Map<Path, SortedSet<Integer>> getImageMap() {
        return _imgPathToSrcPages;
    }

    @Override
    public boolean shouldParseEmbedded(Metadata metadata) {
        return true;
    }

    private void updateFileLocation(Path originalLocation, Path newLocation) {
        var pageMap = _imgPathToSrcPages.remove(originalLocation);
        _imgPathToSrcPages.put(newLocation, pageMap);
    }

    @Override
    public void parseEmbedded(InputStream stream, ContentHandler imHandler,
                              Metadata metadata, boolean outputHtml) throws IOException {
        //Create a new page
        int nextPage = Integer.parseInt(imHandler.toString()) - 1;

        Path localOutputDir;
        if (_separatePages) {
            localOutputDir = _baseOutputDir.resolve("page-" + (_pageNum + 1));
            while (_pageNum < nextPage) {
                _pageNum++;
                localOutputDir = _baseOutputDir.resolve("page-" + (_pageNum + 1));
                Files.createDirectories(localOutputDir);
            }
        }
        else {
            _pageNum = nextPage;
            localOutputDir = _baseOutputDir;
        }


        String cosId = metadata.get(Metadata.EMBEDDED_RELATIONSHIP_ID);
        Path imgAbsPath;
        if (_imgIdToAbsPath.containsKey(cosId) ) {
            if (_separatePages && !_commonImages.contains(cosId)) {
                // For images already encountered, save into a common images folder.
                // Save each image only once in the common images folder.
                _commonImages.add(cosId);
                Path originalPath = _imgIdToAbsPath.get(cosId);
                Path outputPath = _commonImgDir.resolve(originalPath.getFileName());

                if (Files.exists(outputPath)) {
                    // Warn users that folder already contains images being overwritten
                    // This only happens if the same jobID was used on separate runs.
                    LOG.warn("File {} already exists. Can't write file.", outputPath.toAbsolutePath());
                    throw new IOException(String.format("File %s already exists. Can't write new file.",
                            outputPath.toAbsolutePath()));
                }

                Files.move(originalPath, outputPath);
                updateFileLocation(originalPath, outputPath);

                imgAbsPath = outputPath.toAbsolutePath();
                _imgIdToAbsPath.put(cosId, imgAbsPath);
            } else {
                imgAbsPath = _imgIdToAbsPath.get(cosId);
            }
        } else {
            String filename = "image" + _id + '.' + metadata.get(Metadata.CONTENT_TYPE);
            Path outputPath = localOutputDir.resolve(filename);

            if (Files.exists(outputPath)) {
                // Warn users that folder already contains images being overwritten
                // This only happens if the same jobID was used on separate runs.
                LOG.warn("File {} already exists. Can't write new file.", outputPath.toAbsolutePath());
                throw new IOException(String.format("File %s already exists. Can't write file.",
                        outputPath.toAbsolutePath()));
            }
            Files.copy(stream, outputPath);
            imgAbsPath = outputPath.toAbsolutePath();
            _imgIdToAbsPath.put(cosId, imgAbsPath);
            _id++;
        }

        _imgPathToSrcPages.computeIfAbsent(imgAbsPath, k -> new TreeSet<>()).add(_pageNum + 1);
    }
}
