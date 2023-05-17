/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2023 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2023 The MITRE Corporation                                       *
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

import org.mitre.mpf.component.api.detection.MPFComponentDetectionError;

import java.nio.file.Path;
import java.util.UUID;

public class SameDirImageExtractor extends BaseImageExtractor {

    private final Path _outputDir;

    public SameDirImageExtractor(String savePath) throws MPFComponentDetectionError {
        var uniqueId = UUID.randomUUID().toString();
        _outputDir = Path.of(savePath, "tika-extracted", uniqueId);
        createInitialDirectory(_outputDir);
    }

    @Override
    protected Path getOutputDir(int page) {
        return _outputDir;
    }

    @Override
    protected Path processDuplicate(String cosId, Path existingPath, int page) {
        return existingPath;
    }
}
