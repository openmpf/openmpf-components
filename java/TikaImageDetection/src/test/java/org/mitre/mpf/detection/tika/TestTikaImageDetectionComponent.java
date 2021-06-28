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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mitre.mpf.component.api.detection.MPFComponentDetectionError;
import org.mitre.mpf.component.api.detection.MPFGenericJob;
import org.mitre.mpf.component.api.detection.MPFGenericTrack;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * The test class provides the framework for developing Java components.  Test cases can be prepared for a variety
 * of input conditions, and can cover successful executions as well as error conditions.  In most cases, if the
 * getDetections() and support() methods are correctly implemented, the component will work properly.  In cases where
 * the init() or close() methods are overridden, those also should be tested.
 */
public class TestTikaImageDetectionComponent {

    private TikaImageDetectionComponent tikaComponent;

    @Before
    public void setUp() {
        tikaComponent = new TikaImageDetectionComponent();
        tikaComponent.setConfigDirectory("plugin-files/config");
        tikaComponent.init();
    }

    @After
    public void tearDown() {
        tikaComponent.close();
    }


    private void markTempFiles(File directory){
        for (File tmpfile: directory.listFiles()) {
            tmpfile.deleteOnExit();
            if (tmpfile.isDirectory()) {
               markTempFiles(tmpfile);
            }
        }
    }

    private boolean pageCheck(String filepath) {
       String[] files = filepath.split(";");
       boolean filesFound = true;
       for (String file: files) {
           if (!Files.exists(Paths.get(file.trim()))){
               filesFound = false;
               break;
           }
       }
       return filesFound;
    }


    @Test
    public void testGetDetectionsGeneric() throws IOException, MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/TestImageExtraction.pdf").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        Path testDir = Files.createTempDirectory("tmp");
        testDir.toFile().deleteOnExit();

        jobProperties.put("SAVE_PATH", testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        jobProperties.put("ALLOW_EMPTY_PAGES", "true");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);
        boolean debug = false;

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        markTempFiles(testDir.toFile());

        // For human testing.
        if (debug) {
            for (int i = 0; i < tracks.size(); i++) {
                MPFGenericTrack track = tracks.get(i);
                System.out.println(String.format("Generic track number %d", i));
                System.out.println(String.format("  Confidence = %f", track.getConfidence()));
                System.out.println(String.format("  Image URI = %s", track.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));
                System.out.println(String.format("  Pages = %s", track.getDetectionProperties().get("PAGE_NUM")));

            }
        }

        assertEquals("Number of expected tracks does not match.", 3 ,tracks.size());
        // Test each output type.

        //Test extraction of image 0.
        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("Page number not correct", "1; 2; 3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image0.jpg.", testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image0.jpg"));
        assertTrue("Invalid file path for image0.jpg.", pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        testTrack = tracks.get(1);
        assertEquals("Page number not correct", "2; 3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image1.jpg.", testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image1.jpg"));
        assertTrue("Invalid file path for image1.jpg.", pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        testTrack = tracks.get(2);
        assertEquals("Page number not correct", "5", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image2.jpg.", testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image2.jpg"));
        assertTrue("Invalid file path for image2.jpg.", pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        Files.walk(Paths.get(testDir.toString()))
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
    }
}
