/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2019 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2019 The MITRE Corporation                                       *
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
                System.out.println(String.format("  Page = %s", track.getDetectionProperties().get("PAGE_NUM")));
                System.out.println(String.format("  Images = %s", track.getDetectionProperties().get("SAVED_IMAGES")));
            }
        }

        assertEquals("Number of expected tracks does not match.", 5 ,tracks.size());
        // Test each output type.

        //Test extraction of image 0, page 0.
        //Image 0 stored multiple times, should only be reported once.
        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("Page number not correct", "1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image0.jpg in page 1.", testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue("Invalid file paths for page 1.", pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        //Test extraction of image 1, page 1.
        //Image 0 should be listed again.
        testTrack = tracks.get(1);
        assertEquals("Page number not correct", "2", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image0.jpg in page 2.", testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue("Expected image1.jpg in page 2.", testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image1.jpg"));
        assertTrue("Invalid file paths for page 2.", pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        //Test extraction of page 2.
        //Images 0 and 1 should be listed again.
        testTrack = tracks.get(2);
        assertEquals("Page number not correct", "3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image0.jpg in page 3.", testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue("Expected image1.jpg in page 3.", testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image1.jpg"));
        assertTrue("Invalid file paths for page 3.", pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));


        //Test empty page.
        testTrack = tracks.get(3);
        assertEquals("Page number not correct", "4", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertEquals("No images should have been reported.", "", testTrack.getDetectionProperties().get("SAVED_IMAGES"));

        //Test extraction of image 2, page 5.
        testTrack = tracks.get(4);
        assertEquals("Page number not correct", "5", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue("Expected image2.jpg in page 5.", testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image2.jpg"));
        assertTrue("Invalid file paths for page 5.", pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        //Check that images were saved correctly, then clean up test folder.
        assertTrue("Test run directory not created." + testDir.toString() + "./TestRun", Files.exists(Paths.get(testDir.toString() + "/TestRun")));
        assertTrue("Expected image not found (tika-extracted/image0.jpg).", Files.exists(Paths.get(testDir.toString() + "/TestRun/tika-extracted/image0.jpg")));
        assertTrue("Expected image not found (tika-extracted/image1.jpg).", Files.exists((Paths.get(testDir.toString() + "/TestRun/tika-extracted/image1.jpg"))));
        assertTrue("Expected image not found (tika-extracted/image2.jpg).", Files.exists((Paths.get(testDir.toString() + "/TestRun/tika-extracted/image2.jpg"))));

        Files.walk(Paths.get(testDir.toString()))
                .sorted(Comparator.reverseOrder())
                .map(Path::toFile)
                .forEach(File::delete);
    }
}
