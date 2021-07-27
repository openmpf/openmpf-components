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

import org.apache.commons.io.FileUtils;
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

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
        String mediaPath = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
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
                System.out.printf("Generic track number %d%n", i);
                System.out.printf("  Confidence = %f%n", track.getConfidence());
                System.out.printf("  Page = %s%n", track.getDetectionProperties().get("PAGE_NUM"));
                System.out.printf("  Images = %s%n", track.getDetectionProperties().get("SAVED_IMAGES"));
            }
        }

        assertEquals(5 ,tracks.size());

        // Test extraction of image 0, page 1.
        // Image 0 stored multiple times, should only be reported once.
        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        // Test extraction of image 1, page 2.
        // Image 0 should be listed again.
        testTrack = tracks.get(1);
        assertEquals("2", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image1.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        // Test extraction of page 3.
        // Images 0 and 1 should be listed again.
        testTrack = tracks.get(2);
        assertEquals("3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image1.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        // Test empty page.
        testTrack = tracks.get(3);
        assertEquals("4", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertEquals("", testTrack.getDetectionProperties().get("SAVED_IMAGES"));

        // Test extraction of image 2, page 5.
        testTrack = tracks.get(4);
        assertEquals("5", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image2.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        // Check that images were saved correctly, then clean up test folder.
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/image0.jpg")));
        assertTrue(Files.exists((Paths.get(testDir + "/TestRun/tika-extracted/image1.jpg"))));
        assertTrue(Files.exists((Paths.get(testDir + "/TestRun/tika-extracted/image2.jpg"))));

        FileUtils.deleteDirectory(testDir.toFile());
    }

    @Test
    public void testGetDetectionsFirstPageImagesOnly() throws IOException, MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-first-page-images.pdf").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        Path testDir = Files.createTempDirectory("tmp");
        testDir.toFile().deleteOnExit();

        jobProperties.put("SAVE_PATH", testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        jobProperties.put("ALLOW_EMPTY_PAGES", "false");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        markTempFiles(testDir.toFile());

        assertEquals("Number of expected tracks does not match.", 1 ,tracks.size());

        // Test extraction of images 0-2, page 1.
        // Three images should be stored in one track as the first page output. No other tracks should exist.
        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image0.jpg"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image1.png"));
        assertTrue(testTrack.getDetectionProperties().get("SAVED_IMAGES").contains("image2.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("SAVED_IMAGES")));

        FileUtils.deleteDirectory(testDir.toFile());
    }
}
