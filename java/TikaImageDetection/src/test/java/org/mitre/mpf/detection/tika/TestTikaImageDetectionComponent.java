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
import static org.junit.Assert.assertNotEquals;
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

        assertEquals(4 ,tracks.size());

        // Test extraction of image 0.
        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("1; 2; 3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image0.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        // Test extraction of image 1.
        testTrack = tracks.get(1);
        assertEquals("2; 3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image1.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        // Test extraction of image 2.
        testTrack = tracks.get(2);
        assertEquals("6", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image2.jpg"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        // Test extraction of image 3.
        testTrack = tracks.get(3);
        assertEquals("6", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").contains("image3.png"));
        assertTrue(pageCheck(testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI")));

        String uuid = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_URI").split("/")[5];

        // Check that images were saved correctly, then clean up test folder.
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid + "/image0.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid + "/image1.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid + "/image2.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid + "/image3.png")));

        FileUtils.deleteDirectory(testDir.toFile());
    }

    @Test
    public void testMultiMediaJob() throws IOException, MPFComponentDetectionError {
        String mediaPath1 = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        String mediaPath2 = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        Path testDir = Files.createTempDirectory("tmp");
        testDir.toFile().deleteOnExit();

        jobProperties.put("SAVE_PATH", testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        jobProperties.put("ALLOW_EMPTY_PAGES", "true");

        // Test that multiple documents submitted under one job ID are processed properly.
        MPFGenericJob genericSubJob1 = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath1, jobProperties, mediaProperties);
        MPFGenericJob genericSubJob2 = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath2, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks1 = tikaComponent.getDetections(genericSubJob1);
        List<MPFGenericTrack> tracks2 = tikaComponent.getDetections(genericSubJob2);

        String uuid1 = tracks1.get(0).getDetectionProperties().get("DERIVATIVE_MEDIA_URI").split("/")[5];
        String uuid2 = tracks2.get(0).getDetectionProperties().get("DERIVATIVE_MEDIA_URI").split("/")[5];

        assertNotEquals(uuid1, uuid2);

        // Check that images were saved correctly, then clean up test folder.
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid1 + "/image0.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid1 + "/image1.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid1 + "/image2.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid1 + "/image3.png")));

        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid2 + "/image0.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid2 + "/image1.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid2 + "/image2.jpg")));
        assertTrue(Files.exists(Paths.get(testDir + "/TestRun/tika-extracted/" + uuid2 + "/image3.png")));

        FileUtils.deleteDirectory(testDir.toFile());
    }
}
