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

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mitre.mpf.component.api.detection.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import static org.hamcrest.CoreMatchers.containsString;

import java.nio.file.Files;
import java.io.File;
import java.nio.file.Path;

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
        tikaComponent.init();
        tikaComponent.setRunDirectory("..");
    }

    @After
    public void tearDown() {
        tikaComponent.close();
    }


    @Test
    public void testGetDetectionsGeneric() throws Exception {
        String uri = "./test/data/TestImageExtraction.pdf";
        Map<String, String> jobProperties = new HashMap<String, String>();
        Map<String, String> mediaProperties = new HashMap<String, String>();


        //String textTags = "text-tags.json";
        //String rundirectory = tikaComponent.getRunDirectory();
        jobProperties.put("SAVE_PATH", "./test/data/");
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        MPFGenericJob genericJob = new MPFGenericJob("TestRun:TestGenericJob", uri, jobProperties, mediaProperties);
        boolean debug = false;

        try {
            List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

            assertEquals("Number of expected tracks does not match.", 4 ,tracks.size());
            // Test each output type.

            //Test extraction of image 0, page 0
            //image 0 stored multiple times, should only be reported once.
            MPFGenericTrack testTrack = tracks.get(0);
            assertEquals("Page number not correct", "0", testTrack.getDetectionProperties().get("PAGE"));
            assertEquals("Expected images not listed.", "image0.jpg", testTrack.getDetectionProperties().get("IMAGE_FILES"));

            //Test extraction of image 1, page 1
            //image 0 should be listed again.
            testTrack = tracks.get(1);
            assertEquals("Page number not correct", "1", testTrack.getDetectionProperties().get("PAGE"));
            assertEquals("Expected images not listed.", "image0.jpg, image1.jpg", testTrack.getDetectionProperties().get("IMAGE_FILES"));

            //Test empty page
            testTrack = tracks.get(2);
            assertEquals("Page number not correct", "2", testTrack.getDetectionProperties().get("PAGE"));
            assertEquals("No images should have been reported.", null, testTrack.getDetectionProperties().get("IMAGE_FILES"));

            //Test extraction of image 2, page 2
            testTrack = tracks.get(3);
            assertEquals("Page number not correct", "3", testTrack.getDetectionProperties().get("PAGE"));
            assertEquals("Expected images not listed.", "image2.jpg", testTrack.getDetectionProperties().get("IMAGE_FILES"));

            //Check that images were saved correctly, then clean up test folder.
            assertTrue(Files.exists((new File("./test/data/TestRun")).toPath()));
            assertTrue(Files.exists((new File("./test/data/TestRun/image0.jpg")).toPath()));
            assertTrue(Files.exists((new File("./test/data/TestRun/image1.jpg")).toPath()));
            assertTrue(Files.exists((new File("./test/data/TestRun/image2.jpg")).toPath()));
            Files.deleteIfExists((new File("./test/data/TestRun/image0.jpg")).toPath());
            Files.deleteIfExists((new File("./test/data/TestRun/image1.jpg")).toPath());
            Files.deleteIfExists((new File("./test/data/TestRun/image2.jpg")).toPath());
            Files.deleteIfExists((new File("./test/data/TestRun")).toPath());

            // For human testing.
            if (debug) {
                for (int i = 0; i < tracks.size(); i++) {
                    MPFGenericTrack track = tracks.get(i);
                    System.out.println(String.format("Generic track number %d", i));
                    System.out.println(String.format("  Confidence = %f", track.getConfidence()));
                    System.out.println(String.format("  Page = %s", track.getDetectionProperties().get("PAGE")));
                    System.out.println(String.format("  Images = %s", track.getDetectionProperties().get("IMAGE_FILES")));
                }
            }
        } catch (MPFComponentDetectionError e) {
            System.err.println(String.format("An error occurred of type ", e.getDetectionError().name()));
            fail(String.format("An error occurred of type ", e.getDetectionError().name()));
        }
    }
}
