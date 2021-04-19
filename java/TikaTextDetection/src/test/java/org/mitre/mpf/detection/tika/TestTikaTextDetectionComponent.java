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

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.hamcrest.CoreMatchers.containsString;
import static org.junit.Assert.*;

/**
 * The test class provides the framework for developing Java components.  Test cases can be prepared for a variety
 * of input conditions, and can cover successful executions as well as error conditions.  In most cases, if the
 * getDetections() and support() methods are correctly implemented, the component will work properly.  In cases where
 * the init() or close() methods are overridden, those also should be tested.
 */
public class TestTikaTextDetectionComponent {

    private TikaTextDetectionComponent tikaComponent;

    @Before
    public void setUp() {
        tikaComponent = new TikaTextDetectionComponent();
        tikaComponent.init();
        tikaComponent.setRunDirectory("..");
    }

    @After
    public void tearDown() {
        tikaComponent.close();
    }

    @Test
    public void testGetDetectionsGeneric() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.pptx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);
        boolean debug = false;

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals("Number of expected tracks does not match.", 23 ,tracks.size());
        // Test each output type.

        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("Expected language does not match.", "English", testTrack.getDetectionProperties().get("TEXT_LANGUAGE"));
        assertEquals("Expected text does not match.", "Testing Text Detection", testTrack.getDetectionProperties().get("TEXT"));

        // Test language extraction.
        testTrack = tracks.get(3);
        assertEquals("Expected language does not match.", "Japanese", testTrack.getDetectionProperties().get("TEXT_LANGUAGE"));

        // Test no detections.
        testTrack = tracks.get(9);
        assertEquals("Text should be empty", "", testTrack.getDetectionProperties().get("TEXT"));
        assertEquals("Language should be empty", "Unknown", testTrack.getDetectionProperties().get("TEXT_LANGUAGE"));

        testTrack = tracks.get(20);
        assertThat(testTrack.getDetectionProperties().get("TEXT"), containsString("All human beings are born free"));

        testTrack = tracks.get(22);
        assertThat(testTrack.getDetectionProperties().get("TEXT"), containsString("End slide test text"));

        // For human testing.
        if (debug) {
            for (int i = 0; i < tracks.size(); i++) {
                MPFGenericTrack track = tracks.get(i);
                System.out.println(String.format("  Generic track number %d", i));
                System.out.println(String.format("  Confidence = %f", track.getConfidence()));
                System.out.println(String.format("  Text = %s", track.getDetectionProperties().get("TEXT")));
                System.out.println(String.format("  Language = %s", track.getDetectionProperties().get("TEXT_LANGUAGE")));
                assertEquals("Confidence does not match.", -1.0f, track.getConfidence(), 0.1f);
            }
        }
    }

    @Test
    public void testGetDetectionsPowerPointFile() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.pptx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals("Number of expected tracks does not match.", 23, tracks.size());
    }

    @Test
    public void testGetDetectionsDocumentFile() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.docx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals("Number of expected tracks does not match.", 6, tracks.size());
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"),
                containsString("first section"));
        assertThat(tracks.get(1).getDetectionProperties().get("TEXT"),
                containsString("second section"));
        assertThat(tracks.get(2).getDetectionProperties().get("TEXT"),
                containsString("third section"));
    }

    @Test
    public void testGetDetectionsExcelFile() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.xlsx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals("Number of expected tracks does not match.", 1, tracks.size());
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"),
                containsString("Test"));
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"),
                containsString("1"));
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"),
                containsString("3"));
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"),
                containsString("6"));
    }
}
