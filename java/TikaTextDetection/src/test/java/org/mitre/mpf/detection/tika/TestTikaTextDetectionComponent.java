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
    public void testGetDetectionsPowerPointFile() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.pptx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);
        boolean debug = false;

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        // For human testing
        if (debug) {
            for (int i = 0; i < tracks.size(); i++) {
                MPFGenericTrack track = tracks.get(i);
                System.out.printf("  Generic track number %d%n", i);
                System.out.printf("  Confidence = %f%n", track.getConfidence());
                System.out.printf("  Text = %s%n", track.getDetectionProperties().get("TEXT"));
                System.out.printf("  Language = %s%n", track.getDetectionProperties().get("TEXT_LANGUAGE"));
                System.out.printf("  Page = %s%n", track.getDetectionProperties().get("PAGE_NUM"));
                System.out.printf("  Section = %s%n", track.getDetectionProperties().get("SECTION_NUM"));
                assertEquals("Confidence does not match.", -1.0f, track.getConfidence(), 0.1f);
            }
        }

        assertEquals(23 ,tracks.size());

        // Test language extraction
        assertSection(tracks.get(0), "01", "01", "English", "Testing Text Detection");
        assertSection(tracks.get(3), "02", "02", "Japanese", "ジアゼパム");

        // Test no detections
        assertTrue(tracks.get(9).getDetectionProperties().get("TEXT").isEmpty());
        assertEquals("Unknown", tracks.get(9).getDetectionProperties().get("TEXT_LANGUAGE"));

        assertSection(tracks.get(20), "10", "04", "English", "All human beings are born free");
        assertSection(tracks.get(22), "11", "02", "Unknown", "End slide test text"); // cannot determine language
    }

    @Test
    public void testGetDetectionsPdfFile() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.pdf").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "false");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals(5, tracks.size());

        assertSection(tracks.get(0), "1", "1", "Unknown", "OpenMPF"); // Not enough text.
        assertSection(tracks.get(1), "1", "2", "English", "Media Analytics");
        assertSection(tracks.get(2), "1", "3", "English", "web-friendly platform");
        assertSection(tracks.get(3), "1", "4", "English", "The MITRE Corporation");
        assertSection(tracks.get(4), "3", "1", "Unknown", "page 3"); // cannot determine language
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
        assertEquals(6, tracks.size());

        assertSection(tracks.get(0), "1", "1", "English", "first section of page 1");
        assertSection(tracks.get(1), "1", "2", "English", "second section of page 1");
        assertSection(tracks.get(2), "1", "3", "English", "third section of page 1");

        // Can't detect page numbers in documents, so page 1 is used for every track,
        // and the section numbers continue across actual pages.
        assertSection(tracks.get(3), "1", "4", "English", "first section of page 2");
        assertSection(tracks.get(4), "1", "5", "English", "second section of page 2");
        assertSection(tracks.get(5), "1", "6", "English", "third section of page 2");
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
        assertEquals(1, tracks.size());

        assertSection(tracks.get(0), "1", "1", "Unknown", "Test"); // cannot determine language
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"), containsString("1"));
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"), containsString("3"));
        assertThat(tracks.get(0).getDetectionProperties().get("TEXT"), containsString("6"));
    }

    private void assertSection(MPFGenericTrack track, String page, String section, String language, String text) {
        assertEquals(page, track.getDetectionProperties().get("PAGE_NUM"));
        assertEquals(section, track.getDetectionProperties().get("SECTION_NUM"));
        assertEquals(language, track.getDetectionProperties().get("TEXT_LANGUAGE"));
        assertThat(track.getDetectionProperties().get("TEXT"), containsString(text));
    }
}
