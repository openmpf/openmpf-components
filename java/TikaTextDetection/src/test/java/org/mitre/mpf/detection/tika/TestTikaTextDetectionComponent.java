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
    public void testGetDetectionsTxt() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.txt").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "false");
        jobProperties.put("STORE_METADATA", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals(2, tracks.size());

        assertThat(tracks.get(0).getDetectionProperties().get("METADATA"),
                containsString("{\"X-TIKA:Parsed-By\":\"org.apache.tika.parser.DefaultParser\"," +
                        "\"X-TIKA:Parsed-By-Full-Set\":\"org.apache.tika.parser.DefaultParser\"," +
                        "\"Content-Encoding\":\"ISO-8859-1\",\"Content-Type\":\"text/plain; charset=ISO-8859-1\"}"));

        assertSection(tracks.get(1), "-1", "1", "English", "Testing, this is the first section");
    }

    @Test
    public void testGetDetectionsPdf() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.pdf").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "false");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals(30, tracks.size());

        assertSection(tracks.get(0), "1", "1", "English", "OpenMPF");
        assertSection(tracks.get(0), "1", "1", "English", "Bridging the Gap in Media Analytics");
        assertSection(tracks.get(2), "1", "3", "Unknown", "Data Filtering");
        assertSection(tracks.get(23), "1", "24", "English", "The MITRE Corporation");
        assertSection(tracks.get(29), "3", "1", "Unknown", "page 3"); // cannot determine language
    }

    @Test
    public void testGetDetectionsPptx() throws MPFComponentDetectionError {
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
        assertSection(tracks.get(0), "1", "1", "English", "Testing Text Detection");
        assertSection(tracks.get(3), "2", "2", "Japanese", "ジアゼパム");

        // Test no detections
        assertTrue(tracks.get(9).getDetectionProperties().get("TEXT").isEmpty());
        assertEquals("Unknown", tracks.get(9).getDetectionProperties().get("TEXT_LANGUAGE"));

        assertSection(tracks.get(20), "10", "4", "English", "All human beings are born free");
        assertSection(tracks.get(22), "11", "2", "Unknown", "End slide test text"); // cannot determine language
    }

    @Test
    public void testMergeTextPptx() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.pptx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");
        jobProperties.put("MERGE_TEXT", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);
        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        assertEquals(1 ,tracks.size());

        assertSection(tracks.get(0), "1", "1", "English", "Testing Text Detection");
        assertSection(tracks.get(0), "1", "1", "English", "ジアゼパム");
        assertSection(tracks.get(0), "1", "1", "English", "All human beings are born free");
        assertSection(tracks.get(0), "1", "1", "English", "End slide test text");
    }

    @Test
    public void testGetDetectionsOdp() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.odp").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        // TODO: Is it possible to get page (slide) number information from .odp files?
        // NOTE: Text is broken up into more sections than tracks generated from test-tika-detection.pptx.
        // NOTE: A 23rd track is generated in a dev. env. but not in the Docker image.
        assertTrue(tracks.size() == 22 || tracks.size() == 23);

        // Test language extraction
        assertSection(tracks.get(0), "-1", "1", "English", "Testing Text Detection");
        assertSection(tracks.get(3), "-1", "4", "Japanese", "ジアゼパム");

        // TODO: Look into why, unlike tracks generated from test-tika-detection.pptx, there is no track for
        //  blank slide 5.

        assertSection(tracks.get(19), "-1", "20", "English", "All human beings are born free");
        assertSection(tracks.get(20), "-1", "21", "Unknown", "End"); // cannot determine language
        assertSection(tracks.get(21), "-1", "22", "Unknown", "End slide test text"); // cannot determine language

        if (tracks.size() == 23) {
            // TODO: Look into why last section matches first section although the text is not on the last slide.
            assertSection(tracks.get(22), "-1", "23", "English", "Testing Text Detection");
        }
    }

    @Test
    public void testGetDetectionsDocx() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.docx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        // TODO: Is it possible to get page number information from .docx files?
        assertEquals(6, tracks.size());

        // page 1
        assertSection(tracks.get(0), "-1", "1", "English", "first section of page 1");
        assertSection(tracks.get(1), "-1", "2", "English", "second section of page 1");
        assertSection(tracks.get(2), "-1", "3", "English", "third section of page 1");

        // page 2
        assertSection(tracks.get(3), "-1", "4", "English", "first section of page 2");
        assertSection(tracks.get(4), "-1", "5", "English", "second section of page 2");
        assertSection(tracks.get(5), "-1", "6", "English", "third section of page 2");
    }

    @Test
    public void testGetDetectionsOdt() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.odt").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        // TODO: Is it possible to get page number information from .odt files?
        assertEquals(6, tracks.size());

        // page 1
        assertSection(tracks.get(0), "-1", "1", "English", "first section of page 1");
        assertSection(tracks.get(1), "-1", "2", "English", "second section of page 1");
        assertSection(tracks.get(2), "-1", "3", "English", "third section of page 1");

        // page 2
        assertSection(tracks.get(3), "-1", "4", "English", "first section of page 2");
        assertSection(tracks.get(4), "-1", "5", "English", "second section of page 2");
        assertSection(tracks.get(5), "-1", "6", "English", "third section of page 2");
    }

    @Test
    public void testGetDetectionsXlsx() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.xlsx").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        // TODO: Is it possible to get page (sheet) number information from .xlsx files?
        // TODO: Is it possible to split up text into different pages / sections based on the sheet it appears in?
        // TODO: Look into why tracks 1 and 2 contain font info. Oddly, font info is not reported for a
        //  single-page .xlsx file.
        assertEquals(3, tracks.size());

        MPFGenericTrack testTrack = tracks.get(0);
        String testTrackText = testTrack.getDetectionProperties().get("TEXT");

        // sheet 1
        assertSection(testTrack, "-1", "1", "Unknown", "Test"); // cannot determine language
        assertThat(testTrackText, containsString("1"));
        assertThat(testTrackText, containsString("2"));
        assertThat(testTrackText, containsString("3"));
        assertThat(testTrackText, containsString("4"));
        assertThat(testTrackText, containsString("5"));
        assertThat(testTrackText, containsString("6"));

        // sheet 2
        assertThat(testTrackText, containsString("Test 2"));
        assertThat(testTrackText, containsString("7"));
        assertThat(testTrackText, containsString("8"));
        assertThat(testTrackText, containsString("9"));
        assertThat(testTrackText, containsString("10"));
        assertThat(testTrackText, containsString("11"));
        assertThat(testTrackText, containsString("12"));
    }

    @Test
    public void testGetDetectionsOds() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-detection.ods").getPath();

        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);

        // TODO: Is it possible to get page (sheet) number information from .ods files?
        // TODO: Look into why each cell is reported in a different track for .ods files.
        //  Can we make the .pptx and .ods file behavior the same?
        assertEquals(19, tracks.size());

        // sheet 1
        assertSection(tracks.get(5), "-1", "6", "Unknown", "Test"); // cannot determine language
        assertThat(tracks.get(6).getDetectionProperties().get("TEXT"), containsString("1"));
        assertThat(tracks.get(7).getDetectionProperties().get("TEXT"), containsString("2"));
        assertThat(tracks.get(8).getDetectionProperties().get("TEXT"), containsString("3"));
        assertThat(tracks.get(9).getDetectionProperties().get("TEXT"), containsString("4"));
        assertThat(tracks.get(10).getDetectionProperties().get("TEXT"), containsString("5"));
        assertThat(tracks.get(11).getDetectionProperties().get("TEXT"), containsString("6"));

        // sheet 2
        assertThat(tracks.get(12).getDetectionProperties().get("TEXT"), containsString("Test 2"));
        assertThat(tracks.get(13).getDetectionProperties().get("TEXT"), containsString("7"));
        assertThat(tracks.get(14).getDetectionProperties().get("TEXT"), containsString("8"));
        assertThat(tracks.get(15).getDetectionProperties().get("TEXT"), containsString("9"));
        assertThat(tracks.get(16).getDetectionProperties().get("TEXT"), containsString("10"));
        assertThat(tracks.get(17).getDetectionProperties().get("TEXT"), containsString("11"));
        assertThat(tracks.get(18).getDetectionProperties().get("TEXT"), containsString("12"));
    }

    private void assertSection(MPFGenericTrack track, String page, String section, String language, String text) {
        assertEquals(page, track.getDetectionProperties().get("PAGE_NUM"));
        assertEquals(section, track.getDetectionProperties().get("SECTION_NUM"));
        assertEquals(language, track.getDetectionProperties().get("TEXT_LANGUAGE"));
        assertThat(track.getDetectionProperties().get("TEXT"), containsString(text));
    }
}
