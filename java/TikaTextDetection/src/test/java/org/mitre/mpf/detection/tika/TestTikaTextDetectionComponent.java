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
import org.mitre.mpf.component.api.detection.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import static org.hamcrest.CoreMatchers.containsString;

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
    public void testGetDetectionsGeneric() throws Exception {
        String uri = "./test/data/Testing TIKA DETECTION.pptx";
        String text_tag_json ="./test/config/test-text-tags-foreign.json";
        Map<String, String> jobProperties = new HashMap<String, String>();
        Map<String, String> mediaProperties = new HashMap<String, String>();
        String textTags = "text-tags.json";
        jobProperties.put("TAGGING_FILE", text_tag_json);
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");

        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", uri, jobProperties, mediaProperties);
        boolean debug = false;

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals("Number of expected tracks does not match.", 11 ,tracks.size());
        // Test each output type.

        // Test single trigger word tag and text extraction.
        // Same keyword should register under three tags, two case-insensitive, one case-sensitive.
        MPFGenericTrack testTrack = tracks.get(0);
        assertEquals("Expected language does not match.", "English", testTrack.getDetectionProperties().get("TEXT_LANGUAGE"));
        assertEquals("Expected text does not match.", "Testing Text Detection\nSlide 1", testTrack.getDetectionProperties().get("TEXT"));
        assertEquals("Expected tags not found.", "personal; case-sensitive-tag; case-insensitive-tag", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "Text", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "8-11", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test language extraction.
        testTrack = tracks.get(1);
        assertEquals("Expected language does not match.", "Japanese", testTrack.getDetectionProperties().get("TEXT_LANGUAGE"));

        // Test regex tagging.
        testTrack = tracks.get(2);
        assertEquals("Expected tags not found.", "personal; identity document", testTrack.getDetectionProperties().get("TAGS"));

        assertEquals("Expected trigger words not found.", "000-000-0000; citizen", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "25-36; 17-23", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test single tag detection with multiple trigger words (one trigger word also repeats).
        testTrack = tracks.get(3);
        assertEquals("Expected tag not found.", "vehicle", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "auto; bike", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "3-6, 68-71; 21-24", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test no detections.
        testTrack = tracks.get(4);
        assertEquals("Text should be empty", "", testTrack.getDetectionProperties().get("TEXT"));
        assertEquals("Language should be empty", "Unknown", testTrack.getDetectionProperties().get("TEXT_LANGUAGE"));
        assertEquals("Tags should be empty", "" , testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Trigger words should be empty.", "", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Trigger word offsets should be empty.", "", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test multiple regex tags.
        testTrack = tracks.get(5);
        assertEquals("Expected tags not found.", "identity document; travel", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "passport; Passenger", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "10-17; 0-8", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));


        // Test multiple regex tags.
        testTrack = tracks.get(6);
        assertEquals("Expected tags not found.", "financial; personal", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "Financ; 00-000-0000", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "0-5; 15-25", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test multiple regex tags.
        // Three tags should be picked up, with one tag supporting two trigger word detections
        // resulting in 4 trigger words picked up in total.
        testTrack = tracks.get(7);
        assertEquals("Expected tags not found.", "vehicle; financial; personal", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "Bus; Financ; ATM; 102-123-1231", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "0-2; 8-13; 4-6; 21-32", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test delimiter and escaped backspace text tagging.
        testTrack = tracks.get(8);
        assertEquals("Expected tag not found.", "delimiter-test; backslash", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "a[[;] ]b; \\", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "25-30; 36, 37, 40, 43, 44", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test phrase detection.
        // A single short phrase "spirit of brotherhood" is expected.
        testTrack = tracks.get(9);
        assertThat(testTrack.getDetectionProperties().get("TEXT"), containsString("All human beings are born free"));
        assertEquals("Expected tag not found.", "key-phrase", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger phrase not found.", "spirit of brotherhood", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger phrase offset not found.", "229-249", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // Test case-sensitive tagging.
        // Case-sensitive tag for "Text" should not be detected.
        testTrack = tracks.get(10);
        assertThat(testTrack.getDetectionProperties().get("TEXT"), containsString("End slide test text"));
        assertEquals("Expected tag not found.", "personal; case-insensitive-tag", testTrack.getDetectionProperties().get("TAGS"));
        assertTrue("Case sensitive tag for \"Text\" should not have been found", testTrack.getDetectionProperties().get("TAGS").contains("case-insensitive-tag"));
        assertEquals("Expected trigger words not found.", "01/01/2011; text", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "24-33; 19-22", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

        // For human testing.
        if (debug) {
            for (int i = 0; i < tracks.size(); i++) {
                MPFGenericTrack track = tracks.get(i);
                System.out.println(String.format("  Generic track number %d", i));
                System.out.println(String.format("  Confidence = %f", track.getConfidence()));
                System.out.println(String.format("  Text = %s", track.getDetectionProperties().get("TEXT")));
                System.out.println(String.format("  Language = %s", track.getDetectionProperties().get("TEXT_LANGUAGE")));
                System.out.println(String.format("  Text Tags = %s", track.getDetectionProperties().get("TAGS")));
                System.out.println(String.format("  Text Trigger Words = %s", track.getDetectionProperties().get("TRIGGER_WORDS")));
                System.out.println(String.format("  Text Trigger Words Offset = %s", track.getDetectionProperties().get("TRIGGER_WORDS_OFFSET")));
                assertEquals("Confidence does not match.", -1.0f, track.getConfidence(),0.1f);
            }
        }
    }

    @Test
    public void testGetDetectionsShortRegexSearch() throws Exception {
        String uri = "./test/data/Testing TIKA DETECTION.pptx";
        String text_tag_json = "./test/config/test-text-tags-foreign.json";
        Map<String, String> jobProperties = new HashMap<String, String>();
        Map<String, String> mediaProperties = new HashMap<String, String>();
        String textTags = "text-tags.json";
        jobProperties.put("TAGGING_FILE", text_tag_json);
        jobProperties.put("MIN_CHARS_FOR_LANGUAGE_DETECTION", "20");
        jobProperties.put("LIST_ALL_PAGES", "true");
        jobProperties.put("FULL_REGEX_SEARCH", "false");
        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", uri, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
        assertEquals("Number of expected tracks does not match.", 11 ,tracks.size());

        // Test single tag detection with multiple trigger words (one trigger word also repeats).
        // With FULL_REGEX_SEARCH disabled only one should be found.
        MPFGenericTrack testTrack = tracks.get(3);
        assertEquals("Expected only one regex tag to be found", "vehicle", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected only one trigger word.", "auto", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Incorrect trigger word offset.", "3-6", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));


        // Test text, regex tags.
        // This time, each tag only pickes up one trigger word each for three trigger words in total.
        testTrack = tracks.get(7);
        assertEquals("Expected tags not found.", "vehicle; financial; personal", testTrack.getDetectionProperties().get("TAGS"));
        assertEquals("Expected trigger words not found.", "Bus; Financ; 102-123-1231", testTrack.getDetectionProperties().get("TRIGGER_WORDS"));
        assertEquals("Expected trigger word offsets not found.", "0-2; 8-13; 21-32", testTrack.getDetectionProperties().get("TRIGGER_WORDS_OFFSET"));

    }
}
