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

import java.nio.file.Paths;

/**
 * The test class provides the framework for developing Java components.  Test cases can be prepared for a variety
 * of input conditions, and can cover successful executions as well as error conditions.  In most cases, if the
 * getDetections() and support() methods are correctly implemented, the component will work properly.  In cases where
 * the init() or close() methods are overridden, those also should be tested.
 */
public class TikaTextDetectionComponentTest {

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
        Map<String, String> jobProperties = new HashMap<String, String>();
        Map<String, String> mediaProperties = new HashMap<String, String>();
        String textTags = "text-tags.json";
        String rundirectory = tikaComponent.getRunDirectory();
        jobProperties.put("TAGGING_FILE", rundirectory+"/TikaTextDetection/plugin-files/config/"+textTags);
        MPFGenericJob genericJob = new MPFGenericJob("TestGenericJob", uri, jobProperties, mediaProperties);
        boolean debug=false;

        try {
            List<MPFGenericTrack> tracks = tikaComponent.getDetections(genericJob);
            assertEquals("Number of expected tracks does not match.", 6 ,tracks.size());
            // Test each output type.
            if(tracks.size() == 6){
                MPFGenericTrack track = tracks.get(1);
                assertEquals("Expected language does not match.", "English", track.getDetectionProperties().get("TEXT_LANGUAGE"));
                track = tracks.get(2);
                assertEquals("Expected language does not match.", "Japanese", track.getDetectionProperties().get("TEXT_LANGUAGE"));
                track = tracks.get(3);
                assertEquals("Expected tags not found.", "personal", track.getDetectionProperties().get("TAGS_STRING"));
                assertEquals("Expected tags not found.", "personal", track.getDetectionProperties().get("TAGS_REGEX"));
                track = tracks.get(5);
                assertThat(track.getDetectionProperties().get("TEXT"), containsString("End slide test text"));
            }
            
            // For human testing.
            if(debug){
                for (int i = 0; i < tracks.size(); i++) {
                    MPFGenericTrack track = tracks.get(i);
                    System.out.println(String.format("Generic track number %d", i));
                    System.out.println(String.format("  Confidence = %f", track.getConfidence()));
                    System.out.println(String.format("  Text = %s", track.getDetectionProperties().get("TEXT")));
                    System.out.println(String.format("  Language = %s", track.getDetectionProperties().get("TEXT_LANGUAGE")));
                    System.out.println(String.format("  Split Tags = %s", track.getDetectionProperties().get("TAGS_STRING")));
                    System.out.println(String.format("  Regex Tags = %s", track.getDetectionProperties().get("TAGS_REGEX")));
                    assertEquals("Confidence does not match.", -1.0f, track.getConfidence(),0.1f);
                }
            }
        } catch (MPFComponentDetectionError e) {
            System.err.println(String.format("An error occurred of type ", e.getDetectionError().name()));
            fail(String.format("An error occurred of type ", e.getDetectionError().name()));
        }
    }
}
