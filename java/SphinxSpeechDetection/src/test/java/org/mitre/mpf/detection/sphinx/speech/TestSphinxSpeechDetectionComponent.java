/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2022 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2022 The MITRE Corporation                                       *
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

package org.mitre.mpf.detection.sphinx.speech;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.mitre.mpf.component.api.detection.*;

import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

/**
 * The test class provides the framework for developing Java components.  Test cases can be prepared for a variety
 * of input conditions, and can cover successful executions as well as error conditions.  In most cases, if the
 * getDetections() and support() methods are correctly implemented, the component will work properly.  In cases where
 * the init() or close() methods are overridden, those also should be tested.
 */
public class TestSphinxSpeechDetectionComponent {

    private SphinxSpeechDetectionComponent sphinxComponent;

    @Before
    public void setUp() {
        sphinxComponent = new SphinxSpeechDetectionComponent();
        sphinxComponent.init();
    }

    @After
    public void tearDown() {
        sphinxComponent.close();
    }

    @Test
    public void testGetDetectionsAudio() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/left.wav").getPath();
        MPFAudioJob audioJob = new MPFAudioJob("Job TestRun:TestAudioJob", mediaPath, Map.of(),
                Map.of("DURATION", "5020"), 0, -1);
        List<MPFAudioTrack> tracks = sphinxComponent.getDetections(audioJob);
        assertEquals("Unexpected number of tracks.", 1 ,tracks.size());
        assertEquals("Unexpected transcript.", "this three left on the left side of the one closest to us",
                tracks.get(0).getDetectionProperties().get("TRANSCRIPT"));
    }

    @Test
    public void testGetDetectionsVideo() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/left.avi").getPath();
        MPFVideoJob videoJob = new MPFVideoJob("Job TestRun:TestVideoJob", mediaPath, Map.of(),
                Map.of("DURATION", "5008", "FPS", "24", "FRAME_COUNT", "122"), 0, 121);
        List<MPFVideoTrack> tracks = sphinxComponent.getDetections(videoJob);
        assertEquals("Unexpected number of tracks.", 1 ,tracks.size());
        assertEquals("Unexpected transcript.", "this three left on the left side of the one closest to us",
                tracks.get(0).getDetectionProperties().get("TRANSCRIPT"));
    }
}
