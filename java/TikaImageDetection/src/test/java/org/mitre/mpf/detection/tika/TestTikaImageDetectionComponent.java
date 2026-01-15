/******************************************************************************
 * NOTICE                                                                     *
 *                                                                            *
 * This software (or technical data) was produced for the U.S. Government     *
 * under contract, and is subject to the Rights in Data-General Clause        *
 * 52.227-14, Alt. IV (DEC 2007).                                             *
 *                                                                            *
 * Copyright 2024 The MITRE Corporation. All Rights Reserved.                 *
 ******************************************************************************/

/******************************************************************************
 * Copyright 2024 The MITRE Corporation                                       *
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
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.mitre.mpf.component.api.detection.MPFComponentDetectionError;
import org.mitre.mpf.component.api.detection.MPFGenericJob;
import org.mitre.mpf.component.api.detection.MPFGenericTrack;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.*;

public class TestTikaImageDetectionComponent {

    private TikaImageDetectionComponent _tikaComponent;

    @Rule
    public TemporaryFolder _tempFolder = new TemporaryFolder();

    private Path _testDir;

    @Before
    public void setUp() {
        _testDir = _tempFolder.getRoot().toPath();

        _tikaComponent = new TikaImageDetectionComponent();
        _tikaComponent.setConfigDirectory("plugin-files/config");
        _tikaComponent.init();
    }

    @After
    public void tearDown() {
        _tikaComponent.close();
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
    public void testGetDetectionsPdf() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);
        boolean debug = false;

        List<MPFGenericTrack> tracks = _tikaComponent.getDetections(genericJob);

        // For human testing.
        if (debug) {
            for (int i = 0; i < tracks.size(); i++) {
                MPFGenericTrack track = tracks.get(i);
                System.out.printf("Generic track number %d%n", i);
                System.out.printf("  Confidence = %f%n", track.getConfidence());
                System.out.printf("  Page = %s%n", track.getDetectionProperties().get("PAGE_NUM"));
                System.out.printf("  Images = %s%n", track.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH"));
            }
        }

        assertEquals(4 ,tracks.size());

        // Test extraction of image 0.
        MPFGenericTrack testTrack = tracks.get(0);
        String tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("1; 2; 3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image0.jpeg"));
        assertTrue(pageCheck(tempPath));

        // Test extraction of image 1.
        testTrack = tracks.get(1);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("2; 3", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image1.jpeg"));
        assertTrue(pageCheck(tempPath));

        // Test extraction of image 2.
        testTrack = tracks.get(2);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("6", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image2.jpeg"));
        assertTrue(pageCheck(tempPath));

        // Test extraction of image 3.
        testTrack = tracks.get(3);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("6", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image3.png"));
        assertTrue(pageCheck(tempPath));

        String uuid = tempPath.split("/")[5];

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image0.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image3.png")));
    }

    @Test
    public void testGetDetectionsPptx() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-image-extraction.pptx").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = _tikaComponent.getDetections(genericJob);

        assertEquals(5 ,tracks.size());

        // TODO: Is it possible to get page (slide) number information from .pptx files?
        // NOTE: By default, Tika will report all images from a .pptx file on the last slide. We override that to -1.
        MPFGenericTrack testTrack = tracks.get(0);
        String tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image0.x-emf"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(1);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image1.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(2);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image2.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(3);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image3.png"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(4);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image4.jpeg"));
        assertTrue(pageCheck(tempPath));

        String uuid = tempPath.split("/")[5];

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image0.x-emf")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image3.png")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image4.jpeg")));
    }

    @Test
    public void testGetDetectionsOdp() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-image-extraction.odp").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = _tikaComponent.getDetections(genericJob);

        assertEquals(5, tracks.size());

        // TODO: Is it possible to get page (slide) number information from .odp files?
        // NOTE: By default, Tika will report all images from an .odp file on slide 0. We override that to -1.
        MPFGenericTrack testTrack = tracks.get(0);
        String tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image0.png"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(1);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image1.png"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(2);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image2.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(3);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image3.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(4);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image4.emf"));
        assertTrue(pageCheck(tempPath));

        String uuid = tempPath.split("/")[5];

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image0.png")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image1.png")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image3.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image4.emf")));
    }

    @Test
    public void testGetDetectionsDocx() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-image-extraction.docx").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = _tikaComponent.getDetections(genericJob);

        assertEquals(4, tracks.size());

        // TODO: Is it possible to get page (slide) number information from .docx files?
        // NOTE: By default, Tika will report all images from a .docx file on slide 0. We override that to -1.
        MPFGenericTrack testTrack = tracks.get(0);
        String tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image0.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(1);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image1.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(2);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image2.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(3);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image3.png"));
        assertTrue(pageCheck(tempPath));

        String uuid = tempPath.split("/")[5];

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image0.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image3.png")));
    }

    @Test
    public void testGetDetectionsOdt() throws MPFComponentDetectionError {
        String mediaPath = this.getClass().getResource("/data/test-tika-image-extraction.odt").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");
        MPFGenericJob genericJob = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks = _tikaComponent.getDetections(genericJob);

        assertEquals(5, tracks.size());

        // TODO: Is it possible to get page (slide) number information from .odt files?
        // NOTE: By default, Tika will report all images from an .odt file on slide 0. We override that to -1.
        MPFGenericTrack testTrack = tracks.get(0);
        String tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image0.png"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(1);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image1.png"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(2);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image2.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(3);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image3.jpeg"));
        assertTrue(pageCheck(tempPath));

        testTrack = tracks.get(4);
        tempPath = testTrack.getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH");
        assertEquals("-1", testTrack.getDetectionProperties().get("PAGE_NUM"));
        assertTrue(tempPath.contains("image4.jpeg"));
        assertTrue(pageCheck(tempPath));

        String uuid = tempPath.split("/")[5];

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image0.png")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image1.png")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image3.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid + "/image4.jpeg")));
    }

    @Test
    public void testMultiMediaJob() throws MPFComponentDetectionError {
        String mediaPath1 = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        String mediaPath2 = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "false");

        // Test that multiple documents submitted under one job ID are processed properly.
        MPFGenericJob genericSubJob1 = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath1, jobProperties, mediaProperties);
        MPFGenericJob genericSubJob2 = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath2, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks1 = _tikaComponent.getDetections(genericSubJob1);
        List<MPFGenericTrack> tracks2 = _tikaComponent.getDetections(genericSubJob2);

        String uuid1 = tracks1.get(0).getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH").split("/")[5];
        String uuid2 = tracks2.get(0).getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH").split("/")[5];

        assertNotEquals(uuid1, uuid2);

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/image0.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/image3.png")));

        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/image0.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/image3.png")));
    }


    @Test
    public void testMultiMediaJobWithSeparatePages() throws MPFComponentDetectionError {
        String mediaPath1 = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        String mediaPath2 = this.getClass().getResource("/data/test-tika-image-extraction.pdf").getPath();
        Map<String, String> jobProperties = new HashMap<>();
        Map<String, String> mediaProperties = new HashMap<>();

        jobProperties.put("SAVE_PATH", _testDir.toString());
        jobProperties.put("ORGANIZE_BY_PAGE", "true");

        // Test that multiple documents submitted under one job ID are processed properly.
        MPFGenericJob genericSubJob1 = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath1, jobProperties, mediaProperties);
        MPFGenericJob genericSubJob2 = new MPFGenericJob("Job TestRun:TestGenericJob", mediaPath2, jobProperties, mediaProperties);

        List<MPFGenericTrack> tracks1 = _tikaComponent.getDetections(genericSubJob1);
        List<MPFGenericTrack> tracks2 = _tikaComponent.getDetections(genericSubJob2);

        String uuid1 = tracks1.get(0).getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH").split("/")[5];
        String uuid2 = tracks2.get(0).getDetectionProperties().get("DERIVATIVE_MEDIA_TEMP_PATH").split("/")[5];

        assertNotEquals(uuid1, uuid2);

        // Check that images were saved correctly.
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/common/image0.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/common/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/page-6/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid1 + "/page-6/image3.png")));

        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/common/image0.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/common/image1.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/page-6/image2.jpeg")));
        assertTrue(Files.exists(Paths.get(_testDir + "/TestRun/tika-extracted/" + uuid2 + "/page-6/image3.png")));
    }
}
