#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
#                                                                           #
# Licensed under the Apache License, Version 2.0 (the "License");           #
# you may not use this file except in compliance with the License.          #
# You may obtain a copy of the License at                                   #
#                                                                           #
#    http://www.apache.org/licenses/LICENSE-2.0                             #
#                                                                           #
# Unless required by applicable law or agreed to in writing, software       #
# distributed under the License is distributed on an "AS IS" BASIS,         #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  #
# See the License for the specific language governing permissions and       #
# limitations under the License.                                            #
#############################################################################

from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import unittest
import unittest.mock

import mpf_component_api as mpf

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from llama_video_summarization_component import LlamaVideoSummarizationComponent, ChildProcess

logging.basicConfig(level=logging.DEBUG)

USE_MOCKS = eval(os.environ.get('USE_MOCKS', 'True'))
TEST_DATA = pathlib.Path(__file__).parent / 'data'

DUMMY_TIMELINE = \
[
    {
        "timestamp_start": 0,
        "timestamp_end": 50,
        "description": "Something happened."
    },
    {
        "timestamp_start": 51,
        "timestamp_end": 99,
        "description": "Something else happened."
    }
]

class TestComponent(unittest.TestCase):

    def run_patched_job(self, component, job, response):
        detection_func = component.get_detections_from_video

        if not USE_MOCKS:
            return detection_func(job)

        with unittest.mock.patch.object(ChildProcess, '__init__', lambda: None):
            with unittest.mock.patch.object(ChildProcess, '__del__', lambda: None):
                with unittest.mock.patch.object(ChildProcess, 'send_job_get_response', lambda x, y: response):
                    return detection_func(job)

    def test_multiple_videos(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100, {}, {})
        results = self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a cat.",
                "video_length": 100,
                "video_event_timeline": DUMMY_TIMELINE
            }))
        
        job = mpf.VideoJob('dog job', str(TEST_DATA / 'dog.mp4'), 0, 100, {}, {})
        results = self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a dog.",
                "video_length": 100,
                "video_event_timeline": DUMMY_TIMELINE
            }))
        self.assertIn("dog", results[0].detection_properties["TEXT"])


    def test_invalid_timeline(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100,
            { 
                "GENERATION_MAX_ATTEMPTS" : "1"
            }, 
            {})
        with self.assertRaises(mpf.DetectionException) as cm:
            results = self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a cat.",
                "video_length": 500,
                "video_event_timeline": DUMMY_TIMELINE
            }))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Video length", str(cm.exception))

        # test disabling time check
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100, 
            {
                "GENERATION_MAX_ATTEMPTS" : "1",
                "TIMELINE_CHECK_THRESHOLD" : "-1"
            },
            {})
        results = self.run_patched_job(component, job, json.dumps(
        {
            "video_summary": "This is a video of a cat.",
            "video_length": 500,
            "video_event_timeline": DUMMY_TIMELINE
        }))
        self.assertIn("cat", results[0].detection_properties["TEXT"])


    def test_invalid_json_response(self):
        component = LlamaVideoSummarizationComponent()
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100,
            {
                "GENERATION_MAX_ATTEMPTS" : "1"
            },
            {})

        with self.assertRaises(mpf.DetectionException) as cm:
            results = self.run_patched_job(component, job, "garbage xyz")
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("not valid JSON", str(cm.exception))


    def test_empty_response(self):
        component = LlamaVideoSummarizationComponent()
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 100,
            {
                "GENERATION_MAX_ATTEMPTS" : "1"
            },
            {})

        with self.assertRaises(mpf.DetectionException) as cm:
            results = self.run_patched_job(component, job, "")
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Empty response", str(cm.exception))

    def test_video_segmentation(self):
        component = LlamaVideoSummarizationComponent()
        job1 = mpf.VideoJob(
            job_name='japan street walk 1',
            data_uri=str( TEST_DATA / 'ECMuU75kJCs.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=4477,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(FRAME_COUNT=4477),
            feed_forward_track=None)
        job2 = mpf.VideoJob(
            job_name='japan street walk 2',
            data_uri=str( TEST_DATA / 'ECMuU75kJCs.3m0s-6m0s.mp4'),
            start_frame=4478,
            stop_frame=8818,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(FRAME_COUNT=4339),
            feed_forward_track=None)
        json1 = {'video_summary': 'The video showcases a vending machine, a quiet street, and various buildings in an urban setting. It begins with a close-up of the vending machine, then pans to show the street, including a small cafe, before continuing to display more buildings and streets.', 'video_length': 180.0, 'video_event_timeline': [{'timestamp_start': 0.0, 'timestamp_end': 26.57, 'description': 'Close-up of a vending machine filled with drinks.'}, {'timestamp_start': 26.57, 'timestamp_end': 49.44, 'description': 'View of a street with a cafe and a pedestrian crossing.'}, {'timestamp_start': 49.44, 'timestamp_end': 105.84, 'description': 'A view of a sidewalk next to a building with a tree and bushes.'}, {'timestamp_start': 105.84, 'timestamp_end': 139.82, 'description': 'A view of a sidewalk next to a building with a tree and bushes.'}, {'timestamp_start': 139.82, 'timestamp_end': 167.34, 'description': 'A view of a sidewalk next to a building with a tree and bushes.'}, {'timestamp_start': 167.34, 'timestamp_end': 181.67, 'description': 'A view of a sidewalk next to a building with a tree and bushes.'}]}
        job1_result = self.run_patched_job(component, job1, json.dumps(json1))

        json2 = {'video_summary': 'The video starts with a close-up of a vending machine filled with various beverages, then pans out to show the vending machine and its surroundings. It then transitions to a view of a quiet street with a crosswalk and buildings, followed by a series of views of different parts of the street, including sidewalks, buildings, and construction areas. The video concludes with a view of a narrow street with a vending machine and a blue awning.', 'video_length': 180.0, 'video_event_timeline': [{'timestamp_start': 0.0, 'timestamp_end': 24.3, 'description': 'Close-up of a vending machine filled with beverages'}, {'timestamp_start': 24.3, 'timestamp_end': 50.3, 'description': 'Panning shot of a quiet street with a crosswalk and buildings'}, {'timestamp_start': 50.3, 'timestamp_end': 119.3, 'description': 'Views of different parts of the street, including sidewalks, buildings, and construction areas'}, {'timestamp_start': 119.3, 'timestamp_end': 181.2, 'description': 'Narrow street with a vending machine and a blue awning'}]}
        job2_result = self.run_patched_job(component, job2, json.dumps(json2))
        
        self.assertIsInstance(job1_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job1_result[0].detection_properties)
        self.assertEquals(job1_result[0].detection_properties['SEGMENT_ID'], '0-4477')
        self.assertEquals(job1_result[1].detection_properties['SEGMENT_ID'], '0-4477')
        self.assertRegexpMatches(job1_result[0].detection_properties['TEXT'], r'city|street|urban|architecture|signs|building|people|pedestrian')
        self.assertRegexpMatches(job1_result[1].detection_properties['TEXT'], r'vending|machine|drink')

        self.assertIsInstance(job2_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job2_result[0].detection_properties)
        self.assertEquals(job2_result[0].detection_properties['SEGMENT_ID'], '4478-8817')
        self.assertEquals(job2_result[1].detection_properties['SEGMENT_ID'], '4478-8817')
        self.assertRegexpMatches(job2_result[0].detection_properties['TEXT'], r'city|street|urban|architecture|signs|building|people|pedestrian')

    def test_video_segmentation_24fps(self):
        component = LlamaVideoSummarizationComponent()
        job1 = mpf.VideoJob(
            job_name='24fps-1',
            data_uri=str( TEST_DATA / '9258091-hd_1920_1080_24fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=4314,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(),
            feed_forward_track=None)
        job2 = mpf.VideoJob(
            job_name='24fps-2',
            data_uri=str( TEST_DATA / '9258091-hd_1920_1080_24fps.3m0s-4m23s.mp4'),
            start_frame=4315,
            stop_frame=6323,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(),
            feed_forward_track=None)
        json1 = {'video_summary': 'The video captures a busy city street with various vehicles and pedestrians moving along the road. The scene is set against a backdrop of tall buildings and trees, with a clear sky overhead. The video showcases the continuous flow of traffic, including cars, buses, motorcycles, and bicycles. Pedestrians are seen walking on the sidewalks and crossing the street at designated crosswalks.', 'video_length': 180.0, 'video_event_timeline': [{'timestamp_start': 0.0, 'timestamp_end': 179.9, 'description': 'A busy city street with various vehicles and pedestrians.'}]}
        job1_result = self.run_patched_job(component, job1, json.dumps(json1))

        json2 = {'video_summary': 'A busy urban street with tall buildings, a red bike lane in the middle, and various vehicles including cars, buses, and motorcycles moving along it.', 'video_length': 83.46, 'video_event_timeline': [{'timestamp_start': 0.0, 'timestamp_end': 83.46, 'description': 'Traffic flow on a busy street with a red bike lane.'}]}
        job2_result = self.run_patched_job(component, job2, json.dumps(json2))
        
        self.assertIsInstance(job1_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job1_result[0].detection_properties)
        self.assertEquals(job1_result[0].detection_properties['SEGMENT_ID'], '0-4314')
        self.assertEquals(job1_result[1].detection_properties['SEGMENT_ID'], '0-4314')
        self.assertRegexpMatches(job1_result[0].detection_properties['TEXT'], r'busy|city|street|urban|architecture|signs|building|people|pedestrian')
        self.assertRegexpMatches(job1_result[1].detection_properties['TEXT'], r'busy|city|street|urban|architecture|signs|building|people|pedestrian')

        self.assertIsInstance(job2_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job2_result[0].detection_properties)
        self.assertEquals(job2_result[0].detection_properties['SEGMENT_ID'], '4315-6323')
        self.assertEquals(job2_result[1].detection_properties['SEGMENT_ID'], '4315-6323')
        self.assertRegexpMatches(job2_result[0].detection_properties['TEXT'], r'city|street|urban|architecture|signs|building|people|traffic|busy|bike|lane')

    def test_video_segmentation_30fps(self):
        component = LlamaVideoSummarizationComponent()
        job1 = mpf.VideoJob(
            job_name='30fps-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=4314,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(),
            feed_forward_track=None)
        job2 = mpf.VideoJob(
            job_name='30fps-2',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.3m0s-5m1s.mp4'),
            start_frame=4315,
            stop_frame=6323,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(),
            feed_forward_track=None)
       
        json1 = {'video_summary': 'The video shows an aerial view of a city intersection with a Chase bank building in the center. There is a protest happening in front of the building with people holding signs and banners. The traffic is moving smoothly, and there are pedestrians walking on the sidewalks.', 'video_length': 180.0, 'video_event_timeline': [{'timestamp_start': 0.0, 'timestamp_end': 17.5, 'description': 'Aerial view of city intersection with Chase bank building'}, {'timestamp_start': 18.9, 'timestamp_end': 34.7, 'description': 'Protest in front of Chase bank building'}, {'timestamp_start': 36.1, 'timestamp_end': 50.2, 'description': 'People holding signs and banners'}, {'timestamp_start': 51.6, 'timestamp_end': 72.5, 'description': 'Traffic moving smoothly'}, {'timestamp_start': 73.9, 'timestamp_end': 153.6, 'description': 'Pedestrians walking on sidewalks'}]}
        job1_result = self.run_patched_job(component, job1, json.dumps(json1))
        json2 = {"video_summary": "The video shows a busy street with people walking, cars driving by, and a large building in the background. The camera pans to show a group of people standing on the sidewalk near a fountain. They are holding signs and flags, and some are taking pictures. The camera zooms in on the group, and we see that they are protesting against the government.", "video_length": 120.6, "video_event_timeline": [
                {"timestamp_start": 0.0, "timestamp_end": 4.8, "description": "The video starts with a view of a busy street."},
                {"timestamp_start": 4.8, "timestamp_end": 29.6, "description": "The camera pans to show a group of people standing on the sidewalk near a fountain."},
                {"timestamp_start": 29.6, "timestamp_end": 75.3, "description": "The camera zooms in on the group, and we see that they are protesting against the government."},
                {"timestamp_start": 75.3, "timestamp_end": 121.1, "description": "The camera captures the protesters holding signs and flags, and some are taking pictures."}]}
        job2_result = self.run_patched_job(component, job2, json.dumps(json2))
        
        self.assertIsInstance(job1_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job1_result[0].detection_properties)
        self.assertEquals(job1_result[0].detection_properties['SEGMENT_ID'], '0-4314')
        self.assertEquals(job1_result[1].detection_properties['SEGMENT_ID'], '0-4314')
        self.assertRegexpMatches(job1_result[0].detection_properties['TEXT'], r'city|street|building|people|fountain|signs|protest|demonstration|traffic')
        self.assertRegexpMatches(job1_result[1].detection_properties['TEXT'], r'city|street|building|people|fountain|signs|protest|demonstration|traffic')

        self.assertIsInstance(job2_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job2_result[0].detection_properties)
        self.assertEquals(job2_result[0].detection_properties['SEGMENT_ID'], '4315-6323')
        self.assertEquals(job2_result[1].detection_properties['SEGMENT_ID'], '4315-6323')
        self.assertRegexpMatches(job2_result[0].detection_properties['TEXT'], r'aerial|street|protest|banner|flags|signs|people')
    
    def test_video_segmentation_vfr(self):
        component = LlamaVideoSummarizationComponent()
        job1 = mpf.VideoJob(
            job_name='vfr-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_VFR.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=180,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(FPS=1),
            feed_forward_track=None)
        job2 = mpf.VideoJob(
            job_name='vfr-2',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_VFR.3m0s-5m1s.mp4'),
            start_frame=181,
            stop_frame=480,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=2,
                MAX_FRAMES=360,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=dict(FPS=2),
            feed_forward_track=None)
       
        json1 = {'video_summary': 'The video shows an aerial view of a city intersection with a Chase bank building in the center. There is a protest happening in front of the building with people holding signs and banners. The traffic is moving smoothly, and there are pedestrians walking on the sidewalks.', 'video_length': 180.0, 'video_event_timeline': [{'timestamp_start': 0.0, 'timestamp_end': 17.5, 'description': 'Aerial view of city intersection with Chase bank building'}, {'timestamp_start': 18.9, 'timestamp_end': 34.7, 'description': 'Protest in front of Chase bank building'}, {'timestamp_start': 36.1, 'timestamp_end': 50.2, 'description': 'People holding signs and banners'}, {'timestamp_start': 51.6, 'timestamp_end': 72.5, 'description': 'Traffic moving smoothly'}, {'timestamp_start': 73.9, 'timestamp_end': 153.6, 'description': 'Pedestrians walking on sidewalks'}]}
        job1_result = self.run_patched_job(component, job1, json.dumps(json1))
        json2 = {"video_summary": "The video shows a busy street with people walking, cars driving by, and a large building in the background. The camera pans to show a group of people standing on the sidewalk near a fountain. They are holding signs and flags, and some are taking pictures. The camera zooms in on the group, and we see that they are protesting against the government.", "video_length": 120.6, "video_event_timeline": [
                {"timestamp_start": 0.0, "timestamp_end": 4.8, "description": "The video starts with a view of a busy street."},
                {"timestamp_start": 4.8, "timestamp_end": 29.6, "description": "The camera pans to show a group of people standing on the sidewalk near a fountain."},
                {"timestamp_start": 29.6, "timestamp_end": 75.3, "description": "The camera zooms in on the group, and we see that they are protesting against the government."},
                {"timestamp_start": 75.3, "timestamp_end": 121.1, "description": "The camera captures the protesters holding signs and flags, and some are taking pictures."}]}
        job2_result = self.run_patched_job(component, job2, json.dumps(json2))
        
        self.assertIsInstance(job1_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job1_result[0].detection_properties)
        self.assertEquals(job1_result[0].detection_properties['SEGMENT_ID'], '0-180')
        self.assertEquals(job1_result[1].detection_properties['SEGMENT_ID'], '0-180')
        self.assertRegexpMatches(job1_result[0].detection_properties['TEXT'], r'city|street|building|people|fountain|signs|protest|demonstration|traffic')
        self.assertRegexpMatches(job1_result[1].detection_properties['TEXT'], r'city|street|building|people|fountain|signs|protest|demonstration|traffic')

        self.assertIsInstance(job2_result[0], mpf.VideoTrack)
        self.assertIn('SEGMENT_SUMMARY', job2_result[0].detection_properties)
        self.assertEquals(job2_result[0].detection_properties['SEGMENT_ID'], '181-422')
        self.assertEquals(job2_result[1].detection_properties['SEGMENT_ID'], '181-422')
        self.assertRegexpMatches(job2_result[0].detection_properties['TEXT'], r'aerial|street|protest|banner|flags|signs|people')

if __name__ == "__main__":
    unittest.main(verbosity=2)