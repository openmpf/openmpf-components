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
from unittest import mock

import mpf_component_api as mpf

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from llama_video_summarization_component import LlamaVideoSummarizationComponent, ChildProcess

logging.basicConfig(level=logging.DEBUG)

USE_MOCKS = eval(os.environ.get('USE_MOCKS', 'True'))
TEST_DATA = pathlib.Path(__file__).parent / 'data'

DUMMY_TIMELINE = \
[
    {
        "timestamp_start": 0.0,
        "timestamp_end": 3.0,
        "description": "Something happened."
    },
    {
        "timestamp_start": 3.1,
        "timestamp_end": 6.8,
        "description": "Something else happened."
    }
]

CAT_TIMELINE = {
    "video_summary": "A cat is sitting on a cobblestone street, looking around as people walk by.",
    "video_length": 6.8,
    "video_event_timeline": [
        {
            "timestamp_start": 0.0,
            "timestamp_end": 4.9,
            "description": "The cat is sitting on the cobblestone street, looking around."
        },
        {
            "timestamp_start": 5.0,
            "timestamp_end": 6.8,
            "description": "The cat looks back at the camera and then walks away."
        }
    ]
}

CAT_VIDEO_PROPERTIES = {
    'DURATION': '6890',
    'FPS': '25',
    'FRAME_COUNT': '172',
    'FRAME_HEIGHT': '360',
    'FRAME_WIDTH': '640',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

DOG_TIMELINE = {
    "video_summary": "A dog sitting by a window and looking around.",
    "video_length": 6.12,
    "video_event_timeline": [
        {
            "timestamp_start": 0.0,
            "timestamp_end": 6.12,
            "description": "Dog sitting by the window."
        }
    ]
}

DOG_VIDEO_PROPERTIES = {
    'DURATION': '6170',
    'FPS': '25',
    'FRAME_COUNT': '154',
    'FRAME_HEIGHT': '240',
    'FRAME_WIDTH': '426',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

SHORT_TIMELINE = {
    "video_summary": "A person is running around.",
    "video_length": 1,
    "video_event_timeline": [
        {
            "timestamp_start": 0.0,
            "timestamp_end": 1.0,
            "description": "A person running."
        }
    ]
}

SHORT_VIDEO_PROPERTIES = {
    'DURATION': '1000',
    'FPS': '1',
    'FRAME_COUNT': '1',
    'FRAME_HEIGHT': '100',
    'FRAME_WIDTH': '200',
    'HAS_CONSTANT_FRAME_RATE': 'true',
    'MIME_TYPE': 'video/mp4',
    'ROTATION': '0.0'
}

class TestComponent(unittest.TestCase):

    def setUp(self):
        if not USE_MOCKS:
            return

        child_process_init_patcher = mock.patch('llama_video_summarization_component.ChildProcess.__init__')
        self.mock_child_process_init = child_process_init_patcher.start()
        self.mock_child_process_init.return_value = None
        self.addCleanup(child_process_init_patcher.stop)

        child_process_del_patcher = mock.patch('llama_video_summarization_component.ChildProcess.__del__')
        self.mock_child_process_del = child_process_del_patcher.start()
        self.addCleanup(child_process_del_patcher.stop)

        child_process_send_job_patcher = mock.patch('llama_video_summarization_component.ChildProcess.send_job_get_response')
        self.mock_child_process_send_job = child_process_send_job_patcher.start()
        self.addCleanup(child_process_send_job_patcher.stop)
        

    def run_patched_job(self, component, job, response):
        if USE_MOCKS:
            self.mock_child_process_send_job.return_value = response

        return component.get_detections_from_video(job)
    

    def assert_detection_region(self, detection, frame_width, frame_height):    
        self.assertEquals(0, detection.x_left_upper)
        self.assertEquals(0, detection.y_left_upper)
        self.assertEquals(frame_width, detection.width)
        self.assertEquals(frame_height, detection.height)


    def assert_first_middle_last_detections(self, track, frame_width, frame_height):    
        self.assertIn(track.start_frame, track.frame_locations)
        self.assert_detection_region(track.frame_locations[track.start_frame], frame_width, frame_height)

        self.assertIn(track.stop_frame, track.frame_locations)
        self.assert_detection_region(track.frame_locations[track.stop_frame], frame_width, frame_height)

        middle_frame = int((track.stop_frame - track.start_frame) / 2) + track.start_frame
        self.assertIn(middle_frame, track.frame_locations)
        self.assert_detection_region(track.frame_locations[middle_frame], frame_width, frame_height)


    def test_multiple_videos(self):
        component = LlamaVideoSummarizationComponent()


        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 171, {}, CAT_VIDEO_PROPERTIES, {})
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        results = self.run_patched_job(component, job, json.dumps(CAT_TIMELINE))
        self.assertEquals(3, len(results))

        self.assertEquals('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertEquals(6.8, results[0].detection_properties['SEGMENT LENGTH'])
        self.assertIn("looking around as people walk by.", results[0].detection_properties["TEXT"])
        self.assertEquals(0, results[0].start_frame)
        self.assertEquals(171, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("looking around.", results[1].detection_properties["TEXT"])
        self.assertEquals(0, results[1].start_frame) # 0 * 25
        self.assertEquals(121, results[1].stop_frame) # (4.9 * 25) - 1
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)

        self.assertIn("looks back at the camera", results[2].detection_properties["TEXT"])
        self.assertEquals(125, results[2].start_frame) # 5.0 * 25
        self.assertEquals(169, results[2].stop_frame) # (6.8 * 25) - 1
        self.assert_first_middle_last_detections(results[2], frame_width, frame_height)


        job = mpf.VideoJob('dog job', str(TEST_DATA / 'dog.mp4'), 0, 153, {}, DOG_VIDEO_PROPERTIES, {})
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        results = self.run_patched_job(component, job, json.dumps(DOG_TIMELINE))
        self.assertEquals(2, len(results))

        self.assertEquals('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertEquals(6.12, results[0].detection_properties['SEGMENT LENGTH'])
        self.assertIn("sitting by a window and looking around", results[0].detection_properties["TEXT"])
        self.assertEquals(0, results[0].start_frame)
        self.assertEquals(153, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("sitting by the window.", results[1].detection_properties["TEXT"])
        self.assertEquals(0, results[1].start_frame) # 0 * 25
        self.assertEquals(152, results[1].stop_frame) # (6.12 * 25) - 1
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)


        job = mpf.VideoJob('short job', str(TEST_DATA / 'short.mp4'), 0, 0, {}, SHORT_VIDEO_PROPERTIES, {})
        frame_width = int(job.media_properties['FRAME_WIDTH'])
        frame_height = int(job.media_properties['FRAME_HEIGHT'])

        results = self.run_patched_job(component, job, json.dumps(SHORT_TIMELINE))
        self.assertEquals(2, len(results))

        self.assertEquals('TRUE', results[0].detection_properties['SEGMENT SUMMARY'])
        self.assertEquals(1.0, results[0].detection_properties['SEGMENT LENGTH'])
        self.assertIn("A person is running around.", results[0].detection_properties["TEXT"])
        self.assertEquals(0, results[0].start_frame)
        self.assertEquals(0, results[0].stop_frame)
        self.assert_first_middle_last_detections(results[0], frame_width, frame_height)

        self.assertIn("A person running.", results[1].detection_properties["TEXT"])
        self.assertEquals(0, results[1].start_frame) # 0 * 1
        self.assertEquals(0, results[1].stop_frame) # (1 * 1) - 1
        self.assert_first_middle_last_detections(results[1], frame_width, frame_height)


    def test_invalid_timeline(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 15000,
            { 
                "GENERATION_MAX_ATTEMPTS" : "1"
            }, 
            CAT_VIDEO_PROPERTIES, {})
        
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job, json.dumps(
            {
                "video_summary": "This is a video of a cat.",
                "video_length": 500,
                "video_event_timeline": DUMMY_TIMELINE
            }))

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("desired segment length", str(cm.exception))

        # test disabling time check
        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 15000, 
            {
                "GENERATION_MAX_ATTEMPTS" : "1",
                "TIMELINE_CHECK_THRESHOLD" : "-1",
                "SEGMENT_LENGTH_CHECK_THRESHOLD": "-1"
            },
            CAT_VIDEO_PROPERTIES, {})
        
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
            CAT_VIDEO_PROPERTIES, {})

        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job, "garbage xyz") # don't care about results
        
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("not valid JSON", str(cm.exception))


    def test_empty_response(self):
        component = LlamaVideoSummarizationComponent()

        job = mpf.VideoJob('cat job', str(TEST_DATA / 'cat.mp4'), 0, 171,
            {
                "GENERATION_MAX_ATTEMPTS" : "1"
            },
            CAT_VIDEO_PROPERTIES, {})

        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job, "") # don't care about results

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Empty response", str(cm.exception))


    def test_video_segmentation_24fps(self):
        component = LlamaVideoSummarizationComponent()

        job_media_properties = {'DURATION': '263765',
            'FPS': '23.976023976023978',
            'FRAME_COUNT': '6324',
            'FRAME_HEIGHT': '720',
            'FRAME_WIDTH': '1280',
            'HAS_CONSTANT_FRAME_RATE': 'true',
            'MIME_TYPE': 'video/mp4',
            'ROTATION': '0.0'}

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
                TIMELINE_CHECK_THRESHOLD=-1,
                SEGMENT_LENGTH_SPECIFICATION='SECONDS',
                TARGET_SEGMENT_LENGTH=180
            ),
            media_properties=job_media_properties,
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
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)

        json1 = {
            'video_summary': 'The video captures a busy city street with various vehicles and pedestrians moving along the road. '
                'The scene is set against a backdrop of tall buildings and trees, with a clear sky overhead. '
                'The video showcases the continuous flow of traffic, including cars, buses, motorcycles, and bicycles. '
                'Pedestrians are seen walking on the sidewalks and crossing the street at designated crosswalks.',
            'video_length': 180.0,
            'video_event_timeline': [
                { 
                    'timestamp_start': 0.0, 
                    'timestamp_end': 179.9, 
                    'description': 'A busy city street with various vehicles and pedestrians.' 
                } 
            ]
        }
        job1_results = self.run_patched_job(component, job1, json.dumps(json1))

        json2 = {
            'video_summary': 'A busy urban street with tall buildings, a red bike lane in the middle, '
                'and various vehicles including cars, buses, and motorcycles moving along it.',
            'video_length': 83.46,
            'video_event_timeline': [
                { 
                    'timestamp_start': 0.0, 
                    'timestamp_end': 83.46, 
                    'description': 'Traffic flow on a busy street with a red bike lane.' 
                }
            ]
        }
        job2_results = self.run_patched_job(component, job2, json.dumps(json2))
        
        self.assertIsInstance(job1_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job1_results[0].detection_properties)
        self.assertEquals(job1_results[0].detection_properties['SEGMENT ID'], '0-4314')
        self.assertEquals(job1_results[1].detection_properties['SEGMENT ID'], '0-4314')
        self.assertRegexpMatches(job1_results[0].detection_properties['TEXT'], r'busy|city|street|urban|architecture|signs|building|people|pedestrian')
        self.assertRegexpMatches(job1_results[1].detection_properties['TEXT'], r'busy|city|street|urban|architecture|signs|building|people|pedestrian')

        self.assertIsInstance(job2_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job2_results[0].detection_properties)
        self.assertEquals(job2_results[0].detection_properties['SEGMENT ID'], '4315-6323')
        self.assertEquals(job2_results[1].detection_properties['SEGMENT ID'], '4315-6323')
        self.assertRegexpMatches(job2_results[0].detection_properties['TEXT'], r'city|street|urban|architecture|signs|building|people|traffic|busy|bike|lane')


    def test_video_segmentation_30fps(self):
        component = LlamaVideoSummarizationComponent()

        job_media_properties = {
            'DURATION': '301739',
            'FPS': '29.97002997002997',
            'FRAME_COUNT': '9043',
            'FRAME_HEIGHT': '1080',
            'FRAME_WIDTH': '1920',
            'HAS_CONSTANT_FRAME_RATE': 'true',
            'MIME_TYPE': 'video/mp4',
            'ROTATION': '0.0'}

        job1 = mpf.VideoJob(
            job_name='30fps-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=5393,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)
        
        job2 = mpf.VideoJob(
            job_name='30fps-2',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.3m0s-5m1s.mp4'),
            start_frame=5394,
            stop_frame=9042,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)
       
        json1 = {
            'video_summary': 'The video shows an aerial view of a city intersection with a bank building in the center. '
                'There is a protest happening in front of the building with people holding signs and banners. '
                'The traffic is moving smoothly, and there are pedestrians walking on the sidewalks.', 
            'video_length': 180.0, 
            'video_event_timeline': [
                {
                    'timestamp_start': 0.0, 
                    'timestamp_end': 17.5, 
                    'description': 'Aerial view of city intersection with bank building'
                }, 
                {
                    'timestamp_start': 18.9, 
                    'timestamp_end': 34.7, 
                    'description': 'Protest in front of bank building'
                }, 
                {
                    'timestamp_start': 36.1, 
                    'timestamp_end': 50.2, 
                    'description': 'People holding signs and banners'
                    }, 
                {
                    'timestamp_start': 51.6, 
                    'timestamp_end': 72.5, 
                    'description': 'Traffic moving smoothly'
                }, 
                {
                    'timestamp_start': 73.9, 
                    'timestamp_end': 153.6, 
                    'description': 'Pedestrians walking on sidewalks'
                }
            ]
        }
        job1_results = self.run_patched_job(component, job1, json.dumps(json1))

        json2 = {
            "video_summary": "The video shows a busy street with people walking, cars driving by, and a large building in the background. "
                "The camera pans to show a group of people standing on the sidewalk near a fountain. "
                "They are holding signs and flags, and some are taking pictures. "
                "The camera zooms in on the group, and we see that they are protesting against the government.", 
            "video_length": 120.6, 
            "video_event_timeline": [
                { 
                    "timestamp_start": 0.0, 
                    "timestamp_end": 4.8, 
                    "description": "The video starts with a view of a busy street." 
                },
                { 
                    "timestamp_start": 4.8,
                    "timestamp_end": 29.6, 
                    "description": "The camera pans to show a group of people standing on the sidewalk near a fountain." 
                },
                { 
                    "timestamp_start": 29.6, 
                    "timestamp_end": 75.3, 
                    "description": "The camera zooms in on the group, and we see that they are protesting against the government." 
                },
                { 
                    "timestamp_start": 75.3, 
                    "timestamp_end": 121.1, 
                    "description": "The camera captures the protesters holding signs and flags, and some are taking pictures." 
                }
            ]
        }
        job2_results = self.run_patched_job(component, job2, json.dumps(json2))
        
        self.assertIsInstance(job1_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job1_results[0].detection_properties)
        self.assertEquals(job1_results[0].detection_properties['SEGMENT ID'], '0-5393')
        self.assertEquals(job1_results[1].detection_properties['SEGMENT ID'], '0-5393')
        self.assertRegexpMatches(job1_results[0].detection_properties['TEXT'], r'city|street|building|people|fountain|signs|protest|demonstration|traffic')
        self.assertRegexpMatches(job1_results[1].detection_properties['TEXT'], r'city|street|building|people|fountain|signs|protest|demonstration|traffic')

        self.assertIsInstance(job2_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job2_results[0].detection_properties)
        self.assertEquals(job2_results[0].detection_properties['SEGMENT ID'], '5394-9042')
        self.assertEquals(job2_results[1].detection_properties['SEGMENT ID'], '5394-9042')
        self.assertRegexpMatches(job2_results[0].detection_properties['TEXT'], r'aerial|street|protest|banner|flags|signs|people')


    def test_check_timeline_threshold(self):
        component = LlamaVideoSummarizationComponent()

        # 6254278-hd_1920_1080_30fps.mp4
        job_media_properties = {'DURATION': '301739',
            'FPS': '29.97002997002997',
            'FRAME_COUNT': '9043',
            'FRAME_HEIGHT': '1080',
            'FRAME_WIDTH': '1920',
            'HAS_CONSTANT_FRAME_RATE': 'true',
            'MIME_TYPE': 'video/mp4',
            'ROTATION': '0.0'}

        job1 = mpf.VideoJob(
            job_name='24fps-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=4314,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=4,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=20,
                SEGMENT_LENGTH_SPECIFICATION='SECONDS',
                TARGET_SEGMENT_LENGTH=180,
                SEGMENT_LENGTH_CHECK_THRESHOLD=30
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)
        
        json1 = {
            'video_summary': 'The video captures a busy city street with various vehicles and pedestrians moving along the road. '
                'The scene is set against a backdrop of tall buildings and trees, with a clear sky overhead. '
                'The video showcases the continuous flow of traffic, including cars, buses, motorcycles, and bicycles. '
                'Pedestrians are seen walking on the sidewalks and crossing the street at designated crosswalks.',
            'video_length': 143.0,
            'video_event_timeline': [
                { 
                    "timestamp_start": 0.0, 
                    "timestamp_end": 36.7, 
                    "description": "Aerial view of the intersection with a bank building in the background." 
                },
                { 
                    "timestamp_start": 37.7, 
                    "timestamp_end": 118.5, 
                    "description": "People are gathered around a fountain, holding signs and flags, seemingly protesting." 
                },
                { 
                    "timestamp_start": 119.5, 
                    "timestamp_end": 201.2, 
                    "description": "Camera pans around the area, showing the surrounding buildings and streets." 
                }
            ]
        }
        
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(json1)) # don't care about result
        
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("last event timestamp", str(cm.exception))

    def test_check_segment_length_threshold(self):
        component = LlamaVideoSummarizationComponent()

        # 6254278-hd_1920_1080_30fps.mp4
        job_media_properties = {'DURATION': '301739',
            'FPS': '29.97002997002997',
            'FRAME_COUNT': '9043',
            'FRAME_HEIGHT': '1080',
            'FRAME_WIDTH': '1920',
            'HAS_CONSTANT_FRAME_RATE': 'true',
            'MIME_TYPE': 'video/mp4',
            'ROTATION': '0.0'}

        job1 = mpf.VideoJob(
            job_name='24fps-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=1797,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=15,
                SEGMENT_LENGTH_SPECIFICATION='SECONDS',
                TARGET_SEGMENT_LENGTH=90,
                SEGMENT_LENGTH_CHECK_THRESHOLD=20
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)
        
        json1 = {
            'video_summary': 'The video captures a busy city street with various vehicles and pedestrians moving along the road. '
                'The scene is set against a backdrop of tall buildings and trees, with a clear sky overhead. '
                'The video showcases the continuous flow of traffic, including cars, buses, motorcycles, and bicycles. '
                'Pedestrians are seen walking on the sidewalks and crossing the street at designated crosswalks.',
            'video_length': 120.0,
            'video_event_timeline': [
                { "timestamp_start": 0.0, "timestamp_end": 36.7, "description": "Aerial view of the intersection with a bank building in the background." },
                { "timestamp_start": 37.7, "timestamp_end": 118.5, "description": "People are gathered around a fountain, holding signs and flags, seemingly protesting." },
                { "timestamp_start": 119.5, "timestamp_end": 201.2, "description": "Camera pans around the area, showing the surrounding buildings and streets." },
                { "timestamp_start": 202.2, "timestamp_end": 235.7, "description": "Bus passing by, and some cars waiting at the traffic light. "
                    "Protesters are chanting and waving their flags, while others are standing on the sidewalk or sitting on benches."}
            ]
        }
        
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(json1)) # don't care about result

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Video segment length", str(cm.exception))

        job2 = mpf.VideoJob(
            job_name='24fps-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=3596,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=3,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=15,
                SEGMENT_LENGTH_SPECIFICATION='SECONDS',
                TARGET_SEGMENT_LENGTH=120,
                SEGMENT_LENGTH_CHECK_THRESHOLD=20
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)
        
        # shorten the timeline
        json1['video_event_timeline'].pop()
        json1['video_event_timeline'].pop()

        job2_results = self.run_patched_job(component, job2, json.dumps(json1))

        self.assertIsInstance(job2_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job2_results[0].detection_properties)
        self.assertEquals(job2_results[0].detection_properties['SEGMENT ID'], '0-3596')
        self.assertEquals(job2_results[1].detection_properties['SEGMENT ID'], '0-3596')

    def test_timeline_integrity(self):
        component = LlamaVideoSummarizationComponent()

        job_media_properties = {
            'DURATION': '301739',
            'FPS': '29.97002997002997',
            'FRAME_COUNT': '9043',
            'FRAME_HEIGHT': '1080',
            'FRAME_WIDTH': '1920',
            'HAS_CONSTANT_FRAME_RATE': 'true',
            'MIME_TYPE': 'video/mp4',
            'ROTATION': '0.0'}

        job1 = mpf.VideoJob(
            job_name='30fps-1',
            data_uri=str( TEST_DATA / '6254278-hd_1920_1080_30fps.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=5393,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=1,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=20
            ),
            media_properties=job_media_properties,
            feed_forward_track=None)

        json1 = {
            "video_summary": "The video shows a protest taking place in front of a bank, with protesters holding signs and flags, and a camera crew capturing the event.",
            "video_length": 179.96,
            "video_event_timeline": [
                {
                    "timestamp_start": 0.0,
                    "timestamp_end": 34.82,
                    "description": "An aerial view of a city square with a fountain in the middle."
                },
                {
                    "timestamp_start": 35.82,
                    "timestamp_end": 64.47,
                    "description": "A group of protesters gather around the fountain, holding signs and flags."
                },
                {
                    "timestamp_start": 65.47,
                    "timestamp_end": 154.29,
                    "description": "The camera captures the protesters from various angles, showing their chants and gestures."
                },
                {
                    "timestamp_start": 185.81,
                    "timestamp_end": 235.77,
                    "description": "The camera zooms in on the protesters, showing their faces and the details of their signs."
                }
            ]
        }
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(json1))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("starts after video segment", str(cm.exception))

        # test event timeline integrity check
        json1["video_event_timeline"].append({
                    "timestamp_start": 236.77,
                    "timestamp_end": 179.96,
                    "description": "The camera pans out to show the entire scene, including the fountain and the surrounding buildings."
                })
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(json1)) # don't care about result

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("invalid timestamps", str(cm.exception))

if __name__ == "__main__":
    unittest.main(verbosity=2)