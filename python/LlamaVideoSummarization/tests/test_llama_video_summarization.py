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

DRONE_TIMELINE_SEGMENT_1 = {
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
        }
    ]
}

# events beyond video length
DRONE_TIMELINE_SEGMENT_2 = {
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

DRONE_VIDEO_PROPERTIES = {
    'DURATION': '301739',
    'FPS': '29.97002997002997',
    'FRAME_COUNT': '9043',
    'FRAME_HEIGHT': '1080',
    'FRAME_WIDTH': '1920',
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

    def test_check_segment_length_threshold(self):
        component = LlamaVideoSummarizationComponent()

        job1 = mpf.VideoJob(
            job_name='drone.mp4-segment-2',
            data_uri=str( TEST_DATA / 'drone.3m0s-5m1s.mp4'),
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
            media_properties=DRONE_VIDEO_PROPERTIES,
            feed_forward_track=None)
        
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(DRONE_TIMELINE_SEGMENT_2)) # don't care about result

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("Video segment length", str(cm.exception))

        job2 = mpf.VideoJob(
            job_name='drone.mp4-segment-2',
            data_uri=str( TEST_DATA / 'drone.3m0s-5m1s.mp4'),
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
            media_properties=DRONE_VIDEO_PROPERTIES,
            feed_forward_track=None)

        # shorten timeline
        DRONE_TIMELINE_SEGMENT_2['video_event_timeline'].pop()
        DRONE_TIMELINE_SEGMENT_2['video_event_timeline'].pop()

        job2_results = self.run_patched_job(component, job2, json.dumps(DRONE_TIMELINE_SEGMENT_2))

        self.assertIsInstance(job2_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job2_results[0].detection_properties)
        self.assertEquals(job2_results[0].detection_properties['SEGMENT ID'], '0-3596')
        self.assertEquals(job2_results[1].detection_properties['SEGMENT ID'], '0-3596')

    def test_timeline_integrity(self):
        component = LlamaVideoSummarizationComponent()

        job1 = mpf.VideoJob(
            job_name='drone.mp4-segment-1',
            data_uri=str( TEST_DATA / 'drone.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=5393,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=1,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=20,
                SEGMENT_LENGTH_CHECK_THRESHOLD=30
            ),
            media_properties=DRONE_VIDEO_PROPERTIES,
            feed_forward_track=None)

        DRONE_TIMELINE_SEGMENT_1["video_event_timeline"].append({ 
                    "timestamp_start": 155.5,
                    "timestamp_end": 221.2,
                    "description": "Camera pans around the area, showing the surrounding buildings and streets."})
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(DRONE_TIMELINE_SEGMENT_1)) # don't care about result
        
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("last event timestamp", str(cm.exception))

        DRONE_TIMELINE_SEGMENT_1['video_event_timeline'].pop()
        DRONE_TIMELINE_SEGMENT_1['video_event_timeline'].append({
                "timestamp_start": 185.81,
                "timestamp_end": 235.77,
                "description": "The camera zooms in on the protesters, showing their faces and the details of their signs."
            })

        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(DRONE_TIMELINE_SEGMENT_1))
        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("starts after video segment", str(cm.exception))

        # test event timeline integrity check
        DRONE_TIMELINE_SEGMENT_1["video_event_timeline"].append({
                    "timestamp_start": 236.77,
                    "timestamp_end": 179.96,
                    "description": "The camera pans out to show the entire scene, including the fountain and the surrounding buildings."
                })
        with self.assertRaises(mpf.DetectionException) as cm:
            self.run_patched_job(component, job1, json.dumps(DRONE_TIMELINE_SEGMENT_1)) # don't care about result

        self.assertEqual(mpf.DetectionError.DETECTION_FAILED, cm.exception.error_code)
        self.assertIn("invalid timestamps", str(cm.exception))

        job2 = mpf.VideoJob(
            job_name='drone.mp4-segment-1',
            data_uri=str( TEST_DATA / 'drone.0m0s-3m0s.mp4'),
            start_frame=0,
            stop_frame=5393,
            job_properties=dict(
                GENERATION_MAX_ATTEMPTS=1,
                PROCESS_FPS=1,
                MAX_FRAMES=180,
                MAX_NEW_TOKENS=4096,
                TIMELINE_CHECK_THRESHOLD=-1
            ),
            media_properties=DRONE_VIDEO_PROPERTIES,
            feed_forward_track=None)

        # event that starts within range but ends outside of valid frames
        DRONE_TIMELINE_SEGMENT_1["video_event_timeline"][2]["timestamp_end"] = 185.0
        job2_results = self.run_patched_job(component, job2, json.dumps(DRONE_TIMELINE_SEGMENT_1))

        self.assertIsInstance(job2_results[0], mpf.VideoTrack)
        self.assertIn('SEGMENT SUMMARY', job2_results[0].detection_properties)

        self.assertEquals(job2_results[0].detection_properties['SEGMENT ID'], '0-5393')
        self.assertEquals(job2_results[1].detection_properties['SEGMENT ID'], '0-5393')
        self.assertEquals(job2_results[3].start_frame, 1962)
        self.assertEquals(job2_results[3].stop_frame, 5393)
        self.assertIsNotNone(job2_results[3].frame_locations[1962])
        self.assertIsNotNone(job2_results[3].frame_locations[3752])
        self.assertIsNotNone(job2_results[3].frame_locations[5393])
        self.assertEquals(job2_results[4].stop_frame, 5393)
        self.assertIsNotNone(job2_results[4].frame_locations[5393])
        self.assertEquals(job2_results[5].stop_frame, 5393)
        self.assertIsNotNone(job2_results[5].frame_locations[5393])

if __name__ == "__main__":
    unittest.main(verbosity=2)