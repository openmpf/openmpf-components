import os
import cv2
import base64
import json
from typing import Mapping, Sequence
import ollama

import logging

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('LlavaComponent')

class LlavaComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin):
    """
    Idea:
    Map each class to rollup classes, and then map the rollup classes to the question prompts (to account for multiple classes having the same questions).
    """
    detection_type = 'CLASS'

    def __init__(self):
        self.model = 'llava:34b'
        self.class_questions = dict()

    def get_detections_from_image_reader(self, image_job: mpf.ImageJob, image_reader: mpf_util.ImageReader) -> Sequence[mpf.ImageLocation]:
        logger.info('[%s] Received image job: ', image_job.job_name, image_job)

        return self._get_feed_forward_detections(image_job, image_job.feed_forward_location, image_reader.get_image())

    def get_detections_from_video_capture(self, video_job: mpf.VideoJob, video_capture: mpf_util.VideoCapture) -> Sequence[mpf.VideoTrack]:
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)

        return self._get_feed_forward_detections(video_job, video_job.feed_forward_track, video_capture, video_job=True)

    def _get_feed_forward_detections(self, job, job_feed_forward, media, video_job=False):
        try:
            if job_feed_forward is None: 
                raise mpf.DetectionException(
                    "Component can only process feed forward jobs, but no feed forward track provided.",
                    mpf.DetectionError.UNSUPPORTED_DATA_TYPE
                )
            
            # Get job properties
            kwargs = JobConfig(job.job_properties)
            self._update_class_questions(kwargs.config_json_path)
            
            # Encode image for ollama
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            _, buffer = cv2.imencode('.png', media, encode_params)
            b64_bytearr = base64.b64encode(buffer).decode("utf-8")

            # If classification has questions that we want to ask, send queries to ollama model
            classification = job_feed_forward.detection_properties["CLASSIFICATION"]
            if classification in self.class_questions:
                for tag, prompt in self.class_questions[classification].items():
                    job_feed_forward.detection_properties[tag] = ollama.generate(self.model, prompt, images=[b64_bytearr])['response']

            yield job_feed_forward

        except Exception as e:
            logger.exception(f"Failed to complete job due to the following exception: {e}")
    
    def _update_class_questions(self, config_json_path):
        '''
        Updates self.class_questions dictionary to have the following format \n 
        {
            CLASS1: {TAG1: PROMPT1},
            CLASS2: {TAG2: PROMPT2, TAG3: PROMPT3},
            ...
        }
        '''
        try:
            with open(config_json_path, 'r') as f:
                data = json.load(f)
                for class_dict in data:
                    classes, prompts = class_dict['classes'], class_dict['prompts']
                    for cls in classes:
                        if cls not in self.class_questions:
                            self.class_questions[cls] = dict()
                        self.class_questions[cls].update({ dct['detectionProperty']:dct['prompt'] for dct in prompts })
        except:
            raise mpf.DetectionException(
                "Invalid JSON structure for component: ",
                mpf.DetectionError.COULD_NOT_READ_DATAFILE
            )
class JobConfig:
    def __init__(self, job_properties: Mapping[str, str]):
        self.config_json_path = self._get_prop(job_properties, "CONFIG_JSON_PATH", "")
        if self.config_json_path == "":
            self.config_json_path = os.path.join(os.path.dirname(__file__), 'data', 'config.json')
        
        if not os.path.exists(self.config_json_path):
            raise mpf.DetectionException(
                "Invalid path provided for config JSON file: ",
                mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
            )


    @staticmethod
    def _get_prop(job_properties, key, default_value, accept_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accept_values != []) and (prop not in accept_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptible values: {accept_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

EXPORT_MPF_COMPONENT = LlavaComponent
