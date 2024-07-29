import cv2
import base64
import ollama

import logging

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('LlavaComponent')

class LlavaComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin):
    """
    Takes in feed forward tracks from YOLO and asks questions depending on the class of the detection.
    { "PERSON" } --> 
        { "CLOTHING" : "Describe what this person is wearing",
          "ACTIVITY" : "Describe what this person is doing" }

    { "CAR, TRUCK, BUS" } --> 
        { "DESCRIPTION" : "Describe the type, color, and make of this vehicle" }
    """
    detection_type = 'CLASS'

    def __init__(self):
        self.model = 'llava:34b'
        self.class_questions = {
            'person': { "CLOTHING": "Describe what this person is wearing.", "ACTIVITY": "Describe what this person is doing."},
            'car': {"DESCRIPTION": "Describe the type, color, and make of this vehicle."},
            'truck': {"DESCRIPTION": "Describe the type, color, and make of this vehicle."},
            'bus': {"DESCRIPTION": "Describe the type, color, and make of this vehicle."},
            'dog': {"DESCRIPTION": "Describe the color and breed of the dog."}
        }

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info('[%s] Received image job: ', image_job.job_name, image_job)

        return self._get_feed_forward_detections(image_job, image_job.feed_forward_location, image_reader.get_image())

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)

        return self._get_feed_forward_detections(video_job, video_job.feed_forward_track, video_capture, video_job=True)

    def _get_feed_forward_detections(self, job, job_feed_forward, media, video_job=False):
        try:
            if job_feed_forward is None: 
                raise mpf.DetectionException(
                    "Component can only process feed forward jobs, but no feed forward track provided.",
                    mpf.DetectionError.UNSUPPORTED_DATA_TYPE
                )
            
            classification = job_feed_forward.detection_properties["CLASSIFICATION"]
            if classification in self.class_questions:
                for tag, prompt in self.class_questions[classification].items():
                    job_feed_forward.detection_properties[tag] = self._generate_ollama_response(media, prompt)

            yield job_feed_forward

        except Exception as e:
            logger.exception(f"Failed to complete job due to the following exception: {e}")
        
    def _generate_ollama_response(self, image, prompt):
        encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
        _, buffer = cv2.imencode('.png', image, encode_params)
        b64_bytearr = base64.b64encode(buffer).decode("utf-8")
        return ollama.generate(self.model, prompt, images=[b64_bytearr])['response']
  
EXPORT_MPF_COMPONENT = LlavaComponent
