#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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

import logging
import os
import csv
from pkg_resources import resource_filename
from itertools import islice
from typing import Iterable, Mapping

from PIL import Image
import cv2
import base64
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import clip

import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException, triton_to_np_dtype

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = logging.getLogger('ClipComponent')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ClipComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin):
    detection_type = 'CLASS'

    def __init__(self):
        self._model_wrappers = {}

    @staticmethod
    def _get_prop(job_properties, key, default_value, accep_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accep_values != []) and (prop not in accep_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptible values: {accep_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

    def _parse_properties(self, job_properties):
        model_name = self._get_prop(job_properties, "MODEL_NAME", "ViT-L/14", ["ViT-L/14", "ViT-B/32"])
        batch_size = self._get_prop(job_properties, "DETECTION_FRAME_BATCH_SIZE", 64)
        batch_size_triton = self._get_prop(job_properties, "DETECTION_FRAME_BATCH_SIZE_TRITON", 32)
        classification_list = self._get_prop(job_properties, "CLASSIFICATION_LIST", 'coco', ['coco', 'imagenet'])
        classification_path = os.path.expandvars(self._get_prop(job_properties, "CLASSIFICATION_PATH", ''))
        enable_cropping = self._get_prop(job_properties, "ENABLE_CROPPING", True)
        enable_triton = self._get_prop(job_properties, "ENABLE_TRITON", False)
        include_features = self._get_prop(job_properties, "INCLUDE_FEATURES", False)
        num_classifications = self._get_prop(job_properties, "NUMBER_OF_CLASSIFICATIONS", 1)
        num_templates = self._get_prop(job_properties, "NUMBER_OF_TEMPLATES", 80, [1, 7, 80])
        template_path = os.path.expandvars(self._get_prop(job_properties, "TEMPLATE_PATH", ''))
        triton_server = self._get_prop(job_properties, "TRITON_SERVER", 'clip-detection-server:8001')

        return dict(
            model_name = model_name,
            batch_size = batch_size,
            batch_size_triton = batch_size_triton,
            classification_list = classification_list,
            classification_path = classification_path,
            enable_cropping = enable_cropping,
            enable_triton = enable_triton,
            include_features = include_features,
            num_classifications = num_classifications,
            num_templates = num_templates,
            template_path = template_path,
            triton_server = triton_server
        )

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info("Received image job: %s", image_job)

        kwargs = self._parse_properties(image_job.job_properties)
        image = image_reader.get_image()

        num_detections = 0
        try:
            wrapper = self._get_model_wrapper(kwargs['model_name'])
            detections = wrapper.get_classifications((image,), **kwargs)
            for detection in detections:
                yield detection
                num_detections += 1
            logger.info(f"Job complete. Found {num_detections} detection{'' if num_detections == 1 else 's'}.")
        
        except Exception as e:
            logger.exception(f'Job failed due to: {e}')
            raise

    @staticmethod
    def _batches_from_video_capture(video_capture, batch_size):
        frames = []
        for frame in video_capture:
            frames.append(frame)
            if len(frames) >= batch_size:
                yield len(frames), np.stack(frames)
                frames = []
        
        if len(frames):
            padded = np.pad(
                array=np.stack(frames),
                pad_width=((0, batch_size - len(frames)), (0, 0), (0, 0), (0, 0)),
                mode='constant',
                constant_values=0
            )
            yield len(frames), padded

    def get_detections_from_video_capture(self,
                                          video_job: mpf.VideoJob,
                                          video_capture: mpf_util.VideoCapture) -> Iterable[mpf.VideoTrack]:
        logger.info("Received video job: %s", video_job)
        kwargs = self._parse_properties(video_job.job_properties)
        if kwargs['enable_triton']:
            batch_size = kwargs['batch_size_triton']
        else:
            batch_size = kwargs['batch_size']
        
        batch_gen = self._batches_from_video_capture(video_capture, batch_size)
        detections = []
        wrapper = self._get_model_wrapper(kwargs['model_name'])

        for n, batch in batch_gen:
            try:
                detection = list(islice(wrapper.get_classifications(batch, **kwargs), n))
                detections += detection
            except Exception as e:
                logger.exception(f"Job failed due to: {e}")
                raise
        
        tracks = create_tracks(detections)
        logger.info(f"Job complete. Found {len(tracks)} detection{'' if len(tracks) == 1 else 's'}.")
        return tracks

    def _get_model_wrapper(self, model_name):
        if model_name not in self._model_wrappers:
            self._model_wrappers[model_name] = ClipWrapper(model_name)

        return self._model_wrappers[model_name]

class ClipWrapper(object):
    def __init__(self, model_name='ViT-L/14'):
        logger.info("Loading model...")
        model, _ = clip.load(model_name, device=device, download_root='/models')
        logger.info("Model loaded.")

        self._model = model
        self._preprocessor = None
        self._input_resolution = self._model.visual.input_resolution

        self._classification_path = ''
        self._template_path = ''
        self._classification_list = ''

        self._templates = None
        self._class_mapping = None
        self._text_features = None

        self._inferencing_server = None
        self._triton_server_url = None
    
    def get_classifications(self, images, **kwargs) -> Iterable[mpf.ImageLocation]:
        templates_changed = self._check_template_list(kwargs['template_path'], kwargs['num_templates'])
        self._check_class_list(kwargs['classification_path'], kwargs['classification_list'], templates_changed)

        self._preprocessor = ImagePreprocessor(kwargs['enable_cropping'], self._input_resolution)
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        image_sizes = [image.size for image in images]
        torch_imgs = torch.stack([self._preprocessor.preprocess(image).squeeze(0) for image in images]).to(device)
        if kwargs['enable_cropping']:
            torch_imgs = torch_imgs.squeeze(0)

        if kwargs['enable_triton']:
            if self._inferencing_server is None or kwargs['triton_server'] != self._triton_server_url:
                self._inferencing_server = CLIPInferencingServer(kwargs['triton_server'])
                self._triton_server_url = kwargs['triton_server']

            results = self._inferencing_server.get_responses(torch_imgs)
            image_features = torch.Tensor(np.copy(results)).squeeze(0).to(device=device)
        else:
            with torch.no_grad():
                image_features = self._model.encode_image(torch_imgs).float()

        with torch.no_grad():
            image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self._text_features).softmax(dim=-1).to(device)

        if kwargs['enable_cropping']:
            similarity = torch.mean(similarity, 0).unsqueeze(0)
        
        values, indices = similarity.topk(len(self._class_mapping))

        for detection_values, detection_indices, image_size in zip(values, indices, image_sizes):
            classification_list = []
            classification_confidence_list = []
            count = 0
            for value, index in zip(detection_values, detection_indices):
                if count >= kwargs['num_classifications']:
                    break
                class_name = self._class_mapping[list(self._class_mapping.keys())[int(index)]]
                if class_name not in classification_list:
                    classification_list.append(class_name)
                    classification_confidence_list.append(str(value.item()))
                    count += 1
            
            classification_list = '; '.join(classification_list)
            classification_confidence_list = '; '.join(classification_confidence_list)
            
            detection_properties = {
                "CLASSIFICATION": classification_list.split('; ')[0],
                "CLASSIFICATION CONFIDENCE LIST": classification_confidence_list,
                "CLASSIFICATION LIST": classification_list
            }
            
            if kwargs['include_features']:
                detection_properties['FEATURE'] = base64.b64encode(image_features.cpu().numpy()).decode()

            yield mpf.ImageLocation(
                x_left_upper = 0,
                y_left_upper = 0,
                width = image_size[0],
                height = image_size[1],
                confidence = float(classification_confidence_list.split('; ')[0]),
                detection_properties = detection_properties
            )

    def _check_template_list(self, template_path: str, number_of_templates: int) -> bool:
        if template_path != '':
            if (not os.path.exists(template_path)):
                raise mpf.DetectionException(
                    f"The path {template_path} is not valid",
                    mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
                )
            elif self._template_path != template_path:
                self._template_path = template_path
            
                try:
                    logger.info("Updating templates...")
                    self._templates = self._get_templates_from_file(template_path)
                    logger.info("Templates updated.")
                    return True
                except:
                    raise mpf.DetectionException(
                        f"Could not read templates from {template_path}",
                        mpf.DetectionError.COULD_NOT_READ_DATAFILE
                    )

        elif self._templates == None or number_of_templates != len(self._templates):
            if number_of_templates == 80:
                template_filename = 'eighty_templates.txt'
            elif number_of_templates == 7:
                template_filename = 'seven_templates.txt'
            elif number_of_templates == 1:
                template_filename = 'one_template.txt'
            
            template_path = os.path.realpath(resource_filename(__name__, 'data/' + template_filename))
            logger.info("Updating templates...")
            self._templates = self._get_templates_from_file(template_path)
            logger.info("Templates updated.")
            return True
        return False

    def _check_class_list(self, classification_path: str, classification_list: str, templates_changed: bool) -> None:
        if classification_path != "":
            if (not os.path.exists(classification_path)):
                raise mpf.DetectionException(
                    f"The path {classification_path} is not valid",
                    mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
                )
        else:
            if self._classification_list != classification_list.lower():
                self._classification_list = classification_list.lower()
            classification_path = os.path.realpath(resource_filename(__name__, f'data/{self._classification_list}_classification_list.csv'))
        
        if self._classification_path != classification_path or templates_changed:
            self._classification_path = classification_path

            try:
                logger.info("Updating classifications...")
                self._class_mapping = self._get_mapping_from_classifications(classification_path)
                logger.info("Classifications updated.")
            except Exception:
                raise mpf.DetectionException(
                    f"Could not read classifications from {classification_path}",
                    mpf.DetectionError.COULD_NOT_READ_DATAFILE
                )

            with torch.no_grad():
                logger.info("Creating text embeddings...")
                text_features = []
                for label in self._class_mapping.keys():
                    text_phrases = [template.format(label) for template in self._templates]
                    text_tokens = clip.tokenize(text_phrases).to(device)
                    text_embeddings = self._model.encode_text(text_tokens)
                    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
                    text_embedding = text_embeddings.mean(dim=0)
                    text_embedding /= text_embedding.norm()
                    text_features.append(text_embedding)
                self._text_features = torch.stack(text_features, dim=1).float().to(device)
                logger.info("Text embeddings created.")
    
    @staticmethod
    def _get_mapping_from_classifications(classification_path: str) -> Mapping[str, str]:
        with open(classification_path) as csvfile:
            mapping = {}
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                mapping[row[0].strip()] = row[1].strip()
                
        return mapping

    @staticmethod
    def _get_templates_from_file(template_path: str) -> Iterable[str]:
        with open(template_path) as f:
            return [line.strip() for line in f.readlines()]

class CLIPInferencingServer(object):
    '''
    Class that handles Triton inferencing if enabled.
    '''
    def __init__(self, triton_server: int):
        self._model_name = 'ip_clip_512'
        self._input_name = None
        self._output_name = None
        self._dtype = None

        try:
            self._triton_client = grpcclient.InferenceServerClient(url=triton_server, verbose=False)
        except InferenceServerException as e:
            logger.exception("Client creation failed.")
            raise
        
        # Check if triton server is alive and ready
        self._check_triton_server()

        try:
            model_metadata = self._triton_client.get_model_metadata(model_name=self._model_name)
        except InferenceServerException as e:
            logger.exception("Failed to retrieve model metadata.")
            raise

        self._parse_model(model_metadata)

    def _parse_model(self, model_metadata) -> None:
        input_metadata = model_metadata.inputs[0]
        output_metadata = model_metadata.outputs[0]
        
        self._input_name = input_metadata.name
        self._output_name = output_metadata.name
        self._dtype = input_metadata.datatype

    def _get_inputs_outputs(self, images):
        inputs = [grpcclient.InferInput(self._input_name, images.shape, self._dtype)]
        inputs[0].set_data_from_numpy(images)

        outputs = [grpcclient.InferRequestedOutput(self._output_name, class_count=0)]

        yield inputs, outputs

    def get_responses(self, images):
        images = np.array(images.cpu())
        images = images.astype(triton_to_np_dtype(self._dtype))
    
        responses = []
        try:
            for inputs, outputs in self._get_inputs_outputs(images):
                responses.append(self._triton_client.infer(model_name=self._model_name, inputs=inputs, outputs=outputs))
        except Exception:
            raise mpf.DetectionException(
                f"Inference failed.",
                mpf.DetectionError.NETWORK_ERROR
            )
        
        results = []
        for response in responses:
            result = response.as_numpy(self._output_name)       
            results.append(result)
        return results   
    
    def _check_triton_server(self) -> None:
        if not self._triton_client.is_server_live():
            raise mpf.DetectionException(
                "Server is not live.",
                mpf.DetectionError.NETWORK_ERROR
            )

        if not self._triton_client.is_server_ready():
            raise mpf.DetectionException(
                "Server is not ready.",
                mpf.DetectionError.NETWORK_ERROR
            )

        if not self._triton_client.is_model_ready(self._model_name):
            raise mpf.DetectionException(
                f"Model {self._model_name} is not ready.",
                mpf.DetectionError.NETWORK_ERROR
            )

class ImagePreprocessor(object):
    '''
    Class that handles the preprocessing of images before being sent through the CLIP model.
    Values from T.Normalize() taken from OpenAI's code for CLIP, https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
    '''
    def __init__(self, enable_cropping: bool, image_size: int):
        self.image_size = image_size
        if enable_cropping:
            self.preprocess = self.crop
        else:
            self.preprocess = self.resize_pad
    
    def crop(self, image):
        return T.Compose([
            self._resize_images,
            self._get_crops,
            T.Lambda(lambda crops: torch.stack([T.ToTensor()(crop) for crop in crops])),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(image)

    def resize_pad(self, image):
        width, height = image.width, image.height
        resize_ratio = self.image_size / max(width, height)
        new_w, new_h = (round(width * resize_ratio), round(height * resize_ratio))

        # if width < self.image_size and height < self.image_size:
        #     padding = ((self.image_size - width) // 2, (self.image_size - height) // 2, (self.image_size + 1 - width) // 2, (self.image_size + 1 - height) // 2)
        #     new_img = T.Compose([
        #         T.Pad(padding=padding),
        #         T.ToTensor(),
        #         T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        #     ])(image)
        # else:
        if new_w < new_h:
            left = (self.image_size - new_w) // 2
            right = (self.image_size + 1 - new_w) // 2
            padding = (left, 0, right, 0)
        else:
            top = (self.image_size - new_h) // 2
            bottom = (self.image_size + 1 - new_h) // 2
            padding = (0, top, 0, bottom)
        
        new_img = T.Compose([
            T.Resize(size=(new_h, new_w)),
            T.Pad(padding=padding, padding_mode='edge'),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(image)

        return new_img.unsqueeze(0)

    @staticmethod
    def _resize_images(img):
        resized_images = []
        for size in [256, 288, 320, 352]:
            resized_image = TF.resize(img, size)
            width, height = resized_image.size
            resized_images.append(TF.center_crop(resized_image, size))
            resized_images.append(TF.crop(resized_image, 0, 0, size, size))
            resized_images.append(TF.crop(resized_image, height-size, width-size, size, size))
        return resized_images

    @staticmethod
    def _get_crops(imgs):
        crops = ()
        for img in imgs:
            five_crops = TF.five_crop(img, 224)
            resized = TF.resize(img, 224)
            crops += five_crops + (resized, TF.hflip(resized)) + tuple([TF.hflip(fcrop) for fcrop in five_crops])
        return crops

def create_tracks(detections: Iterable[mpf.ImageLocation]) -> Iterable[mpf.VideoTrack]:
    tracks = []
    for idx, detection in enumerate(detections):
        if len(tracks) == 0 or tracks[-1].detection_properties["CLASSIFICATION"] != detection.detection_properties["CLASSIFICATION"]:
            detection_properties = { "CLASSIFICATION": detection.detection_properties["CLASSIFICATION"] }
            frame_locations = { idx: detection }
            track = mpf.VideoTrack(
                start_frame=idx,
                stop_frame=idx,
                confidence=detection.confidence,
                frame_locations=frame_locations,
                detection_properties=detection_properties
            )
            tracks.append(track)
        else:
            tracks[-1].stop_frame = idx
            tracks[-1].frame_locations[idx] = detection
            if tracks[-1].confidence < detection.confidence:
                tracks[-1].confidence = detection.confidence

    return tracks

EXPORT_MPF_COMPONENT = ClipComponent
