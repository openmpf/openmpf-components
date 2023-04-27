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

from PIL import Image
import cv2
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

class ClipComponent(mpf_util.ImageReaderMixin):
    detection_type = 'CLASS'

    def get_detections_from_image_reader(self, image_job, image_reader):
        try:
            logger.info("received image job: %s", image_job)

            image = image_reader.get_image()
            detections = JobRunner().get_classifications(image, image_job.job_properties)
            logger.info(f"Job complete. Found {len(detections)} detections")
            return detections
        
        except Exception:
            logger.exception(f"Failed to complete job {image_job.job_name} due to the following exception:")
            raise

class JobRunner(object):
    def __init__(self):
        logger.info("Loading model...")
        model, _ = clip.load('ViT-B/32', device=device)
        logger.info("Model loaded.")
        self._model = model

        self._classification_path = ''
        self._template_path = ''
        self._classification_list = ''

        self._templates = None
        self._class_mapping = None
        self._text_features = None
    
    def get_classifications(self, image, job_properties):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        kwargs = self._parse_properties(job_properties)
        image_width, image_height = image.size

        self._check_template_list(kwargs['template_path'], kwargs['num_templates'])
        self._check_class_list(kwargs['classification_path'], kwargs['classification_list'])

        image = ImagePreprocessor(kwargs['enable_cropping'], kwargs['enable_triton']).preprocess(image).to(device)

        if kwargs['enable_triton']:
            clip_inference = CLIPInferencingServer(kwargs['triton_server'], image)

            # Check if server and model are ready
            clip_inference.check_triton_server()

            results = clip_inference.get_responses()
            image_tensors= torch.Tensor(np.copy(results)).to(device=device)
            image_features = torch.mean(image_tensors, 0)
        else:
            with torch.no_grad():
                image_features = self._model.encode_image(image).float()
            image_features = torch.mean(image_features, 0).unsqueeze(0)
        
        with torch.no_grad():
            image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ self._text_features).softmax(dim=-1).to(device)
        similarity = torch.mean(similarity, 0)
        values, indices = similarity.topk(kwargs['num_classifications'])

        classification_list = '; '.join([self._class_mapping[list(self._class_mapping.keys())[int(index)]] for index in indices])
        classification_confidence_list = '; '.join([str(value.item()) for value in values])
        
        return [
            mpf.ImageLocation(
                x_left_upper = 0,
                y_left_upper = 0,
                width = image_width,
                height = image_height,
                confidence = float(classification_confidence_list.split('; ')[0]),
                detection_properties = {
                    "CLASSIFICATION": classification_list.split('; ')[0],
                    "CLASSIFICATION CONFIDENCE LIST": classification_confidence_list,
                    "CLASSIFICATION LIST": classification_list
                }
            )
        ]

    def _parse_properties(self, job_properties):
        num_classifications = self._get_prop(job_properties, "NUMBER_OF_CLASSIFICATIONS", 1)
        num_templates = self._get_prop(job_properties, "NUMBER_OF_TEMPLATES", 80, [1, 7, 80])
        classification_list = self._get_prop(job_properties, "CLASSIFICATION_LIST", 'coco', ['coco', 'imagenet'])
        classification_path = self._get_prop(job_properties, "CLASSIFICATION_PATH", '')
        template_path = self._get_prop(job_properties, "TEMPLATE_PATH", '')
        enable_cropping = self._get_prop(job_properties, "ENABLE_CROPPING", True)
        enable_triton = self._get_prop(job_properties, "ENABLE_TRITON", False)
        triton_server = self._get_prop(job_properties, "TRITON_SERVER", 'clip-detection-server:8001')

        return dict(
            num_classifications = num_classifications,
            num_templates = num_templates,
            classification_list = classification_list,
            classification_path = classification_path,
            template_path = template_path,
            enable_cropping = enable_cropping,
            enable_triton = enable_triton,
            triton_server = triton_server
        )
    
    @staticmethod
    def _get_prop(job_properties, key, default_value, accep_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accep_values != []) and (prop not in accep_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptible values: {accep_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

    def _check_template_list(self, template_path, number_of_templates):
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

    def _check_class_list(self, classification_path, classification_list):
        if classification_path != "":
            if (not os.path.exists(classification_path)):
                raise mpf.DetectionException(
                    f"The path {classification_path} is not valid",
                    mpf.DetectionError.COULD_NOT_OPEN_DATAFILE
                )
        elif self._classification_list != classification_list.lower():
            self._classification_list = classification_list.lower()
            classification_path = os.path.realpath(resource_filename(__name__, f'data/{self._classification_list}_classification_list.csv'))
        
        if self._classification_path != classification_path:
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
    def _get_mapping_from_classifications(classification_path):
        with open(classification_path) as csvfile:
            mapping = {}
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                mapping[row[0]] = row[1]
                
        return mapping

    @staticmethod
    def _get_templates_from_file(template_path):
        with open(template_path) as f:
            return [line.strip() for line in f.readlines()]

class CLIPInferencingServer(object):
    '''
    Class that handles Triton inferencing if enabled.
    '''
    def __init__(self, triton_server, images):

        self._images = np.array(images.cpu())
        self._model_name = 'ip_clip_512'

        self._max_batch_size = 2048
        self._input_name = "image_input"
        self._output_name = "feature_vector__0"
        self._input_format = None
        self._dtype = "FP32"

        self._images = self._images.astype(triton_to_np_dtype(self._dtype))

        try:
            self._triton_client = grpcclient.InferenceServerClient(url=triton_server, verbose=False)
        except InferenceServerException as e:
            logger.exception("Client creation failed.")
            raise

    def _get_inputs_outputs(self):
        inputs = [grpcclient.InferInput(self._input_name, self._images.shape, self._dtype)]
        inputs[0].set_data_from_numpy(self._images)

        outputs = [grpcclient.InferRequestedOutput(self._output_name, class_count=0)]

        yield inputs, outputs

    def get_responses(self):
        responses = []
        results = []

        try:
            for inputs, outputs in self._get_inputs_outputs():
                responses.append(self._triton_client.infer(model_name=self._model_name, inputs=inputs, outputs=outputs))
        except Exception:
            raise mpf.DetectionException(
                f"Inference failed.",
                mpf.DetectionError.NETWORK_ERROR
            )
        
        for response in responses:
            result = response.as_numpy(self._output_name)       
            results.append(result)
        return results   
    
    def check_triton_server(self):
        if not self._triton_client.is_server_live():
            raise mpf.DetectionException(
                    "Server is not live.",
                    mpf.DetectionError.NETWORK_ERROR
                )
        elif not self._triton_client.is_server_ready():
            raise mpf.DetectionException(
                    "Server is not ready.",
                    mpf.DetectionError.NETWORK_ERROR
                )
        elif not self._triton_client.is_model_ready(self._model_name):
            raise mpf.DetectionException(
                    f"Model {self._model_name} is not ready.",
                    mpf.DetectionError.NETWORK_ERROR
                )

class ImagePreprocessor(object):
    '''
    Class that handles the preprocessing of images before being sent through the CLIP model.
    Values from T.Normalize() taken from OpenAI's code for CLIP, https://github.com/openai/CLIP/blob/main/clip/clip.py#L85
    '''
    def __init__(self, enable_cropping, enable_triton):
        if enable_cropping:
            self.preprocess = self.crop
        elif enable_triton:
            self.preprocess = lambda image: TF.to_tensor(image).unsqueeze(0)
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
        resize_ratio = 224 / max(width, height)
        new_w, new_h = (int(width * resize_ratio), int(height * resize_ratio))

        if new_w < new_h:
            left = (224 - new_w) // 2
            right = (225 - new_w) // 2
            padding = (left, 0, right, 0)
        else:
            top = (224 - new_h) // 2
            bottom = (225 - new_h) // 2
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
    
    
EXPORT_MPF_COMPONENT = ClipComponent