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
import sys
from pkg_resources import resource_filename
from itertools import islice
from typing import Iterable, Mapping
import argparse

from PIL import Image
import cv2
import base64
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import clip
from CoOp.train import get_trainer

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
    def _get_prop(job_properties, key, default_value, accept_values=[]):
        prop = mpf_util.get_property(job_properties, key, default_value)
        if (accept_values != []) and (prop not in accept_values):
            raise mpf.DetectionException(
                f"Property {key} not in list of acceptible values: {accept_values}",
                mpf.DetectionError.INVALID_PROPERTY
            )
        return prop

    def _parse_properties(self, job_properties):
        model_name = self._get_prop(job_properties, "MODEL_NAME", "ViT-L/14", ["ViT-L/14", "ViT-B/32", "CoOp"])
        batch_size = self._get_prop(job_properties, "DETECTION_FRAME_BATCH_SIZE", 64)
        classification_list = self._get_prop(job_properties, "CLASSIFICATION_LIST", 'coco', ['coco', 'imagenet'])
        classification_path = os.path.expandvars(self._get_prop(job_properties, "CLASSIFICATION_PATH", ''))
        enable_cropping = self._get_prop(job_properties, "ENABLE_CROPPING", True)
        enable_triton = self._get_prop(job_properties, "ENABLE_TRITON", False)
        include_features = self._get_prop(job_properties, "INCLUDE_FEATURES", False)
        num_classifications = self._get_prop(job_properties, "NUMBER_OF_CLASSIFICATIONS", 1)
        template_type = self._get_prop(job_properties, "TEMPLATE_TYPE", 'openai_80', ['openai_1', 'openai_7', 'openai_80'])
        template_path = os.path.expandvars(self._get_prop(job_properties, "TEMPLATE_PATH", ''))
        triton_server = self._get_prop(job_properties, "TRITON_SERVER", 'clip-detection-server:8001')

        return dict(
            model_name = model_name,
            batch_size = batch_size,
            classification_list = classification_list,
            classification_path = classification_path,
            enable_cropping = enable_cropping,
            enable_triton = enable_triton,
            include_features = include_features,
            num_classifications = num_classifications,
            template_type = template_type,
            template_path = template_path,
            triton_server = triton_server
        )

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info("Received image job: %s", image_job)

        kwargs = self._parse_properties(image_job.job_properties)
        image = image_reader.get_image()

        num_detections = 0
        try:
            wrapper = self._get_model_wrapper(model_name=kwargs['model_name'], kwargs=kwargs)
            detections = wrapper.get_detections((image,), **kwargs)
            for detection in detections:
                yield detection
                num_detections += 1
            logger.info(f"Job complete. Found {num_detections} detections.")
        
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
        
        # If processing a video where each frame is cropped into 144 images, the batch size is set to one so that the crops aren't split between batches
        batch_size = 1 if kwargs['enable_cropping'] else kwargs['batch_size']
        
        batch_gen = self._batches_from_video_capture(video_capture, batch_size)
        detections = []
        wrapper = self._get_model_wrapper(model_name=kwargs['model_name'], kwargs=kwargs)

        for n, batch in batch_gen:
            try:
                detections += list(islice(wrapper.get_detections(batch, **kwargs), n))
            except Exception as e:
                logger.exception(f"Job failed due to: {e}")
                raise
        
        tracks = create_tracks(detections)
        logger.info(f"Job complete. Found {len(tracks)} tracks.")
        return tracks

    def _get_model_wrapper(self, model_name, kwargs):   
        if model_name not in self._model_wrappers:
            if model_name == "CoOp":
                self._model_wrappers['CoOp'] = CoOpWrapper(**kwargs)
            else:
                self._model_wrappers[model_name] = ClipWrapper(model_name)

        return self._model_wrappers[model_name]

class CoOpWrapper(object):
    def __init__(self, **kwargs):
        if (kwargs['classification_list'] == 'coco') or (kwargs['template_path'] != '') or (kwargs['classification_path'] != '') or (kwargs['enable_triton'] == True):
            raise mpf.DetectionException(
                f"Properties incompatible with CoOp. Make sure that CLASSIFICATION_LIST='imagenet', TEMPLATE_PATH='', CLASSIFICATION_PATH='', and ENABLE_TRITON=False.",
                mpf.DetectionError.INVALID_PROPERTY
            )
        self._manual_args = ["--seed", "1", "--trainer", "CoOp", "--config-file", "/opt/coop_src/CoOp/configs/trainers/CoOp/vit_l14_ep50.yaml", "--model-dir", "/opt/coop_src/CoOp/output/imagenet/CoOp/vit_l14_ep50_16shots/nctx16_cscFalse_ctpend/seed1", "--load-epoch", "50", "--eval-only", "TRAINER.COOP.N_CTX", "16", "TRAINER.COOP.CSC", "False", "TRAINER.COOP.CLASS_TOKEN_POSITION", "end"]
        self.args = self._create_arg_parser(self._manual_args)
        self._class_mapping = self._get_mapping_from_classifications(os.path.realpath(resource_filename(__name__, f'data/imagenet_classification_list.csv')))
        self.classnames = self._class_mapping.keys()
        # Create trainer object
        print("Creating trainer...")
        self.trainer = get_trainer(self.args, self.classnames)
        print("Trainer created.")

    def get_detections(self, images, **kwargs):
        # Preprocess image
        self._preprocessor = ImagePreprocessor(enable_cropping=False, image_size=224)
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        image_sizes = [image.size for image in images]
        torch_imgs = torch.stack([self._preprocessor.preprocess(image).squeeze(0) for image in images]).to(device)

        # Load model
        self.trainer.load_model(self.args.model_dir, epoch = self.args.load_epoch)
        
        # Pass image through model
        output, image_features = self.trainer.test(images=torch_imgs)

        softmax = torch.nn.Softmax(dim=1)(output)
        values, indices = softmax.topk(kwargs['num_classifications'])

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
    
    def _create_arg_parser(self, manual_args):
        parser = argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default="", help="path to dataset")
        parser.add_argument("--output-dir", type=str, default="", help="output directory")
        parser.add_argument(
            "--resume",
            type=str,
            default="",
            help="checkpoint directory (from which the training resumes)",
        )
        parser.add_argument(
            "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
        )
        parser.add_argument(
            "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
        )
        parser.add_argument(
            "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
        )
        parser.add_argument(
            "--transforms", type=str, nargs="+", help="data augmentation methods"
        )
        parser.add_argument(
            "--config-file", type=str, default="", help="path to config file"
        )
        parser.add_argument(
            "--dataset-config-file",
            type=str,
            default="",
            help="path to config file for dataset setup",
        )
        parser.add_argument("--trainer", type=str, default="", help="name of trainer")
        parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
        parser.add_argument("--head", type=str, default="", help="name of head")
        parser.add_argument("--eval-only", action="store_true", help="evaluation only")
        parser.add_argument(
            "--model-dir",
            type=str,
            default="",
            help="load model from this directory for eval-only mode",
        )
        parser.add_argument(
            "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
        )
        parser.add_argument(
            "--no-train", action="store_true", help="do not call trainer.train()"
        )
        parser.add_argument(
            "opts",
            default=None,
            nargs=argparse.REMAINDER,
            help="modify config options using the command-line",
        )
        args = parser.parse_args(manual_args)
        return args
    
    @staticmethod
    def _get_mapping_from_classifications(classification_path: str) -> Mapping[str, str]:
        with open(classification_path) as csvfile:
            mapping = {}
            csvreader = csv.reader(csvfile)
            for row in csvreader:
                mapping[row[0].strip()] = row[1].strip()
                
        return mapping
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
        self._template_type = None
        self._class_mapping = None
        self._text_features = None

        self._inferencing_server = None

    def get_detections(self, images, **kwargs) -> Iterable[mpf.ImageLocation]:
        templates_changed = self._check_template_list(kwargs['template_path'], kwargs['template_type'])
        self._check_class_list(kwargs['classification_path'], kwargs['classification_list'], templates_changed)

        self._preprocessor = ImagePreprocessor(kwargs['enable_cropping'], self._input_resolution)
        images = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for image in images]
        image_sizes = [image.size for image in images]
        torch_imgs = torch.stack([self._preprocessor.preprocess(image).squeeze(0) for image in images]).to(device)
        if kwargs['enable_cropping']:
            torch_imgs = torch_imgs.squeeze(0)

        if kwargs['enable_triton']:
            if self._inferencing_server is None or \
                    kwargs['triton_server'] != self._inferencing_server.get_url() or \
                    kwargs['model_name'] != self._inferencing_server.get_model_name():
                self._inferencing_server = CLIPInferencingServer(kwargs['triton_server'], kwargs['model_name'])

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

    def _check_template_list(self, template_path: str, template_type: str) -> bool:
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
        elif (self._templates == None) or (template_type != self._template_type):
            if template_type == 'openai_80':
                template_filename = 'eighty_templates.txt'
            elif template_type == 'openai_7':
                template_filename = 'seven_templates.txt'
            elif template_type == 'openai_1':
                template_filename = 'one_template.txt'
            
            template_path = os.path.realpath(resource_filename(__name__, 'data/' + template_filename))
            logger.info("Updating templates...")
            self._templates = self._get_templates_from_file(template_path)
            self._template_type = template_type
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
                self._class_mapping = self._get_mapping_from_classifications(self._classification_path)
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

    MODEL_NAME_ON_SERVER_MAPPING = {'ViT-L/14': 'vit_l_14', 'ViT-B/32': 'vit_b_32'}

    def __init__(self, triton_server: str, model_name: str = 'ViT-L/14'):
        self._url = triton_server
        self._model_name = model_name
        self._model_name_on_server = self.MODEL_NAME_ON_SERVER_MAPPING[model_name]
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
            model_metadata = self._triton_client.get_model_metadata(model_name=self._model_name_on_server)
        except InferenceServerException as e:
            raise mpf.DetectionException(
                f"Failed to retrieve model metadata for {self._model_name_on_server}: {e}",
                mpf.DetectionError.NETWORK_ERROR
            )

        self._parse_model(model_metadata)

    def get_url(self) -> str:
        return self._url

    def get_model_name(self) -> str:
        return self._model_name

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
                responses.append(self._triton_client.infer(model_name=self._model_name_on_server, inputs=inputs, outputs=outputs))
        except Exception as e:
            raise mpf.DetectionException(
                f"Inference failed: {e}",
                mpf.DetectionError.NETWORK_ERROR
            )
        
        results = []
        for response in responses:
            result = response.as_numpy(self._output_name)       
            results.append(result)
        return results   
    
    def _check_triton_server(self) -> None:
        try:
            if not self._triton_client.is_server_live():
                raise mpf.DetectionException(
                    "Server is not live.",
                    mpf.DetectionError.NETWORK_ERROR
            )
        except InferenceServerException as e:
            raise mpf.DetectionException(
                f"Failed to check if server is live: {e}",
                mpf.DetectionError.NETWORK_ERROR
            )

        if not self._triton_client.is_server_ready():
            raise mpf.DetectionException(
                "Server is not ready.",
                mpf.DetectionError.NETWORK_ERROR
            )

        if not self._triton_client.is_model_ready(self._model_name_on_server):
            raise mpf.DetectionException(
                f"Model {self._model_name_on_server} is not ready.",
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
