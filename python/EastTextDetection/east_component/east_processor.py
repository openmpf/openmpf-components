#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2022 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2022 The MITRE Corporation                                      #
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

import numpy as np
import cv2
import os

import mpf_component_api as mpf
from .bbox_utils import *

# The path to the serialized EAST model file.
from pkg_resources import resource_filename
_model_filename = os.path.realpath(resource_filename(__name__, 'east_resnet50.pb'))

# The output layer names for the EAST model. Respectively: the layer
# corresponding to the bounding box geometry, and the layer corresponding to
# the bounding box confidence scores.
_layer_names = ['feature_fusion/concat_3', 'feature_fusion/Conv_7/Sigmoid']

# Mean pixel values used when training the model. These values will be
# subtracted from each pixel of the input image before processing.
_mean_rgb = (123.68, 116.78, 103.94)

class EastProcessor(object):
    def __init__(self, logger):
        self.logger = logger
        self._blob_width = None
        self._blob_height = None
        self._batch_size = None
        self._rotate_on = None
        self._model = None
        self._model_90 = None

    def _set_input_dimensions(self, frame_width, frame_height, batch_size,
                              max_side_len=-1):
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._batch_size = batch_size
        blob_width = frame_width
        blob_height = frame_height

        # Limit the max side
        if max_side_len > 0:
            long_side = max(blob_width, blob_height)
            resized_long_side = min(max_side_len, long_side)
            ratio = float(resized_long_side) / long_side

            blob_width = int(blob_width * ratio)
            blob_height = int(blob_height * ratio)

        # Ensure dimensions are divisible by 32 (required for EAST)
        if blob_width % 32:
            blob_width = (blob_width // 32) * 32
        if blob_height % 32:
            blob_height = (blob_height // 32) * 32

        self._blob_width = max(32, blob_width)
        self._blob_height = max(32, blob_height)

        # Scaling from feature map to blob is 4
        # Get scaling factors to map to frame dimensions
        frame_dims = np.array([self._frame_width, self._frame_height])
        blob_dims = np.array([self._blob_width, self._blob_height])
        self._blob2frame_scale = frame_dims.astype(float) / blob_dims
        self._feat2frame_scale = self._blob2frame_scale * 4.0

    def load_model(self, frame_width, frame_height, max_side_len, rotate_on,
                   batch_size=1):
        old_params = (
            self._blob_width,
            self._blob_height,
            self._batch_size
        )
        self._set_input_dimensions(
            frame_width=frame_width,
            frame_height=frame_height,
            batch_size=batch_size,
            max_side_len=max_side_len
        )
        new_params = (
            self._blob_width,
            self._blob_height,
            self._batch_size
        )
        use_cached = (old_params == new_params)

        if not use_cached:
            self._model = cv2.dnn.readNetFromTensorflow(_model_filename)

        if not use_cached or (rotate_on and not self._rotate_on):
            self._model_90 = cv2.dnn.readNetFromTensorflow(_model_filename)

        self._rotate_on = rotate_on

    def _process_blob(self, blob, min_confidence, suppress_vertical):
        # Run blob through model to get geometry and scores
        self._model.setInput(blob)
        data = np.concatenate(self._model.forward(_layer_names), axis=1)
        if self._rotate_on:
            # Rotate input, run model, rotate output back, interleave
            self._model_90.setInput(np.rot90(blob, k=1, axes=(2,3)))
            data_r = np.concatenate(self._model_90.forward(_layer_names), axis=1)
            data_r = np.rot90(data_r, k=-1, axes=(2,3))
            data = np.repeat(data, 2, axis=0)
            data[1::2,...] = data_r

        # Reshape into [n_frames, feat_h, feat_w, 6]
        data = np.moveaxis(data, 1, -1)

        # Split into AABB, angle, and confidence
        # AABB: [dist2top, dist2right, dist2bottom, dist2left]
        aabb, rotation, scores = data[...,:4], data[...,4], data[...,5]

        structure_score = scores.mean()

        # Take only detections with reasonably high confidence scores
        selected = scores > min_confidence

        # If specified, take only detections whose width is larger than height
        if suppress_vertical:
            vertical = aabb[...,0] + aabb[...,2] > aabb[...,1] + aabb[...,3]
            selected = np.logical_and(selected, np.logical_not(vertical))

        # Get image index and feature map coordinates (x,y) of detections
        origin_coords = np.argwhere(selected)
        batch_idx = origin_coords[:,0]
        origin_coords = origin_coords[:,(2,1)].astype(np.float32)

        # Filter detections based on confidence
        aabb = aabb[selected]
        rotation = rotation[selected]
        scores = scores[selected]
        if self._rotate_on:
            rotated = (batch_idx % 2).astype(np.bool)

        # If we rotated, the frames are interleaved, so batch location is halved
        if self._rotate_on:
            batch_idx = (batch_idx / 2).astype(int)

        # Get sine and cosine of box rotation
        c = np.cos(rotation)[:,None]
        s = np.sin(rotation)[:,None]

        # Rescale AABB values to frame dimensions
        w = aabb[:,(1,3)]
        h = aabb[:,(0,2)]
        wx = self._blob2frame_scale[0] * c * w
        wy = self._blob2frame_scale[1] * s * w
        hx = self._blob2frame_scale[0] * s * h
        hy = self._blob2frame_scale[1] * c * h
        if self._rotate_on:
            inv = self._blob2frame_scale[1] / self._blob2frame_scale[0]
            wx[rotated,:] *= inv
            wy[rotated,:] /= inv
            hx[rotated,:] *= inv
            hy[rotated,:] /= inv
        aabb[:,(1,3)] = np.sqrt(wx ** 2.0 + wy ** 2.0)
        aabb[:,(0,2)] = np.sqrt(hx ** 2.0 + hy ** 2.0)

        # Correct box angles for rescale
        rotation = np.arctan2(wy.sum(axis=-1),wx.sum(axis=-1))
        if self._rotate_on:
            rotation[rotated] -= np.pi / 2

        # Ensure rotation is in the range [-pi, pi]
        rotation = (rotation + np.pi) % (2 * np.pi) - np.pi

        # Rescale origin coordinates to frame dimensions
        origin_coords *= self._feat2frame_scale

        rboxes = np.hstack((
            origin_coords,
            aabb,
            rotation[:,None]
        ))

        return batch_idx, rboxes, scores, structure_score

    def process_image(self, image, temp_padding_x, temp_padding_y,
                      final_padding, merge_on, suppress_vertical,
                      min_confidence, overlap_threshold, min_nms_overlap,
                      max_height_delta, max_rot_delta, min_structured_score,
                      **kwargs):
        """ Process a single image using the given arguments
        :param image: The image to be processed. Takes the following shape:
                (frame_height, frame_width, num_channels)
        :param temp_padding_x: Temporary padding to symmetrically add to the
                width of bounding boxes during non-maximum suppression or
                merging. The right and left side of bounding boxes are both
                extended by temp_padding_x * box_height.
        :param temp_padding_y: Temporary padding to symmetrically add to the
                height of bounding boxes during non-maximum suppression or
                merging. The top and bottom of the bounding boxes are both
                extended by temp_padding_y * box_height.
        :param final_padding: Padding to symmetrically add to output bounding
                boxes after non-maximum suppression or merging. Each side is
                extended by final_padding * box_height.
        :param merge_on: Whether to merge regions according to the provided
                thresholds.
        :param suppress_vertical: Whether to suppress vertical detections.
        :param min_confidence: Threshold to use for filtering detections
                by bounding box confidence.
        :param overlap_threshold: Threshold value for deciding whether
                regions overlap enough to be merged.
        :param min_nms_overlap: Threshold value for non-maximum suppression.
        :param max_height_delta: Threshold value for deciding whether
                regions containing different text sizes should be merged.
        :param max_rot_delta: Threshold value for deciding whether regions
                containing text at different orientations should be merged.
        :param min_structured_score: Threshold value for deciding whether
                detected text should be considered structured or unstructured.
        :return: List of mpf.ImageLocation detections. The angles of detected
                bounding boxes are stored in detection_properties['ROTATION'].
                The rotation is expressed in positive degrees counterclockwise
                from the 3 o'clock position.
        """
        # Convert the image to OpenCV-compatible blob, pass to processor
        batch_idx, rboxes, scores, structure_score = self._process_blob(
            blob=cv2.dnn.blobFromImage(
                image=image,
                size=(self._blob_width, self._blob_height),
                mean=_mean_rgb,
                swapRB=True,
                crop=False
            ),
            min_confidence=min_confidence,
            suppress_vertical=suppress_vertical
        )

        text_type = 'UNSTRUCTURED'
        if structure_score >= min_structured_score:
            text_type = 'STRUCTURED'

        if not len(rboxes):
            return []

        if merge_on:
            ilocs = merge_regions(
                rboxes=rboxes,
                scores=scores,
                temp_padding_x=temp_padding_x,
                temp_padding_y=temp_padding_y,
                final_padding=final_padding,
                overlap_threshold=overlap_threshold,
                max_height_delta=max_height_delta,
                max_rot_delta=max_rot_delta
            )
        else:
            ilocs = nms(
                rboxes=rboxes,
                scores=scores,
                temp_padding_x=temp_padding_x,
                temp_padding_y=temp_padding_y,
                final_padding=final_padding,
                min_nms_overlap=min_nms_overlap
            )

        return [
            mpf.ImageLocation(
                x_left_upper=int(x),
                y_left_upper=int(y),
                width=int(w),
                height=int(h),
                confidence=float(s),
                detection_properties=dict(
                    ROTATION=str(r),
                    TEXT_TYPE=text_type
                )
            )
            for x, y, w, h, r, s in ilocs
        ]

    def process_frames(self, frames, temp_padding_x, temp_padding_y,
                       final_padding, merge_on, suppress_vertical,
                       min_confidence, overlap_threshold, min_nms_overlap,
                       max_height_delta, max_rot_delta, min_structured_score,
                       **kwargs):
        """ Process a volume of images using the given arguments
        :param frames: The images to be processed. Takes the following shape:
                (batch_size, frame_height, frame_width, num_channels)
        :param temp_padding_x: Temporary padding to symmetrically add to the
                width of bounding boxes during non-maximum suppression or
                merging. The right and left side of bounding boxes are both
                extended by temp_padding_x * box_height.
        :param temp_padding_y: Temporary padding to symmetrically add to the
                height of bounding boxes during non-maximum suppression or
                merging. The top and bottom of the bounding boxes are both
                extended by temp_padding_y * box_height.
        :param final_padding: Padding to symmetrically add to output bounding
                boxes after non-maximum suppression or merging. Each side is
                extended by final_padding * box_height.
        :param merge_on: Whether to merge regions according to the provided
                thresholds.
        :param suppress_vertical: Whether to suppress vertical detections.
        :param min_confidence: Threshold to use for filtering detections
                by bounding box confidence.
        :param overlap_threshold: Threshold value for deciding whether
                regions overlap enough to be merged.
        :param min_nms_overlap: Threshold value for non-maximum suppression.
        :param max_height_delta: Threshold value for deciding whether
                regions containing different text sizes should be merged.
        :param max_rot_delta: Threshold value for deciding whether regions
                containing text at different orientations should be merged.
        :param min_structured_score: Threshold value for deciding whether
                detected text should be considered structured or unstructured.
        :return: List lists of mpf.ImageLocation detections. Each list
                corresponds with one frame of the input volume. The angles of
                detected bounding boxes are stored in
                detection_properties['ROTATION']. The rotation is expressed in
                positive degrees counterclockwise from the 3 o'clock position.
        """
        # Convert to compatible shape
        batch_size = frames.shape[0]

        # Convert the images to OpenCV-compatible blob, pass to processor
        batch_idx, rboxes, scores, structure_score = self._process_blob(
            blob=cv2.dnn.blobFromImages(
                images=frames,
                size=(self._blob_width, self._blob_height),
                mean=_mean_rgb,
                swapRB=True,
                crop=False
            ),
            min_confidence=min_confidence,
            suppress_vertical=suppress_vertical
        )

        text_type = 'UNSTRUCTURED'
        if structure_score >= min_structured_score:
            text_type = 'STRUCTURED'

        # Split by frame so that boxes in different frames don't interfere
        # with one another during non-max suppression
        split_points = np.cumsum(np.bincount(batch_idx, minlength=batch_size))
        split_points = [0] + split_points.tolist()

        # Apply non-max suppression to the detections in each frame
        image_locs = []
        for i in range(len(split_points)-1):
            j0, j1 = split_points[i], split_points[i+1]
            if j1 > j0:
                if merge_on:
                    ilocs = merge_regions(
                        rboxes=rboxes[j0:j1],
                        scores=scores[j0:j1],
                        temp_padding_x=temp_padding_x,
                        temp_padding_y=temp_padding_y,
                        final_padding=final_padding,
                        overlap_threshold=overlap_threshold,
                        max_height_delta=max_height_delta,
                        max_rot_delta=max_rot_delta
                    )
                else:
                    ilocs = nms(
                        rboxes=rboxes[j0:j1],
                        scores=scores[j0:j1],
                        temp_padding_x=temp_padding_x,
                        temp_padding_y=temp_padding_y,
                        final_padding=final_padding,
                        min_nms_overlap=min_nms_overlap
                    )

                image_locs.append([
                    mpf.ImageLocation(
                        x_left_upper=int(x),
                        y_left_upper=int(y),
                        width=int(w),
                        height=int(h),
                        confidence=float(s),
                        detection_properties=dict(
                            ROTATION=str(r),
                            TEXT_TYPE=text_type
                        )
                    )
                    for x, y, w, h, r, s in ilocs
                ])
            else:
                image_locs.append([])

        return image_locs
