#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2019 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2019 The MITRE Corporation                                      #
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
from __future__ import division, print_function

import numpy as np
import cv2
import os

import mpf_component_api as mpf
from bbox_utils import *

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

class EASTProcessor(object):
    def __init__(self, logger):
        self.logger = logger
        self._blob_width = None
        self._blob_height = None
        self._rotate_on = None
        self._model = None
        self._model_90 = None

    @staticmethod
    def _get_blob_dimensions(frame_width, frame_height, max_side_len):
        blob_width = frame_width
        blob_height = frame_height

        # limit the max side
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

        blob_width = max(32, blob_width)
        blob_height = max(32, blob_height)

        return (blob_width, blob_height)

    def _load_model(self, blob_width, blob_height, rotate_on):
        # Check if cached model can be used (this saves time for video)
        if self._model is None:
            pass
        elif self._blob_width != blob_width or self._blob_height != blob_height:
            self.logger.info("Cached model incompatible with media properties.")
        elif self._model_90 is None and rotate_on:
            self.logger.info("Cached model can't handle rotated inputs.")
        else:
            self.logger.info("Using cached model.")
            return

        self._model = None
        self._model_90 = None
        self.logger.info("Loading model: " + _model_filename)
        try:
            self._model = cv2.dnn.readNet(_model_filename)
            if rotate_on:
                self._model_90 = cv2.dnn.readNet(_model_filename)
        except Exception as e:
            error_str = "Exception occurred while loading {:s}: {:s}".format(
                _model_filename,
                str(e)
            )
            self.logger.error(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.INVALID_PROPERTY)

        self._blob_width = blob_width
        self._blob_height = blob_height
        self._rotate_on = rotate_on

    def _process_blob(self, blob, frame_width, frame_height,
                      confidence_thresh, padding):
        blob_width = self._blob_width
        blob_height = self._blob_height

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
        feat_height = data.shape[2]
        feat_width = data.shape[3]

        # Reshape into [n_frames, feat_h, feat_w, 6]
        data = np.moveaxis(data, 1, -1)

        # Get scaling factor from feature map to blob (should be (4.0,4.0))
        # and from blob to image
        frame_dims = np.array([frame_width, frame_height], dtype=np.float32)
        blob_dims = np.array([blob_width, blob_height], dtype=np.float32)
        feat_dims = np.array([feat_width, feat_height], dtype=np.float32)
        feat2blob_scale = blob_dims / feat_dims
        blob2frame_scale = frame_dims / blob_dims

        # Split into AABB, angle, and confidence
        # AABB: [dist2top, dist2right, dist2bottom, dist2left]
        aabb, theta, scores = data[...,:4], data[...,4], data[...,5]

        # Take only detections with reasonably high confidence scores
        found = scores > confidence_thresh

        # Get image index and feature map coordinates (x,y) of detections
        origin_coords = np.argwhere(found)
        batch_idx = origin_coords[:,0]
        origin_coords = origin_coords[:,(2,1)].astype(np.float32)

        # Filter detections based on confidence
        aabb = aabb[found]
        theta = theta[found]
        scores = scores[found]
        if self._rotate_on:
            rotated = (batch_idx % 2).astype(np.bool)

        # If we rotated, the frames are interleaved, so batch location is halved
        if self._rotate_on:
            batch_idx = (batch_idx / 2).astype(int)

        # Get sin and cosine of box angles
        cos = np.cos(theta)[:,None]
        sin = np.sin(theta)[:,None]

        # Rescale AABB values to frame dimensions
        w = aabb[:,(1,3)]
        h = aabb[:,(0,2)]
        wx = blob2frame_scale[0] * cos * w
        wy = blob2frame_scale[1] * sin * w
        hx = blob2frame_scale[0] * sin * h
        hy = blob2frame_scale[1] * cos * h
        if self._rotate_on:
            inv = blob2frame_scale[1] / blob2frame_scale[0]
            wx[rotated,:] *= inv
            wy[rotated,:] /= inv
            hx[rotated,:] *= inv
            hy[rotated,:] /= inv
        aabb[:,(1,3)] = np.sqrt(wx ** 2.0 + wy ** 2.0)
        aabb[:,(0,2)] = np.sqrt(hx ** 2.0 + hy ** 2.0)

        # Correct box angles for rescale
        theta = np.arctan2(wy.sum(axis=-1),wx.sum(axis=-1))
        if self._rotate_on:
            theta[rotated] -= np.pi / 2

        # Rescale origin coordinates to frame dimensions
        origin_coords *= feat2blob_scale * blob2frame_scale

        # Add padding
        aabb += padding * (aabb[:,[0]] + aabb[:,[2]])

        rboxes = np.hstack((
            origin_coords,
            aabb,
            theta[:,None]
        ))

        return batch_idx, rboxes, scores

    def process_image(self, image, max_side_len, padding,
                      rotate_on, confidence_threshold, overlap_threshold,
                      text_height_threshold, rotation_threshold):
        """ Process a single image using the given arguments
        :param image: The image to be processed. Takes the following shape:
                (frame_height, frame_width, num_channels)
        :param max_side_len: Maximum length (pixels) for one side of the image.
                Before being processed, the image will be resized such that the
                long edge is at most max_side_length, while maintaining the same
                aspect ratio, and then further resized such that both dimensions
                are divisible by 32 (a requirement for EAST).
        :param padding: Padding to symmetrically add to bounding boxes. Each
                side is extended by 0.5 * padding * box_height.
        :param rotate_on: Whether to perform a second pass on the image after
                rotating 90 degrees. This can potentially pick up more text at
                high angles (larger than +/-60 degrees).
        :param confidence_threshold: Threshold to use for filtering detections
                by bounding box confidence.
        :param overlap_threshold: Threshold value for deciding whether
                regions overlap enough to be merged.
        :param text_height_threshold: Threshold value for deciding whether
                regions containing different text sizes should be merged.
        :param rotation_threshold: Threshold value for deciding whether regions
                containing text at different orientations should be merged.
        :return: List of mpf.ImageLocation detections. The angles of detected
                bounding boxes are stored in detection_properties['ROTATION'].
                The rotation is expressed in positive degrees counterclockwise
                from the 3 o'clock position.
        """
        # Convert to compatible shape
        frame_height = image.shape[0]
        frame_width = image.shape[1]
        blob_width, blob_height = self._get_blob_dimensions(
            frame_width=frame_width,
            frame_height=frame_height,
            max_side_len=max_side_len
        )

        # Load the model (may return immediately if compatible model is cached)
        self._load_model(
            blob_width=blob_width,
            blob_height=blob_height,
            rotate_on=rotate_on
        )

        # Convert the image to OpenCV-compatible blob, pass to processor
        batch_idx, rboxes, scores = self._process_blob(
            blob=cv2.dnn.blobFromImage(
                image=image,
                size=(blob_width, blob_height),
                mean=_mean_rgb,
                swapRB=True,
                crop=False
            ),
            frame_width=frame_width,
            frame_height=frame_height,
            confidence_thresh=confidence_threshold,
            padding=padding
        )

        if not len(rboxes):
            return []

        quads, scores = lanms_approx(
            rboxes=rboxes,
            scores=scores,
            overlap_threshold=overlap_threshold,
            text_height_threshold=text_height_threshold,
            rotation_threshold=rotation_threshold
        )
        return [
            mpf.ImageLocation(
                x_left_upper=x,
                y_left_upper=y,
                width=w,
                height=h,
                confidence=score,
                detection_properties={'ROTATION': rot}
            )
            for x, y, w, h, rot, score in quad_to_iloc(quads, scores)
        ]

    def process_frames(self, frames, max_side_len, padding,
                       rotate_on, confidence_threshold, overlap_threshold,
                       text_height_threshold, rotation_threshold):
        """ Process a volume of images using the given arguments
        :param frames: The images to be processed. Takes the following shape:
                (batch_size, frame_height, frame_width, num_channels)
        :param max_side_len: Maximum length (pixels) for one side of the image.
                Before being processed, the image will be resized such that the
                long edge is at most max_side_length, while maintaining the same
                aspect ratio, and then further resized such that both dimensions
                are divisible by 32 (a requirement for EAST).
        :param padding: Padding to symmetrically add to bounding boxes. Each
                side is extended by 0.5 * padding * box_height.
        :param rotate_on: Whether to perform a second pass on the image after
                rotating 90 degrees. This can potentially pick up more text at
                high angles (larger than +/-60 degrees).
        :param confidence_threshold: Threshold to use for filtering detections
                by bounding box confidence.
        :param overlap_threshold: Threshold value for deciding whether
                regions overlap enough to be merged.
        :param text_height_threshold: Threshold value for deciding whether
                regions containing different text sizes should be merged.
        :param rotation_threshold: Threshold value for deciding whether regions
                containing text at different orientations should be merged.
        :return: List lists of mpf.ImageLocation detections. Each list
                corresponds with one frame of the input volume. The angles of
                detected bounding boxes are stored in
                detection_properties['ROTATION']. The rotation is expressed in
                positive degrees counterclockwise from the 3 o'clock position.
        """
        # Convert to compatible shape
        batch_size = frames.shape[0]
        frame_height = frames.shape[1]
        frame_width = frames.shape[2]
        blob_width, blob_height = self._get_blob_dimensions(
            frame_width=frame_width,
            frame_height=frame_height,
            max_side_len=max_side_len
        )

        # Load the model (may return immediately if compatible model is cached)
        self._load_model(
            blob_width=blob_width,
            blob_height=blob_height,
            rotate_on=rotate_on
        )

        # Convert the image to OpenCV-compatible blob, pass to processor
        batch_idx, rboxes, scores = self._process_blob(
            blob=cv2.dnn.blobFromImages(
                images=frames,
                size=(blob_width, blob_height),
                mean=_mean_rgb,
                swapRB=True,
                crop=False
            ),
            frame_width=frame_width,
            frame_height=frame_height,
            confidence_thresh=confidence_thresh,
            padding=padding
        )

        # Split by frame so that boxes in different frames don't interfere
        # with one another during non-max suppression
        split_points = np.cumsum(np.bincount(batch_idx, minlength=batch_size))
        split_points = [0] + split_points.tolist()

        # Apply non-max suppression to the detections in each frame
        image_locs = []
        for i in range(len(split_points)-1):
            j0, j1 = split_points[i], split_points[i+1]
            if j1 > j0:
                quads, scores = lanms_approx(
                    rboxes=rboxes[j0:j1],
                    scores=scores[j0:j1],
                    overlap_threshold=overlap_threshold,
                    text_height_threshold=text_height_threshold,
                    rotation_threshold=rotation_threshold
                )
                image_locs.append([
                    mpf.ImageLocation(
                        x_left_upper=x,
                        y_left_upper=y,
                        width=w,
                        height=h,
                        confidence=score,
                        detection_properties={'ROTATION': rot}
                    )
                    for x, y, w, h, rot, score in quad_to_iloc(quads, scores)
                ])
            else:
                image_locs.append([])

        return image_locs
