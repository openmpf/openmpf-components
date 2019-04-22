from __future__ import division, print_function

import numpy as np
import cv2

import mpf_component_api as mpf

class EASTProcessor(object):

    def __init__(self, logger):
        self.logger = logger
        self._input_shape = None
        self._rotate_on = None
        self._model_bin = None
        self._layer_names = None
        self._model = None
        self._model_90 = None


    @staticmethod
    def _get_blob_dimensions(frame_width, frame_height, max_side_len):
        # limit the max side
        if max_side_len > 0:
            long_side = max(frame_width, frame_height)
            resized_long_side = min(max_side_len, long_side)
            ratio = float(resized_long_side) / long_side

            frame_width = int(frame_width * ratio)
            frame_height = int(frame_height * ratio)

        if frame_width % 32:
            frame_width = (frame_width // 32) * 32
        if frame_height % 32:
            frame_height = (frame_height // 32) * 32

        frame_width = max(32, frame_width)
        frame_height = max(32, frame_height)

        return (frame_width, frame_height)


    def _load_model(self, rotate_on, model_bin):
        self._model = None
        self._model_90 = None
        try:
            self._model = cv2.dnn.readNet(model_bin)
            if rotate_on:
                self._model_90 = cv2.dnn.readNet(model_bin)
        except Exception as e:
            error_str = "Exception occurred while loading {:s}: {:s}".format(
                model_bin,
                str(e)
            )
            self.logger.error(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.INVALID_PROPERTY)


    def _run_model(self, blob, rotate_on, model_bin, layer_names, **kwargs):
        input_shape = blob.shape[1:]
        if not (self._model is not None
                and self._input_shape == input_shape
                and self._rotate_on == rotate_on
                and self._model_bin == model_bin
                and self._layer_names == layer_names):
            self._load_model(rotate_on=rotate_on, model_bin=model_bin)
            self._input_shape = input_shape
            self._rotate_on = rotate_on
            self._model_bin = model_bin
            self._layer_names = layer_names

        self._model.setInput(blob)
        data = np.concatenate(self._model.forward(layer_names), axis=1)
        if rotate_on:
            # Rotate input, run model, rotate output back, interleave
            self._model_90.setInput(np.rot90(blob, k=1, axes=(2,3)))
            data_r = np.concatenate(self._model_90.forward(layer_names), axis=1)
            data_r = np.rot90(data_r, k=-1, axes=(2,3))
            data = np.repeat(data, 2, axis=0)
            data[1::2,...] = data_r
        return data


    def _process_blob(self, batch, max_side_len, mean, confidence_thresh, **kwargs):
        # If "batch" is a single image, prepend an axis
        single = (len(batch.shape) == 3)
        if single:
            batch = batch[None,...]

        batch_size = batch.shape[0]
        frame_h = batch.shape[1]
        frame_w = batch.shape[2]

        # Convert to compatible shape
        blob = cv2.dnn.blobFromImages(
            images=batch,
            size=self._get_blob_dimensions(
                frame_width=frame_w,
                frame_height=frame_h,
                max_side_len=max_side_len
            ),
            mean=mean,
            swapRB=True,
            crop=False
        )
        blob_h = blob.shape[2]
        blob_w = blob.shape[3]

        # Run blob through model to get geometry and scores
        data = self._run_model(blob, **kwargs)
        feat_h = batch.shape[2]
        feat_w = batch.shape[3]

        # Reshape into [n_frames, feat_h, feat_w, 6]
        data = np.moveaxis(data, 1, -1)

        # Get scaling factor from feature map to blob (should be (4.0,4.0))
        # and from blob to image
        frame_dims = np.array([frame_w, frame_h],dtype=np.float32)
        blob_dims = np.array([blob_w, blob_h],dtype=np.float32)
        feat_dims = np.array([feat_w, feat_h],dtype=np.float32)
        feat2blob_scale = blob_dims / feat_dims
        blob2frame_scale = frame_dims / blob_dims

        # Split into RBOX, angle, and confidence
        # RBOX: [dist2top, dist2right, dist2bottom, dist2left]
        rbox, theta, scores = data[...,:4], data[...,4], data[...,5]

        # Take only detections with reasonably high confidence scores
        found = scores > confidence_thresh

        # Get image index and feature map coordinates (x,y) of detections
        origin_coords = np.argwhere(found)
        batch_idx = origin_coords[:,0]
        origin_coords = origin_coords[:,(2,1)].astype(np.float32)

        # Filter detections based on confidence
        rbox = rbox[found]
        theta = theta[found]
        scores = scores[found]
        if rotate_on:
            rotated = (batch_idx % 2).astype(np.bool)

        # If we rotated, the frames are interleaved, so batch location is halved
        if rotate_on:
            batch_idx = (batch_idx / 2).astype(int)

        # Get sin and cosine of box angles
        cos = np.cos(theta)[:,None]
        sin = np.sin(theta)[:,None]

        # Rescale RBOX values to frame dimensions
        w = rbox[:,(1,3)]
        h = rbox[:,(0,2)]
        wx = blob2frame_scale[0] * cos * w
        wy = blob2frame_scale[1] * sin * w
        hx = blob2frame_scale[0] * sin * h
        hy = blob2frame_scale[1] * cos * h
        if rotate_on:
            inv = blob2frame_scale[1] / blob2frame_scale[0]
            wx[rotated,:] *= inv
            wy[rotated,:] /= inv
            hx[rotated,:] *= inv
            hy[rotated,:] /= inv
        rbox[:,(1,3)] = np.sqrt(wx ** 2.0 + wy ** 2.0)
        rbox[:,(0,2)] = np.sqrt(hx ** 2.0 + hy ** 2.0)

        # Correct box angles for rescale
        theta = np.arctan2(wy.sum(axis=-1),wx.sum(axis=-1))
        if rotate_on:
            theta[rotated] -= np.pi / 2

        # Rescale origin coordinates to frame dimensions
        origin_coords *= feat2blob_scale * blob2frame_scale

        # Get dimesions (w,h) of detected bounding boxes
        bbox_dims = rbox[:,(1,0)] + rbox[:,(3,2)]

        # Get sin and cosine of adjusted angles
        cos = np.cos(theta)
        sin = np.sin(theta)

        # Get coordinates (x,y) of bounding boxes' top left corners
        tl = np.empty_like(origin_coords)
        tl[:,0] = origin_coords[:,0] - sin*rbox[:,0] - cos*rbox[:,3]
        tl[:,1] = origin_coords[:,1] - cos*rbox[:,0] + sin*rbox[:,3]

        # Get coordinates of bounding boxes' centers
        center = np.empty_like(origin_coords)
        center[:,0] = tl[:,0] + 0.5 * (cos*bbox_dims[:,0] + sin*bbox_dims[:,1])
        center[:,1] = tl[:,1] - 0.5 * (sin*bbox_dims[:,0] - cos*bbox_dims[:,1])

        # Convert radians to degrees, the format expected by NMS
        theta = np.degrees(-theta)

        # Return in format (batch_idx, cx, cy, tlx, tly, w, h, theta, score)
        return np.hstack((
            batch_idx[:,None],
            center,
            bbox_dims,
            theta[:,None],
            scores[:,None]
        ))


    def _frame_nms(self, frame_detections, confidence_thresh, nms_thresh):
        boxes = zip(
            frame_detections[:,0:2].tolist(),
            frame_detections[:,4:6].tolist(),
            frame_detections[:,6].tolist()
        )
        confidences = frame_detections[:,7].tolist()
        indices = cv2.dnn.NMSBoxesRotated(
            boxes,
            confidences,
            confidence_thresh,
            nms_thresh
        ).flatten()

        # Return data required for MPF detections (no center coords)
        return frame_detections[indices,2:]


    @staticmethod
    def _get_image_location(detection):
        x, y, w, h, theta, score = detection
        return mpf.ImageLocation(
            x_left_upper=x,
            y_left_upper=y,
            width=w,
            height=h,
            confidence=score,
            detection_properties={'ROTATION': theta}
        )


    def process_image(self, image, max_side_len, mean, **kwargs):
        frame_h = image.shape[0]
        frame_w = image.shape[1]

        # Convert to compatible shape
        blob = cv2.dnn.blobFromImage(
            image=image,
            size=self._get_blob_dimensions(
                frame_width=frame_w,
                frame_height=frame_h,
                max_side_len=max_side_len
            ),
            mean=mean,
            swapRB=True,
            crop=False
        )
        dets = self._process_blob(blob, frame_h, frame_w, **kwargs)
        dets = self._frame_nms(dets[:,1:], **kwargs)
        return [self._get_image_location(d) for d in dets]


    def process_frames(self, frames, max_side_len, mean, **kwargs):
        batch_size = frames.shape[0]
        frame_h = frames.shape[1]
        frame_w = frames.shape[2]

        # Convert to compatible shape
        blob = cv2.dnn.blobFromImages(
            images=frames,
            size=self._get_blob_dimensions(
                frame_width=frame_w,
                frame_height=frame_h,
                max_side_len=max_side_len
            ),
            mean=mean,
            swapRB=True,
            crop=False
        )
        dets = self._process_blob(blob, frame_h, frame_w, **kwargs)

        # Split by frame so that boxes in different frames don't interfere
        # with one another during non-max suppression
        split_points = np.cumsum(np.bincount(dets[:,0], minlength=batch_size))
        groups = np.split(dets[:,1:], split_points)

        # Apply non-max suppression to the detections in each frame
        image_locs = []
        for group in groups:
            if len(group):
                dets = self._frame_nms(group, **kwargs)
                image_locs.append([self._get_image_location(d) for d in dets])
            else:
                image_locs.append([]])

        return image_locs
