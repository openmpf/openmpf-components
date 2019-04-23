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
        blob_width = frame_width
        blob_height = frame_height

        # limit the max side
        if max_side_len > 0:
            long_side = max(blob_width, blob_height)
            resized_long_side = min(max_side_len, long_side)
            ratio = float(resized_long_side) / long_side

            blob_width = int(blob_width * ratio)
            blob_height = int(blob_height * ratio)

        if blob_width % 32:
            blob_width = (blob_width // 32) * 32
        if blob_height % 32:
            blob_height = (blob_height // 32) * 32

        blob_width = max(32, blob_width)
        blob_height = max(32, blob_height)

        return (blob_width, blob_height)

    def _load_model(self, model_bin, layer_names, blob_width, blob_height, rotate_on):
        # Check if cached model can be used (this saves time for video)
        if self._model is None:
            pass
        elif self._model_bin != model_bin or self._layer_names != layer_names:
            self.logger.info("Replacing cached model.")
        elif self._blob_width != blob_width or self._blob_height != blob_height:
            self.logger.info("Cached model incompatible with job properties.")
        elif self._model_90 is None and rotate_on:
            self.logger.info("Cached model can't handle rotated inputs.")
        else:
            self.logger.info("Using cached model.")
            return

        self._model = None
        self._model_90 = None
        self.logger.info("Loading model: " + model_bin)
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

        self._model_bin = model_bin
        self._layer_names = layer_names
        self._blob_width = blob_width
        self._blob_height = blob_height
        self._rotate_on = rotate_on

    def _process_blob(self, blob, frame_width, frame_height, confidence_thresh):
        blob_width = self._blob_width
        blob_height = self._blob_height

        # Run blob through model to get geometry and scores
        self._model.setInput(blob)
        data = np.concatenate(self._model.forward(self._layer_names), axis=1)
        if self._rotate_on:
            # Rotate input, run model, rotate output back, interleave
            self._model_90.setInput(np.rot90(blob, k=1, axes=(2,3)))
            data_r = np.concatenate(self._model_90.forward(self._layer_names), axis=1)
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
        if self._rotate_on:
            rotated = (batch_idx % 2).astype(np.bool)

        # If we rotated, the frames are interleaved, so batch location is halved
        if self._rotate_on:
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
        if self._rotate_on:
            inv = blob2frame_scale[1] / blob2frame_scale[0]
            wx[rotated,:] *= inv
            wy[rotated,:] /= inv
            hx[rotated,:] *= inv
            hy[rotated,:] /= inv
        rbox[:,(1,3)] = np.sqrt(wx ** 2.0 + wy ** 2.0)
        rbox[:,(0,2)] = np.sqrt(hx ** 2.0 + hy ** 2.0)

        # Correct box angles for rescale
        theta = np.arctan2(wy.sum(axis=-1),wx.sum(axis=-1))
        if self._rotate_on:
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
            tl,
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
            detection_properties={'ROTATION': (720 - theta) % 360}
        )


    def process_image(self, image, model_bin, layer_names, max_side_len, mean,
                      rotate_on, confidence_thresh, nms_thresh):
        """ Process a single image using the given arguments
        :param image: The image to be processed. Takes the following shape:
                (frame_height, frame_width, num_channels)
        :param model_bin: Filename for the model protobuf binary (.pb file)
        :param layer_names: Name of the layers where bounding box geometry and
                scores are produced (in that order). The default as set by the
                descriptor is the following:
                ['feature_fusion/concat_3','feature_fusion/Conv_7/Sigmoid']
        :param max_side_len: Maximum length (pixels) for one side of the image.
                Before being processed, the image will be resized such that the
                long edge is at most max_side_length, while maintaining the same
                aspect ratio, and then further resized such that both dimensions
                are divisible by 32 (a requirement for EAST).
        :param mean: The mean pixel value (R,G,B) which will be subtracted from
                the image before processing.
        :param rotate_on: Whether to perform a second pass on the image after
                rotating 90 degrees. This can potentially pick up more text at
                high angles (larger than +/-60 degrees).
        :param confidence_thresh: Threshold to use for filtering detections by
                bounding box confidence.
        :param nms_thresh: Threshold value to use for filtering detections by
                overlap with other bounding boxes (non-max suppression).
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
            model_bin=model_bin,
            layer_names=layer_names,
            blob_width=blob_width,
            blob_height=blob_height,
            rotate_on=rotate_on
        )

        # Convert the image to OpenCV-compatible blob, pass to processor
        dets = self._process_blob(
            blob=cv2.dnn.blobFromImage(
                image=image,
                size=(blob_width, blob_height),
                mean=mean,
                swapRB=True,
                crop=False
            ),
            frame_width=frame_width,
            frame_height=frame_height,
            confidence_thresh=confidence_thresh
        )

        # Perform non-max suppression to filter out overlapping boxes
        dets = self._frame_nms(
            frame_detections=dets[:,1:],
            confidence_thresh=confidence_thresh,
            nms_thresh=nms_thresh
        )

        # Convert to mpf.ImageLocation detections and return
        return [self._get_image_location(d) for d in dets]


    def process_frames(self, frames, model_bin, layer_names, max_side_len, mean,
                       rotate_on, confidence_thresh, nms_thresh):
        """ Process a volume of images using the given arguments
        :param frames: The images to be processed. Takes the following shape:
                (batch_size, frame_height, frame_width, num_channels)
        :param max_side_len: Maximum length (pixels) for one side of the image.
                Before being processed, the image will be resized such that the
                long edge is at most max_side_length, while maintaining the same
                aspect ratio, and then further resized such that both dimensions
                are divisible by 32 (a requirement for EAST).
        :param mean: The mean pixel value (R,G,B) which will be subtracted from
                the image before processing.
        :param rotate_on: Whether to perform a second pass on the image after
                rotating 90 degrees. This can potentially pick up more text at
                high angles (larger than +/-60 degrees).
        :param model_bin: Filename for the model protobuf binary (.pb file)
        :param layer_names: Name of the layers where bounding box geometry and
                scores are produced (in that order). The default as set by the
                descriptor is the following:
                ['feature_fusion/concat_3','feature_fusion/Conv_7/Sigmoid']
        :param confidence_thresh: Threshold to use for filtering detections by
                bounding box confidence.
        :param nms_thresh: Threshold value to use for filtering detections by
                overlap with other bounding boxes (non-max suppression).
        :return: List of mpf.ImageLocation detections. The angles of detected
                bounding boxes are stored in detection_properties['ROTATION'].
                The rotation is expressed in positive degrees counterclockwise
                from the 3 o'clock position.
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
            model_bin=model_bin,
            layer_names=layer_names,
            blob_width=blob_width,
            blob_height=blob_height,
            rotate_on=rotate_on
        )

        # Convert the image to OpenCV-compatible blob, pass to processor
        dets = self._process_blob(
            blob=cv2.dnn.blobFromImages(
                images=frames,
                size=(blob_width, blob_height),
                mean=mean,
                swapRB=True,
                crop=False
            ),
            frame_width=frame_width,
            frame_height=frame_height,
            confidence_thresh=confidence_thresh
        )

        # Split by frame so that boxes in different frames don't interfere
        # with one another during non-max suppression
        split_points = np.cumsum(np.bincount(dets[:,0].astype(int), minlength=batch_size))
        groups = np.split(dets[:,1:], split_points)

        # Apply non-max suppression to the detections in each frame
        image_locs = []
        for group in groups:
            if len(group):
                dets = self._frame_nms(
                    frame_detections=group,
                    confidence_thresh=confidence_thresh,
                    nms_thresh=nms_thresh
                )
                image_locs.append([self._get_image_location(d) for d in dets])
            else:
                image_locs.append([])

        return image_locs
