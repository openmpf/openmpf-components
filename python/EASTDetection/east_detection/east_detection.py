from __future__ import division, print_function

import numpy as np
import cv2
import os

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = mpf.configure_logging('east-detection.log', __name__ == '__main__')

class EASTComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin, object):
    detection_type = 'TEXT'

    def __init__(self):
        logger.info('Creating instance of EASTComponent')
        self._input_width = None
        self._input_height = None
        self._rotate_on = None
        self._model_bin = None
        self._layer_names = None
        self._run_model = None
        self._initialized = False


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

    @staticmethod
    def _parse_properties(job_properties):
        """
        :param job_properties: Properties object from VideoJob or ImageJob
        :return: cv2.dnn model object
        """
        max_side_len = int(job_properties.get('MAX_SIDE_LENGTH','-1'))

        confidence_thresh = float(job_properties.get('CONFIDENCE_THRESHOLD','0.8'))
        nms_thresh = float(job_properties.get('NMS_THRESHOLD','0.2'))

        # Get mean channel values
        mean_r = float(job_properties.get('SUBTRACT_RED_VALUE','0'))
        mean_g = float(job_properties.get('SUBTRACT_BLUE_VALUE','0'))
        mean_b = float(job_properties.get('SUBTRACT_GREEN_VALUE','0'))
        mean_ch = (mean_r,mean_g,mean_b)

        rotate_on = (job_properties.get('ROTATE_ON','').lower() == 'true')
        plugin_path = os.path.realpath(job_properties.get('MODELS_DIR_PATH','.'))
        common_models_dir = os.path.join(plugin_path, 'models')
        settings = (
            util.ModelsIniParser(pkg_resources.resource_filename(__name__, 'models'))
            .register_path_field('model_bin')
            .register_field('rbox_layer_name')
            .register_field('score_layer_name')
            .build_class()(model_name, common_models_dir)
        )

        model_bin = os.path.realpath(settings.model_bin)
        if not os.path.isfile(model_bin):
            error_str = 'Model protobuf file could not be found: ' + model_bin
            logger.error(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.INVALID_PROPERTY)

        layer_names = [settings.rbox_layer_name, settings.score_layer_name]
        return dict(
            rotate_on=rotate_on,
            model_bin=model_bin,
            layer_names=layer_names,
            max_side_len=max_side_len,
            mean_ch=mean_ch,
            confidence_thresh=confidence_thresh,
            nms_thresh=nms_thresh
        )

    def _load_model(self, rotate_on, model_bin, layer_names):
        """
        :param job_properties: Properties object from VideoJob or ImageJob
        :return: cv2.dnn model object
        """
        try:
            model = cv2.dnn.readNet(model_bin)
            if rotate_on:
                rot_model = cv2.dnn.readNet(model_bin)
        except Exception as e:
            error_str = "Exception occurred while loading {:s}: {:s}".format(
                model_bin,
                str(e)
            )
            logger.error(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.INVALID_PROPERTY)

        def run_model(blob):
            model.setInput(blob)
            data = np.concatenate(model.forward(layer_names), axis=1)
            if rotate_on:
                # Rotate input, run model, rotate output back, interleave
                rot_model.setInput(np.rot90(blob, k=1, axes=(2,3)))
                rot_data = np.concatenate(rot_model.forward(layer_names), axis=1)
                rot_data = np.rot90(rot_data, k=-1, axes=(2,3))
                data = np.repeat(data, 2, axis=0)
                data[1::2,...] = rot_data
            return data

        self._run_model = run_model

    def _process_batch(self, batch,
                       rotate_on, model_bin, layer_names, max_side_len,
                       mean_ch, confidence_thresh, nms_thresh):
        # If "batch" is a single image, prepend an axis
        if len(batch.shape) == 3:
            batch = batch[None,...]

        batch_size, frame_height, frame_width, n_channels = batch.shape
        blob = cv2.dnn.blobFromImages(
            images=batch,
            size=self._get_blob_dimensions(
                frame_width=frame_width,
                frame_height=frame_height,
                max_side_len=max_side_len
            ),
            mean=mean_ch,
            swapRB=True,
            crop=False
        )

        blob_height = blob.shape[2]
        blob_width = blob.shape[3]
        if not (self._initialized
                and self._input_width == blob_width
                and self._input_height == blob_height
                and self._rotate_on == rotate_on
                and self._model_bin == model_bin
                and self._layer_names == layer_names):
            self._initialized = False
            self._load_model(
                rotate_on=rotate_on,
                model_bin=model_bin,
                layer_names=layer_names
            )
            self._input_width = blob_width
            self._input_height = blob_height
            self._rotate_on = rotate_on
            self._model_bin = model_bin
            self._layer_names = layer_names
            self._initialized = True

        data = self._run_model(blob)

        # Reshape into [n_frames, feat_h, feat_w, 6]
        data = np.moveaxis(data, 1, -1)
        _, feat_height, feat_width, _ = batch.shape

        # Get scaling factor from feature map to blob (should be (4.0,4.0))
        # and from blob to image
        frame_dims = np.array([frame_width, frame_height],dtype=np.float32)
        blob_dims = np.array([blob_width, blob_height],dtype=np.float32)
        feat_dims = np.array([feat_width, feat_height],dtype=np.float32)
        feat2blob_scale = blob_dims / feat_dims
        blob2frame_scale = frame_dims / blob_dims

        # Split into AABB, angle, and confidence
        rbox, theta, scores = data[...,:4], data[...,4], data[...,5]

        # Get sin and cosine of line angle
        cos = np.cos(theta)[...,None]
        sin = np.sin(theta)[...,None]

        # Scale bounding box width and height to image dimensions
        w = rbox[...,(1,3)]
        h = rbox[...,(0,2)]
        wx = blob2frame_scale[0] * cos * w
        wy = blob2frame_scale[1] * sin * w
        hx = blob2frame_scale[0] * sin * h
        hy = blob2frame_scale[1] * cos * h
        if rotate_on:
            inv = blob2frame_scale[1] / blob2frame_scale[0]
            wx[1::2,...] *= inv
            wy[1::2,...] /= inv
            hx[1::2,...] *= inv
            hy[1::2,...] /= inv
        rbox[...,(1,3)] = np.sqrt(wx ** 2.0 + wy ** 2.0)
        rbox[...,(0,2)] = np.sqrt(hx ** 2.0 + hy ** 2.0)
        theta = np.arctan2(wy.sum(axis=-1),wx.sum(axis=-1))
        if rotate_on:
            theta[1::2,...] -= np.pi / 2

        # Take only detections with reasonably high confidence scores
        found = scores > confidence_thresh

        # Get image index and feature map coordinates (x,y) of detections
        origin_coords = np.argwhere(found)
        b_locs = origin_coords[:,0]
        origin_coords = origin_coords[:,(2,1)].astype(np.float32)

        # If we rotated, the frames are interleaved, so batch location is halved
        if rotate_on:
            b_locs = (b_locs / 2).astype(int)

        # Filter detections based on confidence
        rbox = rbox[found]
        theta = theta[found]
        scores = scores[found]

        # Rescale origin coordinates and RBOX distances to original frame
        origin_coords *= feat2blob_scale * blob2frame_scale

        # Get dimesions (w,h) of detected bounding boxes
        bbox_dims = rbox[:,(1,0)] + rbox[:,(3,2)]

        # Get sin and cosine of line angle
        cos = np.cos(theta)
        sin = np.sin(theta)

        # Get coordinates (x,y) of bounding boxes'top left corners
        tl = np.empty_like(origin_coords)
        tl[:,0] = origin_coords[:,0] - sin*rbox[:,0] - cos*rbox[:,3]
        tl[:,1] = origin_coords[:,1] - cos*rbox[:,0] + sin*rbox[:,3]

        # Get coordinates of bounding boxes' centers
        center = np.empty_like(origin_coords)
        center[:,0] = tl[:,0] + 0.5 * (cos*bbox_dims[:,0] + sin*bbox_dims[:,1])
        center[:,1] = tl[:,1] - 0.5 * (sin*bbox_dims[:,0] - cos*bbox_dims[:,1])

        # Convert radians to degrees, the format expected by NMS
        theta = np.degrees(-theta)

        # Put in format (cx, cy, tlx, tly, w, h, theta, score)
        data = np.hstack((center, bbox_dims, theta[:,None], scores[:,None]))

        def frame_nms(frame_detections):
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
            return frame_detections[indices,2:]

        # If we're processing only one image
        if batch_size == 1:
            return frame_nms(data)

        # Split by frame so that boxes in different frames don't interfere
        # with one another during non-max suppression
        split_points = np.cumsum(np.bincount(b_locs, minlength=batch_size))
        groups = np.split(data, split_points)

        # Apply non-max suppression to the detections in each frame
        out_groups = []
        for group in groups:
            if len(group):
                out_groups.append(frame_nms(group))
            else:
                out_groups.append(np.array([],dtype=np.float32))

        return out_groups


    def get_detections_from_image_reader(self, image_job, image_reader):
        image = image_reader.get_image()

        boxes = self._process_batch(
            batch=image,
            **self._parse_properties(image_job.job_properties)
        )

        detections = [
            mpf.ImageLocation(x, y, w, h, score, {'ANGLE' : theta})
            for x, y, w, h, theta, score in boxes
        ]

        return detections


    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)
        # If frame index is not required, you can just loop over video_capture directly
        for frame_index, frame in enumerate(video_capture):
            for result_track in run_detection_algorithm_on_frame(frame_index, frame):
                # Alternatively, while iterating through the video, add tracks to a list. When done, return that list.
                yield result_track
