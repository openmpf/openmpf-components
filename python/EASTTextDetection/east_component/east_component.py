from __future__ import division, print_function

import numpy as np
import os

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from east_processor import EASTProcessor

logger = mpf.configure_logging('east-detection.log', __name__ == '__main__')
ModelSettings = (
    util.ModelsIniParser(pkg_resources.resource_filename(__name__, 'models'))
    .register_path_field('model_bin')
    .register_field('rbox_layer_name')
    .register_field('score_layer_name')
    .build_class()
)


class EASTComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin, object):
    detection_type = 'TEXT'

    def __init__(self):
        logger.info('Creating instance of EASTComponent')
        self.processor = EASTProcessor(logger)


    @staticmethod
    def _parse_properties(job_properties):
        """
        :param job_properties: Properties object from VideoJob or ImageJob
        :return: Dictionary of properties, pass as **kwargs to processor
        """
        max_side_len = int(job_properties.get('MAX_SIDE_LENGTH','-1'))

        confidence_thresh = float(job_properties.get('CONFIDENCE_THRESHOLD','0.8'))
        nms_thresh = float(job_properties.get('NMS_THRESHOLD','0.2'))

        # Get mean channel values
        mean_r = float(job_properties.get('SUBTRACT_RED_VALUE','0'))
        mean_g = float(job_properties.get('SUBTRACT_BLUE_VALUE','0'))
        mean_b = float(job_properties.get('SUBTRACT_GREEN_VALUE','0'))
        mean = (mean_r,mean_g,mean_b)

        rotate_on = (job_properties.get('ROTATE_ON','').lower() == 'true')
        common_models_dir = os.path.realpath(job_properties.get('MODELS_DIR_PATH', '.'))
        plugin_models_dir = os.path.join(common_models_dir, 'EASTTextDetection')
        settings = ModelSettings(model_name, plugin_models_dir)

        model_bin = os.path.realpath(settings.model_bin)
        layer_names = [settings.rbox_layer_name, settings.score_layer_name]
        return dict(
            max_side_len=max_side_len,
            mean=mean,
            rotate_on=rotate_on,
            model_bin=model_bin,
            layer_names=layer_names,
            confidence_thresh=confidence_thresh,
            nms_thresh=nms_thresh
        )


    def get_detections_from_image_reader(self, image_job, image_reader):
        image = image_reader.get_image()

        return self.processor.process_image(
            image=image,
            **self._parse_properties(image_job.job_properties)
        )

    @staticmethod
    def _batches_from_video_capture(video_capture, batch_size):
        frames = []
        for frame in video_capture:
            frames.append(frame)
            if len(frames) >= batch_size:
                yield np.stack(frames)
            frames = []
        if len(frames):
            yield np.stack(frames)


    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)

        kwargs = self._parse_properties(video_job.job_properties)
        batch_size = int(job_properties.get('BATCH_SIZE','32'))

        tracks = []

        batch_offset = 0
        batch_gen = self._batches_from_video_capture(video_capture, batch_size)
        for batch in batch_gen:
            frames_dets = self.processor.process_frames(frames=batch, **kwargs)
            for i, frame_dets in enumerate(frames_dets):
                frame_index = batch_offset + i
                for image_loc in frame_dets:
                    tracks.append(mp.VideoTrack(
                        start_frame=frame_index,
                        stop_frame=frame_index,
                        confidence=image_loc.confidence,
                        frame_locations={frame_index: image_loc},
                        detection_properties=image_loc.detection_properties
                    ))
                ])
            batch_offset += len(batch)

        return tracks
