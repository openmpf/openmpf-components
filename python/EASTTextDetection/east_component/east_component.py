from __future__ import division, print_function

import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from east_processor import EASTProcessor

logger = mpf.configure_logging('east-detection.log', __name__ == '__main__')


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
        # Get the maximum side length (pixels) of the images
        max_side_len = int(job_properties.get('MAX_SIDE_LENGTH','-1'))

        # Get the threshold values for filtering bounding boxes
        confidence_threshold = float(job_properties.get('CONFIDENCE_THRESHOLD','0.8'))
        overlap_threshold = float(job_properties.get('OVERLAP_THRESHOLD','0.2'))
        text_height_threshold = float(job_properties.get('TEXT_HEIGHT_THRESHOLD','0.2'))
        rotation_threshold = float(job_properties.get('ROTATION_THRESHOLD','5'))

        # Get whether to doa second pass at 90 degrees
        rotate_on = (job_properties.get('ROTATE_ON','').lower() == 'true')

        padding = float(job_properties.get('PADDING','0.15'))

        return dict(
            max_side_len=max_side_len,
            padding=padding,
            rotate_on=rotate_on,
            confidence_threshold=confidence_threshold,
            overlap_threshold=overlap_threshold,
            text_height_threshold=text_height_threshold,
            rotation_threshold=rotation_threshold
        )

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info('[%s] Received image job: %s', image_job.job_name, image_job)
        return self.processor.process_image(
            image=image_reader.get_image(),
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
        # Pass leftover frames as their own batch; EAST doesn't require
        # consistent batch sizes
        if len(frames):
            yield np.stack(frames)

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('[%s] Received video job: %s', video_job.job_name, video_job)

        kwargs = self._parse_properties(video_job.job_properties)
        batch_size = int(video_job.job_properties.get('BATCH_SIZE','32'))

        tracks = []

        batch_offset = 0
        batch_gen = self._batches_from_video_capture(video_capture, batch_size)
        for batch in batch_gen:
            try:
                frames_dets = self.processor.process_frames(frames=batch, **kwargs)
            except Exception as e:
                error_str = "Exception occurred while processing batch: " + str(e)
                logger.error(error_str)
                raise mpf.DetectionException(error_str, mpf.DetectionError.DETECTION_FAILED)

            # Convert from lists of ImageLocations to VideoTracks
            for i, frame_dets in enumerate(frames_dets):
                frame_index = batch_offset + i
                for image_loc in frame_dets:
                    tracks.append(mpf.VideoTrack(
                        start_frame=frame_index,
                        stop_frame=frame_index,
                        confidence=image_loc.confidence,
                        frame_locations={frame_index: image_loc},
                        detection_properties=image_loc.detection_properties
                    ))
            batch_offset += len(batch)

        return tracks


EXPORT_MPF_COMPONENT = EASTComponent
