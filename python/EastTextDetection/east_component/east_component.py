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

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from east_processor import EastProcessor

logger = mpf.configure_logging('east-text-detection.log', __name__ == '__main__')


class EastComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin, object):
    detection_type = 'TEXT REGION'

    def __init__(self):
        logger.info('Creating instance of EastComponent')
        self.processor = EastProcessor(logger)

    @staticmethod
    def _parse_properties(job_properties):
        """
        :param job_properties: Properties object from VideoJob or ImageJob
        :return: Dictionary of properties, pass as **kwargs to processor
        """
        # Get the maximum side length (pixels) of the images
        max_side_len = int(job_properties.get('MAX_SIDE_LENGTH','-1'))

        # Get the batch size for video
        batch_size = int(job_properties.get('BATCH_SIZE','1'))

        # Get the threshold values for filtering bounding boxes
        confidence_threshold = float(job_properties.get('CONFIDENCE_THRESHOLD','0.8'))
        overlap_threshold = float(job_properties.get('OVERLAP_THRESHOLD','0.1'))
        text_height_threshold = float(job_properties.get('TEXT_HEIGHT_THRESHOLD','0.3'))
        rotation_threshold = float(job_properties.get('ROTATION_THRESHOLD','5'))

        # Get whether to do a second pass at 90 degrees
        rotate_on = (job_properties.get('ROTATE_AND_DETECT','').lower() == 'true')

        padding = float(job_properties.get('PADDING','0.15'))

        return dict(
            max_side_len=max_side_len,
            batch_size=batch_size,
            padding=padding,
            rotate_on=rotate_on,
            confidence_threshold=confidence_threshold,
            overlap_threshold=overlap_threshold,
            text_height_threshold=text_height_threshold,
            rotation_threshold=rotation_threshold
        )

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info(
            '[%s] Received image job: %s',
            image_job.job_name,
            image_job
        )

        kwargs = self._parse_properties(image_job.job_properties)
        image = image_reader.get_image()

        try:
            logger.info('[%s] Loading model', image_job.job_name)
            self.processor.load_model(
                frame_width=image.shape[1],
                frame_height=image.shape[0],
                max_side_len=kwargs['max_side_len'],
                rotate_on=kwargs['rotate_on']
            )
        except Exception as e:
            error_str = "[{:s}] Exception occurred while loading model: {:s}".format(
                image_job.job_name,
                str(e)
            )
            logger.exception(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.INVALID_PROPERTY)

        dets = self.processor.process_image(image, **kwargs)
        logger.info(
            '[%s] Processing complete. Found %d detections',
            image_job.job_name,
            len(dets)
        )
        return dets

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
        logger.info(
            '[%s] Received video job: %s',
            video_job.job_name,
            video_job
        )

        kwargs = self._parse_properties(video_job.job_properties)

        try:
            frame_width, frame_height = video_capture.frame_size
            logger.info('[%s] Loading model', video_job.job_name)
            self.processor.load_model(
                frame_width=frame_width,
                frame_height=frame_height,
                max_side_len=kwargs['max_side_len'],
                rotate_on=kwargs['rotate_on']
            )
        except Exception as e:
            error_str = "[{:s}] Exception occurred while loading model: {:s}".format(
                video_job.job_name,
                str(e)
            )
            logger.exception(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.INVALID_PROPERTY)

        tracks = []

        batch_offset = 0
        batch_gen = self._batches_from_video_capture(
            video_capture,
            kwargs['batch_size']
        )
        for batch in batch_gen:
            try:
                frames_dets = self.processor.process_frames(batch, **kwargs)
            except Exception as e:
                error_str = "[{:s}] Exception occurred while processing batch: {:s}".format(
                    video_job.job_name,
                    str(e)
                )
                logger.exception(error_str)
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

        logger.info(
            '[%s] Processing complete. Found %d tracks',
            video_job.job_name,
            len(tracks)
        )
        return tracks


EXPORT_MPF_COMPONENT = EastComponent
