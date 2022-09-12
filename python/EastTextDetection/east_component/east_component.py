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

import logging

import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from .east_processor import EastProcessor

logger = logging.getLogger('EastComponent')


class EastComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin, object):
    detection_type = 'TEXT REGION'

    def __init__(self):
        logger.info('Creating instance of EastComponent')
        self.processor = EastProcessor()

    @staticmethod
    def _parse_properties(props):
        """
        :param job_properties: Properties object from VideoJob or ImageJob
        :return: Dictionary of properties, pass as **kwargs to processor
        """
        # Get the maximum side length (pixels) of the images
        max_side_len = int(props.get('MAX_SIDE_LENGTH','-1'))

        # Get the batch size for video
        batch_size = int(props.get('BATCH_SIZE','1'))

        # Get the threshold values for filtering bounding boxes
        min_confidence = float(props.get('CONFIDENCE_THRESHOLD','0.8'))
        overlap_threshold = float(props.get('MERGE_OVERLAP_THRESHOLD','0.0'))
        min_nms_overlap = float(props.get('NMS_MIN_OVERLAP','0.1'))
        max_height_delta = float(props.get('MERGE_MAX_TEXT_HEIGHT_DIFFERENCE','0.3'))
        max_rot_delta = float(props.get('MERGE_MAX_ROTATION_DIFFERENCE','10.0'))
        min_structured_score = float(props.get('MIN_STRUCTURED_TEXT_THRESHOLD','0.01'))

        # Get whether to do a second pass at 90 degrees
        rotate_on = (props.get('ROTATE_AND_DETECT','FALSE').lower() == 'true')
        merge_on = (props.get('MERGE_REGIONS','TRUE').lower() == 'true')
        vsupp_on = (props.get('SUPPRESS_VERTICAL','TRUE').lower() == 'true')

        temp_padding_x = float(props.get('TEMPORARY_PADDING_X','0.1'))
        temp_padding_y = float(props.get('TEMPORARY_PADDING_Y','0.1'))
        final_padding = float(props.get('FINAL_PADDING','0.0'))

        return dict(
            max_side_len=max_side_len,
            batch_size=batch_size,
            temp_padding_x=temp_padding_x,
            temp_padding_y=temp_padding_y,
            final_padding=final_padding,
            rotate_on=rotate_on,
            merge_on=merge_on,
            suppress_vertical=vsupp_on,
            min_confidence=min_confidence,
            overlap_threshold=overlap_threshold,
            min_nms_overlap=min_nms_overlap,
            max_height_delta=max_height_delta,
            max_rot_delta=max_rot_delta,
            min_structured_score=min_structured_score
        )

    def get_detections_from_image_reader(self, image_job, image_reader):
        logger.info('Received image job: %s', image_job)

        kwargs = self._parse_properties(image_job.job_properties)
        image = image_reader.get_image()

        try:
            logger.info('Loading model...')
            self.processor.load_model(
                frame_width=image.shape[1],
                frame_height=image.shape[0],
                max_side_len=kwargs['max_side_len'],
                rotate_on=kwargs['rotate_on']
            )
            logger.info('Model loaded.')
        except Exception as e:
            error_str = "Exception occurred while loading model: {}".format(e)
            logger.exception(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.DETECTION_NOT_INITIALIZED)

        dets = self.processor.process_image(image, **kwargs)
        logger.info('Processing complete. Found %d detections', len(dets))
        return dets

    @staticmethod
    def _batches_from_video_capture(video_capture, batch_size):
        frames = []
        for frame in video_capture:
            frames.append(frame)
            if len(frames) >= batch_size:
                yield len(frames), np.stack(frames)
                frames = []

        # Pad the leftover frames rather than reload the model potentially twice
        if len(frames):
            padded = np.pad(
                array=np.stack(frames),
                pad_width=((0, batch_size - len(frames)), (0,0), (0,0), (0,0)),
                mode='constant',
                constant_values=0
            )
            yield len(frames), padded

    def get_detections_from_video_capture(self, video_job, video_capture):
        logger.info('Received video job: %s', video_job)

        kwargs = self._parse_properties(video_job.job_properties)

        try:
            frame_width, frame_height = video_capture.frame_size
            logger.info('Loading model...')
            self.processor.load_model(
                frame_width=frame_width,
                frame_height=frame_height,
                max_side_len=kwargs['max_side_len'],
                rotate_on=kwargs['rotate_on'],
                batch_size=kwargs['batch_size']
            )
            logger.info('Model loaded.')
        except Exception as e:
            error_str = "Exception occurred while loading model: {}".format(e)
            logger.exception(error_str)
            raise mpf.DetectionException(error_str, mpf.DetectionError.DETECTION_NOT_INITIALIZED)

        tracks = []

        batch_offset = 0
        batch_gen = self._batches_from_video_capture(
            video_capture,
            kwargs['batch_size']
        )
        for n, batch in batch_gen:
            try:
                frames_dets = self.processor.process_frames(batch, **kwargs)[:n]
            except Exception as e:
                error_str = "Exception occurred while processing batch: {}".format(e)
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

        logger.info('Processing complete. Found %d tracks', len(tracks))
        return tracks
