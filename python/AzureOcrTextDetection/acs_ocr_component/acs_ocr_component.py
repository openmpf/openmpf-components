#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2024 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2024 The MITRE Corporation                                      #
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

import json
import logging
import math
import os
import urllib
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util


logger = logging.getLogger('AcsOcrComponent')


class AcsOcrComponent(mpf_util.ImageReaderMixin, mpf_util.VideoCaptureMixin):

    def get_detections_from_image_reader(self, image_job, image_reader):
        try:
            logger.info('Received image job: %s', image_job)

            num_detections = 0
            image = image_reader.get_image()
            detections = JobRunner(image_job.job_properties).get_indexed_detections((image,))
            for idx, detection in detections:
                # We don't need to anything with the index for a image jobs.
                yield detection
                num_detections += 1

            logger.info('Processing complete. Found %s detections.', num_detections)
        except Exception:
            logger.exception('Failed to complete job due to the following exception:')
            raise


    def get_detections_from_video_capture(self, video_job, video_capture):
        try:
            logger.info('Received video job: %s', video_job)

            num_tracks = 0
            detections = JobRunner(video_job.job_properties).get_indexed_detections(video_capture)
            for idx, detection in detections:
                # Convert each mpf.ImageLocation in to an mpf.VideoTrack with a single detection.
                yield mpf.VideoTrack(idx, idx, -1, {idx: detection}, detection.detection_properties)
                num_tracks += 1

            logger.info('Processing complete. Found %s tracks.', num_tracks)
        except Exception:
            logger.exception('Failed to complete job due to the following exception:')
            raise


class JobRunner(object):
    """ Class process a single job and hold its configuration info. """

    def __init__(self, job_properties):
        self._acs_url = self.get_acs_url(job_properties)
        subscription_key = self._get_acs_property_or_env_value('ACS_SUBSCRIPTION_KEY', job_properties)
        self._acs_headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                             'Content-Type': 'application/octet-stream'}
        self._http_retry = mpf_util.HttpRetry.from_properties(job_properties, logger.warning)

        merge_regions_enabled = mpf_util.get_property(job_properties, 'MERGE_REGIONS', False)
        # Select the merge mode for this job
        if merge_regions_enabled:
            self._results_post_processor = OcrResultsProcessor.merged_results
        else:
            self._results_post_processor = OcrResultsProcessor.unmerged_results


    def get_indexed_detections(self, frames):
        """
        :param frames: An iterable of frames in a numpy.ndarray with shape (h, w, 3)
        :return: An iterable of (frame_number, mpf.ImageLocation)
        """
        encoder = FrameEncoder()

        for idx, frame in enumerate(frames):
            # Encode frame in a format compatible with ACS API. This may result in the frame size changing.
            encoded_frame, resized_frame_dimensions = encoder.resize_and_encode(frame)
            ocr_results_json = self._post_to_acs(encoded_frame)
            initial_frame_size = mpf_util.Size.from_frame(frame)

            # Convert the results from the ACS JSON to mpf.ImageLocations
            detections = self._results_post_processor(ocr_results_json, initial_frame_size, resized_frame_dimensions)
            for detection in detections:
                yield idx, detection


    def _post_to_acs(self, encoded_frame):
        request = urllib.request.Request(self._acs_url, bytes(encoded_frame), self._acs_headers)
        with self._http_retry.urlopen(request) as response:
            return json.load(response)

    @classmethod
    def get_acs_url(cls, job_properties):
        """ Adds query string parameters to the ACS URL if needed. """
        url = cls._get_acs_property_or_env_value('ACS_URL', job_properties)
        language = job_properties.get('LANGUAGE')
        detect_orientation = job_properties.get('DETECT_ORIENTATION', 'true')

        url_parts = urllib.parse.urlparse(url)
        query_dict = urllib.parse.parse_qs(url_parts.query)
        query_dict['detectOrientation'] = detect_orientation
        if language:
            query_dict['language'] = language
        query_string = urllib.parse.urlencode(query_dict, doseq=True)

        return urllib.parse.urlunparse(url_parts._replace(query=query_string))

    @staticmethod
    def _get_acs_property_or_env_value(property_name, job_properties):
        property_value = job_properties.get(property_name)
        if property_value:
            return property_value

        env_value = os.getenv(property_name)
        if env_value:
            return env_value

        raise mpf.DetectionException(
            'The "{}" property must be provided as a job property or environment variable.'.format(property_name),
            mpf.DetectionError.MISSING_PROPERTY)




class OcrResultsProcessor(object):
    """
    Class to convert results from Azure Cognitive Services in to the format used by MPF.
    This involves extracting information from the JSON and converting the bounding boxes in to the format used by MPF.
    """

    # Simplified Chinese, Traditional Chinese, and Japanese do not put spaces between words.
    NO_SPACE_LANGUAGES = frozenset(('zh-Hans', 'zh-Hant', 'ja'))

    @classmethod
    def merged_results(cls, ocr_results_json, initial_frame_size, resized_frame_size):
        if ocr_results_json['regions']:
            return cls(ocr_results_json, initial_frame_size, resized_frame_size).get_merged_results()
        return ()

    @classmethod
    def unmerged_results(cls, ocr_results_json, initial_frame_size, resized_frame_size):
        if ocr_results_json['regions']:
            return cls(ocr_results_json, initial_frame_size, resized_frame_size).get_unmerged_results()
        return ()

    def __init__(self, ocr_results_json, initial_frame_size, resized_frame_size):
        self._results_json = ocr_results_json
        self._text_angle_degrees = math.degrees(float(ocr_results_json.get('textAngle', '0')))

        self._language = ocr_results_json['language']
        self._word_separator = '' if self._language in self.NO_SPACE_LANGUAGES else ' '

        self._initial_frame_size = initial_frame_size
        self._resize_scale_factor = initial_frame_size.width / resized_frame_size.width


        # ACS describes rotation using an "orientation" field and a "textAngle" field. The "orientation" value
        # (up, down, left, or right) refers to the direction that the top of the recognized text is facing, after the
        # image has been rotated around its center according to the detected text angle ("textAngle" property).
        # Each orientation uses a slightly different coordinate system so each case needs to be handled separately.
        # ACS only reports a single "orientation" field for the entire frame, so we can pre-select the
        # orientation correction method here.
        orientation = ocr_results_json.get('orientation', 'Up')
        if orientation == 'Up':
            self._orientation_correction = self._correct_up_oriented
        elif orientation == 'Down':
            self._orientation_correction = self._correct_down_oriented
        elif orientation == 'Left':
            self._orientation_correction = self._correct_left_oriented
        elif orientation == 'Right':
            self._orientation_correction = self._correct_right_oriented
        else:
            raise NotImplementedError('Unsupported orientation: ' + orientation)

        frame_center = initial_frame_size.width / 2, initial_frame_size.height / 2
        # ACS only reports a single "textAngle" field for the entire frame, so we can create the rotation matrix
        # here once and re-use it for every region.
        self._rotation_mat = cv2.getRotationMatrix2D(frame_center, self._text_angle_degrees, 1)


    def get_merged_results(self):
        regions = iter(self._results_json['regions'])
        try:
            region = next(regions)
            bounding_box = mpf_util.Rect(*(int(x) for x in region['boundingBox'].split(',')))
            text_builder = [self._get_region_text(region)]
        except StopIteration:
            return ()

        for region in regions:
            bounding_box = bounding_box.union([int(x) for x in region['boundingBox'].split(',')])
            text_builder.append(self._get_region_text(region))

        text = '\n\n'.join(text_builder)
        return (self._create_image_location(text, bounding_box),)


    def get_unmerged_results(self):
        regions = self._results_json['regions']
        for region in regions:
            text = self._get_region_text(region)
            bounding_box = mpf_util.Rect(*(int(x) for x in region['boundingBox'].split(',')))
            yield self._create_image_location(text, bounding_box)


    def _create_image_location(self, text, bounding_box):
        corrected_bounding_box, angle = self._correct_region_bounding_box(bounding_box)

        detection_properties = dict(TEXT=text, TEXT_LANGUAGE=self._language,
                                    ROTATION=str(mpf_util.normalize_angle(angle)))

        return mpf.ImageLocation(int(corrected_bounding_box.x), int(corrected_bounding_box.y),
                                 int(corrected_bounding_box.width), int(corrected_bounding_box.height),
                                 -1, detection_properties)


    def _correct_region_bounding_box(self, bounding_box):
        """
        Convert ACS representation of bounding box to MPF's bounding box and rotation.

        :param bounding_box: ACS formatted bounding box
        :return: A tuple containing the MPF representation of bounding box and the box's rotation angle in degrees.
        """
        # Map coordinates from resized frame back to the original frame
        x = bounding_box.x * self._resize_scale_factor
        y = bounding_box.y * self._resize_scale_factor
        width = bounding_box.width * self._resize_scale_factor
        height = bounding_box.height * self._resize_scale_factor

        # Map ACS orientation info to MPF rotation info
        corrected_top_left, corrected_angle = self._orientation_correction(x, y)
        corrected_box = mpf_util.Rect.from_corner_and_size(corrected_top_left, (width, height))
        return corrected_box, mpf_util.normalize_angle(corrected_angle)


    def _correct_up_oriented(self, x, y):
        return self._rotate_point((x, y)), self._text_angle_degrees

    def _correct_down_oriented(self, x, y):
        return (self._rotate_point((self._initial_frame_size.width - x, self._initial_frame_size.height - y)),
                self._text_angle_degrees + 180)

    def _correct_left_oriented(self, x, y):
        return self._rotate_point((y, self._initial_frame_size.height - x)), self._text_angle_degrees + 90

    def _correct_right_oriented(self, x, y):
        return self._rotate_point((self._initial_frame_size.width - y, x)), self._text_angle_degrees + 270


    def _get_region_text(self, region):
        """The ACS results are divided in to regions. The regions are divided in to lines.
           The lines are divided in to individual words."""
        lines = region['lines']
        return '\n'.join(self._get_line_text(line) for line in lines)


    def _get_line_text(self, line):
        """"ACS returns each word in the line in its own JSON entry, so we combine the words from a single line
            putting a space between each word when the language isn't Chinese or Japanese. """
        return self._word_separator.join(w['text'] for w in line['words'])

    def _rotate_point(self, point):
        """"Rotate point around center of frame"""
        return np.matmul(self._rotation_mat, (point[0], point[1], 1))




class FrameEncoder(object):
    """Class to handle converting frames in to a format acceptable to Azure Cognitive Services."""

    # These constraints are from Azure Cognitive Services
    # (https://westus.dev.cognitive.microsoft.com/docs/services/56f91f2d778daf23d8ec6739/operations/56f91f2e778daf14a499e1fc)
    MAX_PIXELS = 10000000
    MIN_DIMENSION_LENGTH = 50  # pixels
    MAX_DIMENSION_LENGTH = 4200  # pixels
    MAX_FILE_SIZE = 4 * 1024 * 1024  # bytes

    def __init__(self):
        self._encode = self._default_encode

    def resize_and_encode(self, frame):
        """
        Encodes the given frame as a PNG. Resizes the frame if it is outside ACS's constraints.

        :param frame: A numpy.ndarray containing a frame
        :return: A tuple of (encoded_frame, frame_size)
        """
        initial_frame_size = mpf_util.Size.from_frame(frame)
        corrected_frame_size = self._resize_for_dimensions(frame)
        if initial_frame_size == corrected_frame_size:
            resized_frame = frame
        else:
            resized_frame = cv2.resize(frame, (corrected_frame_size.width, corrected_frame_size.height),
                                       interpolation=cv2.INTER_AREA)
        encoded_frame = self._encode(resized_frame)
        if len(encoded_frame) <= self.MAX_FILE_SIZE:
            return encoded_frame, corrected_frame_size

        return self._resize_for_file_size(frame, corrected_frame_size, len(encoded_frame))


    def _default_encode(self, frame):
        """Encode the given frame using the default OpenCV settings. This is faster than _encode_max_compression,
           but will result in a larger file."""
        success, encoded_frame = cv2.imencode('.png', frame)
        if not success:
            raise mpf.DetectionException('Failed to encode frame.')
        if len(encoded_frame) <= self.MAX_FILE_SIZE:
            return encoded_frame

        # The resulting file was too big when encoded with the default settings, so we switch the encoding method to
        # use the maximum compression.
        self._encode = self._encode_max_compression
        return self._encode_max_compression(frame)

    @staticmethod
    def _encode_max_compression(frame):
        """Encode the frame using the maximum compression settings. This is slower than _default_encode,
        but produces a smaller file."""
        success, encoded_frame = cv2.imencode('.png', frame, (cv2.IMWRITE_PNG_COMPRESSION, 9))
        if not success:
            raise mpf.DetectionException('Failed to encode frame.')
        return encoded_frame


    @classmethod
    def _resize_for_file_size(cls, frame, corrected_frame_size, previous_encode_length):
        """
        Handles the case when the encoded frame contains more bytes than ACS allows.

        :param frame: A numpy.ndarray containing a frame
        :param corrected_frame_size: The frame dimensions from the previous encode attempt.
        :param previous_encode_length: The number of bytes from the previous encode attempt.
        :return: A tuple containing the encoded_frame and its dimensions
        """
        corrected_frame_size = scale_size(
            corrected_frame_size,
            math.sqrt(cls.MAX_FILE_SIZE / previous_encode_length))

        resized_frame = cv2.resize(frame, (corrected_frame_size.width, corrected_frame_size.height),
                                   interpolation=cv2.INTER_AREA)

        encoded_frame = cls._encode_max_compression(resized_frame)
        if len(encoded_frame) <= cls.MAX_FILE_SIZE:
            logger.warning(
                'Downsampling frame because Azure Cognitive Services requires the encoded frame to be at most '
                '%s bytes, but it was too big.', cls.MAX_FILE_SIZE)
            return encoded_frame, corrected_frame_size

        # Recurse to retry with smaller size. This should almost never happen.
        # It was only observed with specific test images that were designed to be hard to compress.
        return cls._resize_for_file_size(frame, corrected_frame_size, len(encoded_frame))


    @classmethod
    def _resize_for_dimensions(cls, frame):
        """
        Handles the case when the frame has more pixels than ACS allows. ACS limits both the number of pixels per
        dimension and the total number of pixels.

        :param frame: A numpy.ndarray containing a frame
        :return: A new frame size that maintains the original aspect ratio and is within the ACS limits.
        """
        original_frame_size = mpf_util.Size.from_frame(frame)
        new_frame_size = original_frame_size

        min_dimension = min(new_frame_size.width, new_frame_size.height)
        max_dimension = max(new_frame_size.width, new_frame_size.height)

        if min_dimension < cls.MIN_DIMENSION_LENGTH and max_dimension > cls.MAX_DIMENSION_LENGTH:
            raise mpf.DetectionException(
                'Unable to resize frame with size of {}x{} to an acceptable size because one dimension is under the '
                'minimum number of pixels per dimension ({}) and the other is over the maximum ({}).'.format(
                    original_frame_size.width, original_frame_size.height, cls.MIN_DIMENSION_LENGTH,
                    cls.MAX_DIMENSION_LENGTH),
                mpf.DetectionError.BAD_FRAME_SIZE)

        if min_dimension < cls.MIN_DIMENSION_LENGTH:
            new_frame_size = scale_size(new_frame_size, cls.MIN_DIMENSION_LENGTH / min_dimension)
            if max(new_frame_size.width, new_frame_size.height) > cls.MAX_DIMENSION_LENGTH:
                raise mpf.DetectionException(
                    'Unable to resize frame with size of {}x{} to an acceptable size because one dimension was '
                    'below the minimum number of pixels per dimension ({}) and upsampling the frame will result in '
                    'the other dimension being over the maximum ({}).'.format(
                        original_frame_size.width, original_frame_size.height, cls.MIN_DIMENSION_LENGTH,
                        cls.MAX_DIMENSION_LENGTH),
                    mpf.DetectionError.BAD_FRAME_SIZE)

            logger.warning(
                'Upsampling frame because Azure Cognitive Services requires both dimensions to be at least %s pixels, '
                'but the frame was %sx%s.',
                cls.MIN_DIMENSION_LENGTH, original_frame_size.width, original_frame_size.height)

        if max_dimension > cls.MAX_DIMENSION_LENGTH:
            new_frame_size = scale_size(new_frame_size, cls.MAX_DIMENSION_LENGTH / max_dimension)
            if min(new_frame_size.width, new_frame_size.height) < cls.MIN_DIMENSION_LENGTH:
                raise mpf.DetectionException(
                    'Unable to resize frame with size of {}x{} to an acceptable size because one dimension was '
                    'above the maximum number of pixels per dimension ({}) and downsampling the frame will result in '
                    'the other dimension being under the minimum ({}).'.format(
                        original_frame_size.width, original_frame_size.height, cls.MAX_DIMENSION_LENGTH,
                        cls.MIN_DIMENSION_LENGTH),
                    mpf.DetectionError.BAD_FRAME_SIZE)
            logger.warning(
                'Downsampling frame because Azure Cognitive Services requires both dimensions to be at most %s pixels, '
                'but the frame was %sx%s.',
                cls.MAX_DIMENSION_LENGTH, original_frame_size.width, original_frame_size.height)

        new_pixel_count = new_frame_size.width * new_frame_size.height
        if new_pixel_count > cls.MAX_PIXELS:
            new_frame_size = scale_size(new_frame_size, math.sqrt(cls.MAX_PIXELS / new_pixel_count))
            logger.warning(
                'Downsampling frame because Azure Cognitive Services requires the frame to contain at most %s pixels, '
                'but the frame contained %s pixels.',
                cls.MAX_PIXELS, original_frame_size.width * original_frame_size.height)

        return new_frame_size


def scale_size(size, scale_factor):
    """Multiplies the width and height of the given size by scale_factor.
        Useful for changing a size while maintaining the aspect ratio."""
    return mpf_util.Size(int(size.width * scale_factor), int(size.height * scale_factor))


