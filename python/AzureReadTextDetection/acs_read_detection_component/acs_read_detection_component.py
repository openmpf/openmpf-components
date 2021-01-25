#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2021 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2021 The MITRE Corporation                                      #
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
import math
import os
import time
import urllib
import urllib.error
import urllib.parse
import urllib.request

import cv2
import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = mpf.configure_logging('acs-read-detection.log', __name__ == '__main__')


class AcsReadDetectionComponent(mpf_util.VideoCaptureMixin, mpf_util.ImageReaderMixin, object):
    detection_type = 'TEXT'

    def get_detections_from_generic(self, generic_job):
        try:
            logger.info('[%s] Received generic job: %s', generic_job.job_name, generic_job)
            detections = JobRunner(generic_job.job_name,
                                   generic_job.job_properties,
                                   False).get_pdf_detections(generic_job.data_uri)
            num_detections = len(detections)
            logger.info('[%s] Processing complete. Found %s detections.', generic_job.job_name, num_detections)

        except Exception:
            logger.exception('[%s] Failed to complete job due to the following exception:', generic_job.job_name)
            raise

        return detections

    def get_detections_from_image_reader(self, image_job, image_reader):
        try:
            logger.info('[%s] Received image job: %s', image_job.job_name, image_job)

            num_detections = 0
            image = image_reader.get_image()
            detections = JobRunner(image_job.job_name, image_job.job_properties, True).get_image_detections(image)
            for detection in detections:
                yield detection
                num_detections += 1
            logger.info('[%s] Processing complete. Found %s detections.', image_job.job_name, num_detections)
        except Exception:
            logger.exception('[%s] Failed to complete job due to the following exception:', image_job.job_name)
            raise



    def get_detections_from_video_capture(self, video_job, video_capture):
        try:
            logger.info('[%s] Received video job: %s', video_job.job_name, video_job)

            num_tracks = 0

            for frame_idx, frame in enumerate(video_capture):
                detections = JobRunner(video_job.job_name, video_job.job_properties, True).get_image_detections(frame,
                                                                                                                True)
                for detection in detections:
                     yield mpf.VideoTrack(frame_idx, frame_idx, -1, {frame_idx: detection}, detection.detection_properties)
                     num_tracks += 1

            logger.info('[%s] Processing complete. Found %s tracks.', video_job.job_name, num_tracks)
        except Exception:
            logger.exception('[%s] Failed to complete job due to the following exception:', video_job.job_name)
            raise

class JobRunner(object):
    """ Class process a single job and hold its configuration info. """

    def __init__(self, job_name, job_properties, is_image):
        self._job_name = job_name
        self._acs_url = self.get_acs_url(job_properties)
        self._is_image = is_image
        self._merge_lines = mpf_util.get_property(job_properties, 'MERGE_LINES', True)
        self._max_attempts = mpf_util.get_property(job_properties, 'MAX_GET_READ_RESULT_ATTEMPTS', -1)
        subscription_key = self._get_acs_property_or_env_value('ACS_SUBSCRIPTION_KEY', job_properties)

        if self._is_image:
            self._acs_headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                                 'Content-Type': 'application/octet-stream'}
        else:
            # Only supports PDF files.
            self._acs_headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                                 'Content-Type': 'application/pdf'}


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

    def get_image_detections(self, frame, is_video=False):
        """
        :param frame: A numpy.ndarray with shape (h, w, 3)
        :return: An iterable of mpf.ImageLocation
        """
        encoder = FrameEncoder()

        # Encode frame in a format compatible with ACS API. This may result in the frame size changing.
        encoded_frame, resized_frame_dimensions = encoder.resize_and_encode(frame)
        read_results_json = self._post_to_acs(encoded_frame)
        initial_frame_size = mpf_util.Size.from_frame(frame)

        resize_scale_factor = initial_frame_size.width / resized_frame_dimensions.width

        # Convert the results from the ACS JSON to mpf.ImageLocations
        detections = ReadResultsProcessor(self._is_image,
                                          resize_scale_factor,
                                          self._merge_lines).process_read_results(read_results_json, is_video)
        return detections
            
    def get_pdf_detections(self, data_uri):
        with open(data_uri, 'rb') as f:
            pdf_content = f.read()
        
        read_results_json = self._post_to_acs(pdf_content)
        return ReadResultsProcessor(self._is_image, 1, self._merge_lines).process_read_results(read_results_json)

    def _get_acs_result(self, result_request):
        logger.info('[%s] Contacting ACS server for read results.', self._job_name)
        attempt_num = 0
        wait_sec = 5
        while attempt_num < self._max_attempts or self._max_attempts <= 0:
            logger.info('[%s] Attempting to retrieve layout Analysis results from %s',
                        self._job_name, result_request.get_full_url())

            try:
                results = urllib.request.urlopen(result_request)
                json_results = json.load(results)
                if results.status != 200:
                    raise mpf.DetectionError.DETECTION_FAILED.exception('GET request failed with HTTP status {}\n'
                                                                        .format(results.status) +
                                                                        '\nHTTP request body:\n{}'.format(json_results))
                status = json_results["status"]
                if status == "succeeded":
                    logger.info('[%s] Layout Analysis succeeded. Processing read results.', self._job_name)
                    return json_results
                if status == "failed":
                    raise mpf.DetectionError.DETECTION_FAILED.exception('GET request failed. HTTP request body:\n{}'
                                                                        .format(json_results))
                # Analysis still running. Wait and then retry GET request.
                logger.info('[%s] Layout Analysis results not available yet. Recontacting server in %s seconds.',
                            self._job_name, wait_sec)

                time.sleep(wait_sec)
                attempt_num += 1
            except urllib.error.HTTPError as e:
                response_content = e.read().decode('utf-8', errors='replace')
                raise mpf.DetectionException('GET Request failed with HTTP status {} and message: {}'
                                             .format(e.code, response_content), mpf.DetectionError.DETECTION_FAILED)
        logger.error('[%s] Max attempts exceeded. Unable to retrieve server results. Ending job.', self._job_name)
        raise mpf.DetectionException('Unable to retrieve results from ACS server.')

    def _post_to_acs(self, encoded_frame):
        logger.info('[%s] Sending POST request to %s', self._job_name, self._acs_url)
        request = urllib.request.Request(self._acs_url, bytes(encoded_frame), self._acs_headers)
        try:
            response = urllib.request.urlopen(request)
            results_url = response.headers['operation-location']
            result_request = urllib.request.Request(results_url, None, self._acs_headers)
            logger.info('[%s] Received response from ACS server. Obtained read results url.', self._job_name)
            return self._get_acs_result(result_request)
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionException('Request failed with HTTP status {} and message: {}'
                                         .format(e.code, response_content), mpf.DetectionError.DETECTION_FAILED)

    @classmethod
    def get_acs_url(cls, job_properties):
        """ Adds query string parameters to the ACS URL if needed. """
        url = cls._get_acs_property_or_env_value('ACS_URL', job_properties)
        language = job_properties.get('LANGUAGE')
        url_parts = urllib.parse.urlparse(url)
        query_dict = urllib.parse.parse_qs(url_parts.query)
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


class FrameEncoder(object):
    """Class to handle converting frames in to a format acceptable to Azure Cognitive Services."""

    # These constraints are from Azure Cognitive Services
    # (https://docs.microsoft.com/en-us/azure/cognitive-services/computer-vision/concept-recognizing-text#input-requirements)
    # Please note that the following constraints follow the paid tier format.

    MIN_DIMENSION_LENGTH = 50  # pixels
    MAX_DIMENSION_LENGTH = 10000  # pixels
    MAX_FILE_SIZE = 50 * 1024 * 1024  # bytes

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

        return new_frame_size


def scale_size(size, scale_factor):
    """Multiplies the width and height of the given size by scale_factor.
        Useful for changing a size while maintaining the aspect ratio."""
    return mpf_util.Size(int(size.width * scale_factor), int(size.height * scale_factor))


class ReadResultsProcessor(object):
    """
    Class to convert results from Azure Cognitive Services in to the format used by MPF.
    This involves extracting information from the JSON and converting the bounding boxes in to the format used by MPF.
    """
    def __init__(self, is_image, resize_scale_factor, merge_lines):
        self._is_image = is_image
        self._resize_scale_factor = resize_scale_factor
        self._merge_lines = merge_lines

    def _create_detection(self, detection_properties, bounding_box):
        if self._is_image:
            if len(bounding_box) > 0:
                corrected_bounding_box = self._correct_region_bounding_box(bounding_box)
                return mpf.ImageLocation(int(corrected_bounding_box.x), int(corrected_bounding_box.y),
                                         int(corrected_bounding_box.width), int(corrected_bounding_box.height),
                                         -1, detection_properties)
            else:
                return mpf.ImageLocation(0, 0, 0, 0, -1, detection_properties)
        else:
            return mpf.GenericTrack(-1, detection_properties)

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
        top_left = (x, y)
        corrected_box = mpf_util.Rect.from_corner_and_size(top_left, (width, height))
        return corrected_box

    def _convert_to_rect(self, bonding_box):
        p1 = (bonding_box[0], bonding_box[1])
        p2 = (bonding_box[4], bonding_box[5])
        return mpf_util.Rect.from_corners(p1, p2)


    def process_read_results(self, read_results_json, is_video = False):
        detections = []
        # Extract text results.
        if 'readResults' in read_results_json['analyzeResult']:
            read_results = read_results_json['analyzeResult']['readResults']
            for page in read_results:
                page_num = page['page']
                if 'lines' in page:
                    line_num = 1
                    lines = []
                    bounding_box = []

                    for line in page['lines']:
                        if not self._merge_lines:
                            detection_properties = dict(OUTPUT_TYPE='LINE',
                                                        LINE_NUM=str(line_num),
                                                        TEXT=line['text'])
                            if not(is_video):
                                detection_properties["PAGE_NUM"] = str(page_num)


                            if self._is_image:
                                bounding_box = self._convert_to_rect(line['boundingBox'])
                            detections.append(self._create_detection(detection_properties, bounding_box))
                            line_num += 1
                        else:
                            lines.append(line['text'])

                            if self._is_image:
                                if len(bounding_box) == 0:
                                    bounding_box = self._convert_to_rect(line['boundingBox'])
                                else:
                                    second_bounding_box = self._convert_to_rect(line['boundingBox'])
                                    bounding_box = bounding_box.union(second_bounding_box)

                    if self._merge_lines and len(lines) > 0:
                        detection_properties = dict(OUTPUT_TYPE='MERGED_LINES',
                                                    TEXT='\n'.join(lines))
                        if not(is_video):
                            detection_properties["PAGE_NUM"] = str(page_num)
                        detections.append(self._create_detection(detection_properties, bounding_box))

        return detections



    def _remove_elements(self, json_object, bounding_box_list):
        if 'fields' in json_object:
            for key in json_object['fields']:
                self._remove_elements(json_object['fields'][key], bounding_box_list)
        if 'type' in json_object:
            if json_object['type'] == 'array':
                for obj in json_object['valueArray']:
                    self._remove_elements(obj, bounding_box_list)
            if json_object['type'] == 'object':
                for key in json_object['valueObject']:
                    self._remove_elements((json_object['valueObject'][key]), bounding_box_list)

        box = json_object.pop('boundingBox', None)
        json_object.pop('elements', None)
        if box is not None:
            bounding_box_list.append(box)


