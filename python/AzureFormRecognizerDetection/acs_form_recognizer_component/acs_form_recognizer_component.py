#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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

import collections
import csv
import io
import json
import math
import os
import re
import urllib
import urllib.error
import urllib.parse
import urllib.request
import ssl
import time

import cv2
import numpy as np

import mpf_component_api as mpf
import mpf_component_util as mpf_util

logger = mpf.configure_logging('acs-form-recognizer-detection.log', __name__ == '__main__')

class AcsFormRecognizerComponent(mpf_util.ImageReaderMixin, object):
    detection_type = 'TEXT'

    def get_detections_from_generic(self, generic_job):
        try:
            logger.info('[%s] Received generic job: %s', generic_job.job_name, generic_job)
            detections = JobRunner(generic_job.job_properties, False).get_pdf_detections(generic_job.data_uri)
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
            detections = JobRunner(image_job.job_properties, True).get_image_detections((image,))
            for detection in detections:
                # We don't need to anything with the index for a image jobs.
                yield detection
                num_detections += 1
            logger.info('[%s] Processing complete. Found %s detections.', image_job.job_name, num_detections)
        except Exception:
            logger.exception('[%s] Failed to complete job due to the following exception:', image_job.job_name)
            raise

class JobRunner(object):
    """ Class process a single job and hold its configuration info. """

    def __init__(self, job_properties, is_image):
        self._acs_url = self.get_acs_url(job_properties)
        self._is_image = is_image
        self._merge_lines = json.loads(job_properties.get("MERGE_LINES", "true").lower())
        subscription_key = self._get_acs_property_or_env_value('ACS_SUBSCRIPTION_KEY', job_properties)


        if (self._is_image):
            self._acs_headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                                 'Content-Type': 'application/octet-stream'}
        else:
            # Only supports PDF files.
            self._acs_headers = {'Ocp-Apim-Subscription-Key': subscription_key,
                                 'Content-Type': 'application/pdf'}
        self._text_tagger = TextTagger(job_properties)

    def create_detection(self, detection_properties):
        if self._is_image:
            return mpf.ImageLocation(0, 0, 0, 0, -1, detection_properties)
        else:
            return mpf.GenericTrack(-1, detection_properties)

    # Uses results_post_processor as a reference.
    def _process_image_results(self, form_results_json):

        # Extract text results.
        page_num = 0
        detections = []
        for page in form_results_json['analyzeResult']['readResults']:
            page_num = page['page'] - 1
            line_num = 0
            lines = []
            for line in page["lines"]:
                if not self._merge_lines:
                    detection_properties = dict(OUTPUT_TYPE = "LINE",
                                                PAGE_NUM = str(page_num),
                                                LINE_NUM = str(line_num),
                                                TEXT = line["text"])
                    detections.append(self.create_detection(detection_properties))
                    line_num += 1
                else:
                    lines.append(line["text"])
            if self._merge_lines:
                detection_properties = dict(OUTPUT_TYPE = "MERGED_LINES",
                                            PAGE_NUM = str(page_num),
                                            TEXT = "\n".join(lines))
                detections.append(self.create_detection(detection_properties))

        # Extract table results.
        for page in form_results_json['analyzeResult']['pageResults']:
            page_num = page['page'] - 1
            table_num = 0
            for table in page['tables']:
                row = table['rows']
                col = table['columns']
                table_array = np.full([row, col], None)
                for cell in table['cells']:
                    table_array[cell['rowIndex']][cell['columnIndex']] = cell['text']
                output = io.StringIO()
                writer = csv.writer(output)
                for row in table_array:
                    writer.writerow(row)
                cs_output = output.getvalue()
                output.close()
                print("Adding table result")
                detection_properties = dict(OUTPUT_TYPE = "TABLE",
                                            PAGE_NUM=str(page_num),
                                            TABLE_NUM = str(table_num),
                                            TABLE_CSV_OUTPUT=cs_output)
                detections.append(self.create_detection(detection_properties))
                table_num += 1

        return detections

    def get_pdf_detections(self, data_uri):
        pdf_content = None
        with open(data_uri, 'rb') as f:
            pdf_content = f.read()
        form_results_json = self._post_to_acs(pdf_content)

        return self._process_image_results(form_results_json)



    def get_image_detections(self, frames):
        """
        :param frames: An iterable of frames in a numpy.ndarray with shape (h, w, 3)
        :return: An iterable of (frame_number, mpf.ImageLocation)
        """
        encoder = FrameEncoder()

        for idx, frame in enumerate(frames):
            # Encode frame in a format compatible with ACS API. This may result in the frame size changing.
            encoded_frame, resized_frame_dimensions = encoder.resize_and_encode(frame)
            form_results_json = self._post_to_acs(encoded_frame)
            initial_frame_size = mpf_util.Size.from_frame(frame)

            # Convert the results from the ACS JSON to mpf.ImageLocations
            # TODO: Add self._text_tagger to process LINE results.
            detections = self._process_image_results(form_results_json)
            for detection in detections:
                yield detection

    def _post_to_acs(self, encoded_frame):
        request = urllib.request.Request(self._acs_url, bytes(encoded_frame), self._acs_headers)
        try:
            response = urllib.request.urlopen(request)
            results_url = response.headers["operation-location"]
            result_request = urllib.request.Request(results_url, None, self._acs_headers)
            results = urllib.request.urlopen(result_request)
            return json.load(results)
        except urllib.error.HTTPError as e:
            response_content = e.read().decode('utf-8', errors='replace')
            raise mpf.DetectionException('Request failed with HTTP status {} and message: {}'
                                         .format(e.code, response_content), mpf.DetectionError.DETECTION_FAILED)

    @classmethod
    def get_acs_url(cls, job_properties):
        """ Adds query string parameters to the ACS URL if needed. """
        url = cls._get_acs_property_or_env_value('ACS_URL', job_properties)
        include_text_details = job_properties.get('INCLUDE_TEXT_DETAILS', 'true')
        url_parts = urllib.parse.urlparse(url)
        query_dict = urllib.parse.parse_qs(url_parts.query)
        query_dict['includeTextDetails'] = include_text_details
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


class TextTagger(object):
    def __init__(self, job_properties):
        tags_json_file = self._get_tags_json_path(job_properties)
        if tags_json_file:
            self._tags_to_compiled_regexes = self._load_tag_dict(tags_json_file)
        else:
            self._tags_to_compiled_regexes = {}

        self.tagging_enabled = len(self._tags_to_compiled_regexes) > 0


    def find_tags(self, content):
        """
        Searches the given content for text matching the loaded tags.

        :param content: The text to search for matches.
        :return: A unicode string containing a comma-separated list of matching tags.
        """
        matching_tags = (tag for tag, rxs in self._tags_to_compiled_regexes.items()
                         if self._any_regex_matches(content, rxs))
        return ', '.join(matching_tags)


    @staticmethod
    def _any_regex_matches(content, regexes):
        return any(r.search(content) for r in regexes)


    @staticmethod
    def _get_tags_json_path(job_properties):
        tagging_file = job_properties.get('TAGGING_FILE')
        if not tagging_file:
            return None

        expanded_tagging_file = os.path.expandvars(tagging_file)

        absolute_path = os.path.abspath(expanded_tagging_file)
        if os.path.exists(absolute_path):
            return absolute_path

        internal_path = os.path.abspath(os.path.join(os.path.dirname(__file__), expanded_tagging_file))
        if os.path.exists(internal_path):
            return internal_path

        if absolute_path == internal_path:
            error_msg = 'Tagging file not found. Expected it to exist at "{}".'.format(absolute_path)
        else:
            error_msg = 'Tagging file not found. Expected it to exist at either "{}" or "{}".'.format(
                absolute_path, internal_path)

        raise mpf.DetectionException(error_msg, mpf.DetectionError.COULD_NOT_READ_DATAFILE)


    @classmethod
    def _load_tag_dict(cls, tags_json_file):
        """
        Loads text tags from a JSON file.

        :param tags_json_file: Path to file containing the text tags JSON.
        :return: A dictionary where the key the tag and the value is a list of compiled regexes.
        :rtype: dict[unicode, list[re.RegexObject]]
        """
        with open(tags_json_file) as f:
            try:
                tags_json = json.load(f)
            except ValueError as e:
                raise mpf.DetectionException(
                    'Failed to load text tags JSON from \"{}\" due to: {}'.format(tags_json_file, e),
                    mpf.DetectionError.COULD_NOT_READ_DATAFILE)

        tags_to_compiled_regexes = collections.defaultdict(list)
        regex_flags = re.MULTILINE | re.IGNORECASE | re.UNICODE | re.DOTALL

        tags_by_regex = tags_json.get('TAGS_BY_REGEX', {})
        for tag, regexes in tags_by_regex.items():
            tags_to_compiled_regexes[tag] = [re.compile(r, regex_flags) for r in regexes]

        tags_by_keyword = tags_json.get('TAGS_BY_KEYWORD', {})
        for tag, keywords in tags_by_keyword.items():
            compiled_regexes = (cls._convert_keyword_to_regex(w, regex_flags) for w in keywords)
            tags_to_compiled_regexes[tag].extend(compiled_regexes)

        return tags_to_compiled_regexes


    _SLASH_THEN_WHITE_SPACE_REGEX = re.compile(r'\\\s', re.UNICODE)

    @classmethod
    def _convert_keyword_to_regex(cls, word, regex_flags):
        # Converts 'hello world' to r'\bhello\s+world\b'.

        # re.escape doesn't just escape regex metacharacters, it escapes all non-printable and non-ASCII text.
        # This causes space characters to also be escaped which makes it harder to replace them with '\s+'
        escaped = re.escape(word)
        # word = 'hello world'
        # escaped = 'hello\ world'

        with_generic_white_space = cls._SLASH_THEN_WHITE_SPACE_REGEX.sub(r'\\s+', escaped)
        # with_generic_white_space = 'hello\s+world'

        with_word_boundaries = r'\b' + with_generic_white_space + r'\b'
        # with_word_boundaries = '\bhello\s+world\b'

        return re.compile(with_word_boundaries, regex_flags)


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


