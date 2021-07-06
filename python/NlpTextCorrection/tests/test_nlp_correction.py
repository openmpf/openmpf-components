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

import logging
import unittest
import mpf_component_api as mpf
import os
from nlp_correction_component.nlp_correction_component import NlpCorrectionComponent

logging.basicConfig(level=logging.DEBUG)


class TestNlpCorrection(unittest.TestCase):

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), filename)

    # test that the component works when passed a text file
    def test_text_file(self):
        job = mpf.GenericJob(
            job_name='test-file',
            data_uri=self._get_test_file("data/sample.txt"),
            job_properties={},
            media_properties={},
            feed_forward_track=None
        )

        results = list(NlpCorrectionComponent().get_detections_from_generic(job))
        self.assertEqual(1, len(results))

        expected_text = "this is some sample text explain is misspelled"
        self.assertEqual(expected_text, results[0].detection_properties.get("CORRECTED TEXT"))

    # test that the component works with a custom dictionary
    def test_custom_dictionary(self):
        custom_dictionary_path = self._get_test_file('sample_dict.dic')

        job = mpf.GenericJob(
            job_name='test-file',
            data_uri=self._get_test_file("data/sample.txt"),
            job_properties=dict(CUSTOM_DICTIONARY=custom_dictionary_path),
            media_properties={},
            feed_forward_track=None
        )

        results = list(NlpCorrectionComponent().get_detections_from_generic(job))
        self.assertEqual(1, len(results))

        expected_text = "this is some sample text explane is misspelled"
        self.assertEqual(expected_text, results[0].detection_properties.get("CORRECTED TEXT"))

    def test_preservation_of_punctuation(self):
        job = mpf.GenericJob(
            job_name='test-file',
            data_uri=self._get_test_file("data/sample_newlines.txt"),
            job_properties={},
            media_properties={},
            feed_forward_track=None
        )

        expected_text = (
            "This is to test that punctuation is preserved.\n"
            "Single newline characters are not lost, and neither are \"quotation marks\" and (parenthesis.)\n"
            "Is punctuation preserved with \"these?\" They should be!\n\nTwo or more newlines are "
            "preserved, here's a misspelling for good measure.\n\n\n\n\nTesting that larger gaps between blocks of "
            "text are preserved.\n")

        results = list(NlpCorrectionComponent().get_detections_from_generic(job))
        self.assertEqual(1, len(results))

        self.assertEqual(expected_text, results[0].detection_properties.get("CORRECTED TEXT"))


if __name__ == '__main__':
    unittest.main(verbosity=2)
