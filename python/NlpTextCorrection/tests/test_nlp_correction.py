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

import unittest
import mpf_component_api as mpf
import os
from nlp_correction_component.nlp_correction_component import NlpCorrectionComponent


class TestNlpCorrection(unittest.TestCase):

    @staticmethod
    def _get_test_file(filename):
        return os.path.join(os.path.dirname(__file__), 'data', filename)

    # test that the component works when passed a text file
    def test_text_file(self):
        job = mpf.GenericJob(
            job_name='test-file',
            data_uri=self._get_test_file("sample.txt"),
            job_properties=dict(),
            media_properties={},
            feed_forward_track=None
        )

        results = list(NlpCorrectionComponent().get_detections_from_generic(job))
        self.assertEqual(1, len(results))

        expected_text = "this is some sample text explain is misspelled"
        self.assertEqual(expected_text, results[0].detection_properties.get("CORRECTED TEXT"))

    # test that the component works with a custom dictionary
    def test_custom_dictionary(self):
        job = mpf.GenericJob(
            job_name='test-file',
            data_uri=self._get_test_file("sample.txt"),
            job_properties=dict(CUSTOM_DICTIONARY='sample_dict.txt'),
            media_properties={},
            feed_forward_track=None
        )

        results = list(NlpCorrectionComponent().get_detections_from_generic(job))
        self.assertEqual(1, len(results))

        expected_text = "this is some sample text explan is misspelled"
        self.assertEqual(expected_text, results[0].detection_properties.get("CORRECTED TEXT"))
