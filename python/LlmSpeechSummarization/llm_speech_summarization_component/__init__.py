#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2025 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2025 The MITRE Corporation                                      #
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

from .llm_speech_summarization_component import LlmSpeechSummaryComponent, JobConfig

from .schema import StructuredResponseClassFactory

from .llm_util.classifiers import get_classifier_lines, get_classifier_dict
from .llm_util.slapchop import split_csv_into_chunks, summarize_summaries, BOUNDARY_TOKEN_FOR_COUNTING
from .llm_util.input_cleanup import convert_feed_forward_inputs_to_csv