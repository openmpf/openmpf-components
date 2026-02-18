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

import json

def get_classifier_lines(classifier_path, enabled_classifiers='ALL'):
    with open(classifier_path, 'r') as f:
        data = json.load(f)
    is_enabled = lambda _: True
    if enabled_classifiers != 'ALL':
        classifiers_enabled_list = tuple(map(lambda x: x.lower().strip(), enabled_classifiers.split(',')))
        is_enabled = lambda classifier: classifier.lower().strip() in classifiers_enabled_list
    return "\n".join([f"{classifier['Classifier']}: {classifier['Definition']}{(' - Specific Items of Interest: ' + classifier['Items of Interest']) if classifier['Items of Interest'] and len(classifier['Items of Interest']) > 0 else ''}" for classifier in data if is_enabled(classifier['Classifier'])])