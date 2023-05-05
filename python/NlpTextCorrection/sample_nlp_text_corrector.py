#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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

import sys

from nlp_correction_component import NlpCorrectionComponent
import mpf_component_api as mpf


def main():
    # custom dictionary can just be ""
    if len(sys.argv) == 3:
        _, file, custom_dictionary = sys.argv
        job_props = dict(CUSTOM_DICTIONARY=custom_dictionary)
    elif len(sys.argv) == 2:
        _, file = sys.argv
        job_props = {}
    else:
        sys.exit('Error: Invalid number of arguments. \n' 
                 'Usage: {} <file_path> or '
                 '{} <file_path> <custom_dictionary_path>'
                 .format(sys.argv[0], sys.argv[0]))

    component = NlpCorrectionComponent()

    job = mpf.GenericJob("Sample job", file, job_props, {}, None)

    detections = list(component.get_detections_from_generic(job))
    print(detections)


if __name__ == '__main__':
    main()
