#!/usr/bin/env python3

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

from argos_translation_component import TranslationWrapper


def main():
    if len(sys.argv) != 3:
        sys.exit(f'Usage {sys.argv[0]} <from_lang> <text>')

    _, from_lang, text = sys.argv

    detection_props = dict(TEXT=text)
    job_props = dict(DEFAULT_SOURCE_LANGUAGE=from_lang)
    TranslationWrapper(job_props).add_translations(detection_props)

    print('TRANSLATION SOURCE LANGUAGE:', detection_props['TRANSLATION_SOURCE_LANGUAGE'])
    print('TRANSLATION:')
    print(detection_props['TRANSLATION'])


if __name__ == '__main__':
    main()