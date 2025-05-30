#!/usr/bin/env python3

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

import sys

from acs_translation_component import TranslationClient
from nlp_text_splitter import TextSplitterModel


def main():
    if len(sys.argv) != 5:
        sys.exit(f'Usage {sys.argv[0]} <acs_url> <acs_subscription_key> <to_lang> <text>')

    _, acs_url, acs_subscription_key, to_lang, text = sys.argv

    detection_props = dict(TEXT=text)
    job_props = dict(TO_LANGUAGE=to_lang, ACS_URL=acs_url,
                     ACS_SUBSCRIPTION_KEY=acs_subscription_key)

    wtp_model = TextSplitterModel("wtp-bert-mini", "cpu")
    TranslationClient(job_props, wtp_model).add_translations(detection_props)

    print('TRANSLATION SOURCE LANGUAGE:', detection_props['TRANSLATION SOURCE LANGUAGE'])
    print('TRANSLATION SOURCE LANGUAGE CONFIDENCE:',
          detection_props['TRANSLATION SOURCE LANGUAGE CONFIDENCE'])
    print('TRANSLATION:')
    print(detection_props['TRANSLATION'])

if __name__ == '__main__':
    main()
