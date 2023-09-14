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

from transformer_tagging_component import TransformerTaggingComponent, TransformerWrapper
import mpf_component_api as mpf


def main():
    wrapper = TransformerWrapper({})
    detection_props = dict(TEXT="I also have a knife. I have a gun. I took a plane to Florida. I bought some cocaine. "
                                "It did not go well.")
    print(detection_props["TEXT"])
    wrapper.add_tags(detection_props)

    for prop in detection_props:
        print(prop, ": ", detection_props[prop])


if __name__ == '__main__':
    main()