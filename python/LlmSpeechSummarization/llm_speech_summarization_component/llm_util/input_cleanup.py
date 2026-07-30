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

from typing import List
import mpf_component_api as mpf

def convert_feed_forward_inputs_to_csv(
        ff_inputs: List[mpf.ImageLocation] 
                   | List[mpf.VideoTrack]
                   | List[mpf.AudioTrack]
                   | List[mpf.GenericTrack]) -> str:
    from csv import DictWriter
    import io
    buffer = io.StringIO()
    writer = DictWriter(buffer, ['speaker_id', 'gender', 'start_timestamp', 'end_timestamp', 'english_text', 'original_language'], delimiter='|')
    writer.writeheader()
    for input in ff_inputs:
        props = input.detection_properties

        text = None
        if 'TRANSLATION' in props:
            text = str(props['TRANSLATION'])
        elif 'TRANSCRIPT' in props:
            text = str(props['TRANSCRIPT'])
        else:
            text = str(props['TEXT'])
        # this is a slight compromise BUT spoken newlines don't exist. If it's one utterance, treat as one line.
        text = text.replace('\n', ' ')
        writer.writerow({
            "speaker_id": props['LONG_SPEAKER_ID'] if 'LONG_SPEAKER_ID' in props else (props['SPEAKER_ID'] if 'SPEAKER_ID' in props else None),
            "gender": props['GENDER'] if 'GENDER' in props else None,
            "start_timestamp": 0, #TODO
            "end_timestamp": 1, #TODO
            "english_text": text,
            "original_language": props['DECODED_LANGUAGE'] if 'DECODED_LANGUAGE' in props else None,
        })
    output = buffer.getvalue()
    del writer
    buffer.close()
    return output