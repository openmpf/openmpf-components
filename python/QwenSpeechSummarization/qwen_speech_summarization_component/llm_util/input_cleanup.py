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
from typing import List
import mpf_component_api as mpf

def clean_input_json(input):
    result = {}
    input = json.loads(input)
    for x in ['jobId', 'timeStart', 'timeStop']:
        result[x] = input[x]
    result['media'] = input['media']
    for media in result['media']:
        del media['output']['TRACKS MERGED']
        unused = []
        for i, speech in enumerate(media['output']['SPEECH']):
            # we only want azurespeech
            if 'algorithm' in speech and speech['algorithm'] == 'VISTASPEECH':
                unused.append(i)
                continue
            for track in speech['tracks']:
                # already in trackProperties
                del track['exemplar']['detectionProperties']
                for detection in track['detections']:
                    del detection['detectionProperties']
        tmp = media['output']['SPEECH']
        media['output']['SPEECH'] = [tmp[i] for i in range(0, len(tmp)) if i not in unused]
    return json.dumps(result)

def convert_to_csv(input):
    input = json.loads(input)
    from csv import DictWriter
    import io
    buffer = io.StringIO()
    writer = DictWriter(buffer, ['speaker_id', 'gender', 'start_timestamp', 'end_timestamp', 'english_text', 'original_language'], delimiter='|')
    writer.writeheader()
    for media in input['media']:
        for speech in media['output']['SPEECH']:
            for track in speech['tracks']:
                writer.writerow({
                    "speaker_id": track['trackProperties']['LONG_SPEAKER_ID'] if 'LONG_SPEAKER_ID' in track['trackProperties'] else track['trackProperties']['SPEAKER_ID'],
                    "gender": track['trackProperties']['GENDER'],
                    "start_timestamp": track['startOffsetTime'],
                    "end_timestamp": track['stopOffsetTime'],
                    "english_text": track['trackProperties']['TRANSLATION'] if 'SKIPPED TRANSLATION' not in track['trackProperties'] else track['trackProperties']['TRANSCRIPT'],
                    "original_language": track['trackProperties']['DECODED_LANGUAGE'],
                })
    output = buffer.getvalue()
    del writer
    buffer.close()
    return output

def convert_tracks_to_csv(input: List[mpf.VideoTrack]|List[mpf.AudioTrack]):
    from csv import DictWriter
    import io
    buffer = io.StringIO()
    writer = DictWriter(buffer, ['speaker_id', 'gender', 'start_timestamp', 'end_timestamp', 'english_text', 'original_language'], delimiter='|')
    writer.writeheader()
    for track in input:
        writer.writerow({
            "speaker_id": track.detection_properties['SPEAKER_ID'] if not 'LONG_SPEAKER_ID' in track.detection_properties else track.detection_properties['LONG_SPEAKER_ID'],
            "gender": track.detection_properties['GENDER'],
            "start_timestamp": 0, #TODO
            "end_timestamp": 1, #TODO
            "english_text": track.detection_properties['TRANSLATION'] if 'SKIPPED TRANSLATION' not in track.detection_properties else track.detection_properties['TRANSCRIPT'],
            "original_language": track.detection_properties['DECODED_LANGUAGE'],
        })
    output = buffer.getvalue()
    del writer
    buffer.close()
    return output