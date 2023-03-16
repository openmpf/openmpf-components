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

from whisper_detection_component import WhisperDetectionWrapper


def main():
    if len(sys.argv) != 3:
        sys.exit(f'Usage {sys.argv[0]} <audio_file> <whisper_mode>')

    _, audio_file, whisper_mode = sys.argv


    whisper_mode = int(whisper_mode)

    job_props = {"WHISPER_MODE": whisper_mode}


    audio_tracks = WhisperDetectionWrapper().process_audio(audio_file, 0, 0, job_props)
    detection_props = audio_tracks[0].detection_properties

    if whisper_mode == 0:
        print('DETECTED LANGUAGE:', detection_props['DETECTED_LANGUAGE'])
        print('DETECTED LANGUAGE CONFIDENCE:', audio_tracks[0].confidence)
    elif whisper_mode == 1:
        print('DECODED LANGUAGE:', detection_props['DECODED_LANGUAGE'])
        print('TRANSCRIPT:', detection_props['TRANSCRIPT'])
    elif whisper_mode == 2:
        print('DECODED LANGUAGE:', detection_props['DECODED_LANGUAGE'])
        print('TRANSCRIPT:', detection_props['TRANSCRIPT'])
        print('TRANSLATED AUDIO', detection_props['TRANSLATED_AUDIO'])


if __name__ == '__main__':
    main()
