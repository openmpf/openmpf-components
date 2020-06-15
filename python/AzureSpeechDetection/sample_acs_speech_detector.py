#!/usr/bin/env python

#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2020 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2020 The MITRE Corporation                                      #
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

from __future__ import division, print_function

import sys
import json
import argparse
import mimetypes
from datetime import timedelta

import mpf_component_api as mpf

from acs_speech_component import AcsSpeechComponent

def guess_type(filename):
    if filename.endswith('.mkv'):
        return ('video/x-matroska', None)
    return mimetypes.guess_type(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Sample Azure Speech component on audio or video files.'
        )
    )
    parser.add_argument('--START_TIME_MILLIS', type=int)
    parser.add_argument('--STOP_TIME_MILLIS', type=int)
    parser.add_argument('--START_FRAME', type=int)
    parser.add_argument('--STOP_FRAME', type=int)
    parser.add_argument('--FRAME_COUNT', type=int)
    parser.add_argument('--FPS', type=float)
    parser.add_argument('--DURATION', type=float)

    parser.add_argument('--ACS_URL', type=str)
    parser.add_argument('--ACS_ACCOUNT_NAME', type=str)
    parser.add_argument('--ACS_SUBSCRIPTION_KEY', type=str)
    parser.add_argument('--ACS_SPEECH_KEY', type=str)
    parser.add_argument('--ACS_CONTAINER_NAME', type=str)
    parser.add_argument('--ACS_ENDPOINT_SUFFIX', type=str)
    parser.add_argument('--LANGUAGE', type=str, default="en-US")
    parser.add_argument('--CLEANUP', action='store_true')
    parser.add_argument('--JSON_FILE', type=str, default=None)
    parser.add_argument(
        'files',
        nargs='+',
        help='locations of the audio or video files'
    )
    args = parser.parse_args()

    properties = dict(
        LANGUAGE=str(args.LANGUAGE),
        CLEANUP=str(args.CLEANUP)
    )

    if args.ACS_URL is not None:
        properties['ACS_URL'] = str(args.ACS_URL)
    if args.ACS_ACCOUNT_NAME is not None:
        properties['ACS_ACCOUNT_NAME'] = str(args.ACS_ACCOUNT_NAME)
    if args.ACS_SUBSCRIPTION_KEY is not None:
        properties['ACS_SUBSCRIPTION_KEY'] = str(args.ACS_SUBSCRIPTION_KEY)
    if args.ACS_SPEECH_KEY is not None:
        properties['ACS_SPEECH_KEY'] = str(args.ACS_SPEECH_KEY)
    if args.ACS_CONTAINER_NAME is not None:
        properties['ACS_CONTAINER_NAME'] = str(args.ACS_CONTAINER_NAME)
    if args.ACS_ENDPOINT_SUFFIX is not None:
        properties['ACS_ENDPOINT_SUFFIX'] = str(args.ACS_ENDPOINT_SUFFIX)
    if args.ACS_URL is not None:
        properties['ACS_URL'] = str(args.ACS_URL)

    media_properties = dict()
    if args.DURATION is not None:
        media_properties['DURATION'] = str(args.DURATION)
    if args.FPS is not None:
        media_properties['FPS'] = str(args.FPS)
    if args.FRAME_COUNT is not None:
        media_properties['FRAME_COUNT'] = str(args.FRAME_COUNT)

    if not len(args.files):
        parser.error("Must provide at least one audio or video file")

    filetype = guess_type(args.files[0])[0].split('/')[0]
    for uri in args.files:
        t = guess_type(uri)[0].split('/')[0]
        if t != filetype:
            parser.error((
                "When processing multiple files, must either be all video or"
                " all audio ({:s} is a {:s} file, while {:s} is a {:s} file)."
            ).format(args.files[0], filetype, uri, y))

    if filetype == 'audio':
        if args.FPS is not None:
            parser.error(
                "FPS not used when processing audio files."
            )
        if args.START_FRAME is not None or args.STOP_FRAME is not None:
            parser.error(
                "START_FRAME and STOP_FRAME not used when processing audio"
                " files. Use START_TIME_MILLIS and STOP_TIME_MILLIS."
            )
        # if args.VOICED_SEGMENTS and args.DURATION is None:
        #     parser.error(
        #         "DURATION is required when VOICED_SEGMENTS is supplied."
        #     )
    elif filetype == 'video':
        if args.FPS is None:
            parser.error(
                "FPS must be provided when passing video files."
            )
        if args.START_TIME_MILLIS is not None or args.STOP_TIME_MILLIS is not None:
            parser.error(
                "START_TIME_MILLIS and STOP_TIME_MILLIS not used when"
                " processing video files. Use START_FRAME and STOP_FRAME."
            )
        # if args.VOICED_SEGMENTS and args.FRAME_COUNT is None:
        #     parser.error(
        #         "FRAME_COUNT is required when VOICED_SEGMENTS is supplied."
        #     )

    comp = AcsSpeechComponent()
    for uri in args.files:
        print("Processing %s file: %s"%(filetype,uri))
        if filetype == 'audio':
            start = args.START_TIME_MILLIS
            stop = args.STOP_TIME_MILLIS
            if start is None:
                start = 0
            if stop is None:
                stop = -1
            dets = comp.get_detections_from_audio(mpf.AudioJob(
                job_name="acs_speech_sample:%s"%uri,
                data_uri=uri,
                start_time=start,
                stop_time=stop,
                job_properties=properties,
                media_properties=media_properties,
                feed_forward_track=None
            ))

        elif filetype == 'video':
            start = args.START_FRAME
            stop = args.STOP_FRAME
            if start is None:
                start = 0
            if stop is None:
                stop = -1
            dets = comp.get_detections_from_video(mpf.VideoJob(
                job_name="acs_speech_sample:%s"%uri,
                data_uri=uri,
                start_frame=start,
                stop_frame=stop,
                job_properties=properties,
                media_properties=media_properties,
                feed_forward_track=None
            ))

        if args.JSON_FILE is not None:
            obj = []
            for det in dets:
                d = det.detection_properties._dict
                d['CONFIDENCE'] = det.confidence
                if filetype == 'audio':
                    d['START_TIME'] = det.start_time
                    d['STOP_TIME'] = det.stop_time
                elif filetype == 'video':
                    d['START_FRAME'] = det.start_frame
                    d['STOP_FRAME'] = det.stop_frame
                obj.append(d)
            with open(args.JSON_FILE, 'w') as fout:
                json.dump(obj, fout)

        for det in dets:
            props = det.detection_properties
            speaker_id = props['SPEAKER_ID']
            transcript = props['TRANSCRIPT']
            # gend_lab = props['GENDER']
            # gend_conf = float(props['GENDER_CONFIDENCE'])
            # lang_lab = props['DECODED_LANGUAGE']
            # lang_conf = float(props['DECODED_LANGUAGE_CONFIDENCE'])
            # lang_labs = props['SPEAKER_LANGUAGES']
            # lang_confs = props['SPEAKER_LANGUAGE_CONFIDENCES']
            # missing_langs = props['MISSING_LANGUAGE_MODELS']

            print('SPEAKER ID: %s' % speaker_id)
            print('  CONFIDENCE: {:.2f}'.format(det.confidence))
            if filetype == 'audio':
                print('  SEGMENT:    {} - {}'.format(
                    timedelta(seconds=det.start_time // 1000),
                    timedelta(seconds=det.stop_time // 1000)
                ))
            elif filetype == 'video':
                print('  FRAMES:     {:d} - {:d}'.format(
                    det.start_frame,
                    det.stop_frame
                ))

            # print('  GENDER:     {:<6s}  {:.2f}'.format(gend_lab, gend_conf))
            # print('  LANGUAGE:  ', end = '')

            # langs = lang_labs.split(', ')
            # confs = map(float, lang_confs.split(', '))
            # print('>' if langs[0] == lang_lab else ' ', end = '')
            # print('{:<6s}  {:.2f}'.format(langs.pop(0), next(confs)))
            # for lang, conf in zip(langs, confs):
            #     print('             ', end = '')
            #     print('>' if lang == lang_lab else ' ', end = '')
            #     print('{:<6s}  {:.2f}'.format(lang, conf))
            # if missing_langs:
            #     print('  MISSING:    {:s}'.format(missing_langs))
            print('  TRANSCRIPT: {:s}'.format(transcript))
