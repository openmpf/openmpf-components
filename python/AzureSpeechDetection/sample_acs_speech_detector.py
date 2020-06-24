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
    parser.add_argument('--start-time-millis', type=int)
    parser.add_argument('--stop-time-millis', type=int)
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--stop-frame', type=int)
    parser.add_argument('--frame-count', type=int)
    parser.add_argument('--fps', type=float)
    parser.add_argument('--duration', type=float)

    parser.add_argument('--acs-endpoint-url', type=str)
    parser.add_argument('--acs-container-url', type=str)
    parser.add_argument('--acs-subscription-key', type=str)
    parser.add_argument('--acs-service-key', type=str)
    parser.add_argument('--language', type=str, default="en-US")
    parser.add_argument('--no-cleanup', action='store_true')
    parser.add_argument('--json-file', type=str, default=None)
    parser.add_argument(
        'files',
        nargs='+',
        help='locations of the audio or video files'
    )
    args = parser.parse_args()

    properties = dict(
        LANGUAGE=str(args.language),
        CLEANUP=str(not args.no_cleanup)
    )

    if args.acs_endpoint_url is not None:
        properties['ACS_ENDPOINT_URL'] = str(args.acs_endpoint_url)
    if args.acs_container_url is not None:
        properties['ACS_CONTAINER_URL'] = str(args.acs_container_url)
    if args.acs_subscription_key is not None:
        properties['ACS_SUBSCRIPTION_KEY'] = str(args.acs_subscription_key)
    if args.acs_service_key is not None:
        properties['ACS_SERVICE_KEY'] = str(args.acs_service_key)

    media_properties = dict()
    if args.duration is not None:
        media_properties['DURATION'] = str(args.duration)
    if args.fps is not None:
        media_properties['FPS'] = str(args.fps)
    if args.frame_count is not None:
        media_properties['FRAME_COUNT'] = str(args.frame_count)

    if not len(args.files):
        parser.error("Must provide at least one audio or video file")

    filetype = guess_type(args.files[0])[0].split('/')[0]
    for uri in args.files:
        t = guess_type(uri)[0].split('/')[0]
        if t != filetype:
            parser.error((
                "When processing multiple files, must either be all video or"
                " all audio ({:s} is a {:s} file, while {:s} is a {:s} file)."
            ).format(args.files[0], filetype, uri, t))

    if filetype == 'audio':
        if args.fps is not None:
            parser.error(
                "FPS not used when processing audio files."
            )
        if args.start_frame is not None or args.stop_frame is not None:
            parser.error(
                "START_FRAME and STOP_FRAME not used when processing audio"
                " files. Use START_TIME_MILLIS and STOP_TIME_MILLIS."
            )
    elif filetype == 'video':
        if args.fps is None:
            parser.error(
                "FPS must be provided when passing video files."
            )
        if args.start_time_millis is not None or args.stop_time_millis is not None:
            parser.error(
                "START_TIME_MILLIS and STOP_TIME_MILLIS not used when"
                " processing video files. Use START_FRAME and STOP_FRAME."
            )
    else:
        parser.error(
            "Provided file must be an audio or video file"
        )

    comp = AcsSpeechComponent()
    for uri in args.files:
        print("Processing %s file: %s"%(filetype,uri))
        if filetype == 'audio':
            start = args.start_time_millis
            stop = args.stop_time_millis
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
            start = args.start_frame
            stop = args.stop_frame
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

        if args.json_file is not None:
            obj = []
            for det in dets:
                d = dict(det.detection_properties)
                d['CONFIDENCE'] = det.confidence
                if filetype == 'audio':
                    d['START_TIME'] = det.start_time
                    d['STOP_TIME'] = det.stop_time
                elif filetype == 'video':
                    d['START_FRAME'] = det.start_frame
                    d['STOP_FRAME'] = det.stop_frame
                obj.append(d)
            with open(args.json_file, 'w') as fout:
                json.dump(obj, fout)

        for det in dets:
            props = det.detection_properties
            speaker_id = props['SPEAKER_ID']
            transcript = props['TRANSCRIPT']

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

            print('  TRANSCRIPT: {:s}'.format(transcript))
