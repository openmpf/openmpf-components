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

import os
import sys
import json
import argparse
import mimetypes
from urllib2 import Request, urlopen, HTTPError

import mpf_component_api as mpf
from test_acs_speech import transcription_url, blobs_url, outputs_url

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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
    parser.add_argument('--start_time', type=int)
    parser.add_argument('--stop_time', type=int)
    parser.add_argument('--start_frame', type=int)
    parser.add_argument('--stop_frame', type=int)
    parser.add_argument('--frame_count', type=int)
    parser.add_argument('--fps', type=float)
    parser.add_argument('--duration', type=float)

    parser.add_argument('--diarize', action='store_true')
    parser.add_argument('--language', type=str, default="en-US")
    parser.add_argument('--job_name', type=str, required=True)
    parser.add_argument('filepath')
    args = parser.parse_args()

    properties = dict(
        DIARIZE=str(args.diarize).upper(),
        LANGUAGE=str(args.language),
        CLEANUP="TRUE"
    )

    media_properties = dict()
    if args.duration is not None:
        media_properties['DURATION'] = str(args.duration)
    if args.fps is not None:
        media_properties['FPS'] = str(args.fps)
    if args.frame_count is not None:
        media_properties['FRAME_COUNT'] = str(args.frame_count)

    filetype = guess_type(args.filepath)[0].split('/')[0]
    if filetype == 'audio':
        if args.fps is not None:
            parser.error(
                "FPS not used when processing audio files."
            )
        if args.start_frame is not None or args.stop_frame is not None:
            parser.error(
                "start_frame and stop_frame not used when processing audio"
                " files. Use start_time and stop_time."
            )
    elif filetype == 'video':
        if args.fps is None:
            parser.error(
                "FPS must be provided when passing video files."
            )
        if args.start_time is not None or args.stop_time is not None:
            parser.error(
                "start_time and stop_time not used when"
                " processing video files. Use start_frame and stop_frame."
            )

    comp = AcsSpeechComponent()
    parsed_properties = comp._parse_properties(properties)
    comp.processor._update_acs(
        acs_endpoint_url=parsed_properties['acs_endpoint_url'],
        acs_container_url=parsed_properties['acs_container_url'],
        acs_subscription_key=parsed_properties['acs_subscription_key'],
        acs_service_key=parsed_properties['acs_service_key'],
    )

    job_name = args.job_name
    gen_job_name = (job_name is None)

    recording_id = args.filepath.split("/")[-1]
    if gen_job_name:
        job_name = os.path.splitext(recording_id)[0]
        if args.diarize:
            job_name += "_diar"
        job_name += "_" + args.language

    if filetype == 'audio':
        start_time = args.start_time
        stop_time = args.stop_time
        if gen_job_name and not (start_time is None and stop_time is None):
            job_name += (
                "_" +
                ("0" if start_time is None else str(start_time)) +
                "-" +
                ("EOF" if stop_time < 0 else str(stop_time))
            )
        if start_time is None:
            start_time = 0
        if stop_time is None:
            stop_time = -1
    elif filetype == 'video':
        start_frame = args.start_frame
        stop_frame = args.stop_frame
        if gen_job_name and not (start_frame is None and stop_frame is None):
            job_name += (
                "_" +
                ("0" if start_frame is None else str(start_frame)) +
                "-" +
                ("EOF" if stop_frame is None else str(stop_frame))
            )
        # Convert frame locations to timestamps
        fpms = float(media_properties['FPS']) / 1000.0
        start_time = 0 if start_frame is None else start_frame / fpms
        stop_time = -1 if stop_frame is None else stop_frame / fpms

    recording_url = comp.processor.acs.upload_file_to_blob(
        filepath=args.filepath,
        recording_id=recording_id,
        start_time=start_time,
        stop_time=stop_time
    )
    try:
        output_loc = comp.processor.acs.submit_batch_transcription(
            recording_url=recording_url,
            job_name=job_name,
            diarize=parsed_properties['diarize'],
            language=parsed_properties['lang']
        )
    except:
        comp.processor.acs.delete_blob(recording_id)
        raise

    transcription = None
    try:
        result = comp.processor.acs.poll_for_result(output_loc)
        if result['status'] == 'Succeeded':
            results_uri = result['resultsUrls']['channel_0']
            response = urlopen(results_uri)
            transcription = json.load(response)
    finally:
        comp.processor.acs.delete_blob(recording_id)
        comp.processor.acs.delete_transcription(output_loc)


    result['recordingsUrl'] = "{}/{}?test_queries".format(
        blobs_url,
        recording_id
    )
    result['reportFileUrl'] = "test.report.file"
    for k in result['resultsUrls'].keys():
        result['resultsUrls'][k] = "{}/{}.json?test_queries".format(
            outputs_url,
            job_name
        )

    path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        'test_data',
        'transcriptions',
        job_name
    ) + '.json'
    with open(path, 'w') as fout:
        json.dump(result, fout, indent=4)

    if transcription is not None:
        path = os.path.join(
            os.path.realpath(os.path.dirname(__file__)),
            'test_data',
            'outputs',
            job_name
        ) + '.json'
        with open(path, 'w') as fout:
            json.dump(transcription, fout, indent=4)
