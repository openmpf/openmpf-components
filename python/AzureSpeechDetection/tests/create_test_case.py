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

import os
import sys
import json
import argparse
import mimetypes
from urllib import request

from test_acs_speech import (
    speech_base_url, transcriptions_url, blobs_url, outputs_url, models_url,
    get_test_properties
)
import mpf_component_util as util
import mpf_component_api as mpf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from acs_speech_component import AcsSpeechComponent
from acs_speech_component.job_parsing import AzureJobConfig


def guess_type(filename):
    if filename.endswith('.mkv'):
        return ('video/x-matroska', None)
    return mimetypes.guess_type(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sample Azure Speech component on audio or video files.'
    )

    parser.add_argument('--frame-count', type=int)
    parser.add_argument('--fps', type=float)
    parser.add_argument('--duration', type=float)

    parser.add_argument('--start-time', type=int, default=0)
    parser.add_argument('--stop-time', type=int, default=-1)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--stop-frame', type=int, default=-1)

    parser.add_argument('--diarize', action='store_true')
    parser.add_argument('--language', type=str, default='en-US')
    parser.add_argument('--blob-access-time', type=int, default=120)
    parser.add_argument('--transcription-expiration', type=int, default=120)
    parser.add_argument('--api-version', type=str, default='2025-10-15')
    parser.add_argument('--job-name', type=str, required=True)
    parser.add_argument('filepath')
    args = parser.parse_args()

    properties = get_test_properties(
        DIARIZE=str(args.diarize).upper(),
        LANGUAGE=str(args.language),
        CLEANUP='TRUE',
        USE_SAS_AUTH='TRUE',
        BLOB_ACCESS_TIME=str(args.blob_access_time),
        TRANSCRIPTION_EXPIRATION=str(args.transcription_expiration),
        ACS_API_VERSION=args.api_version,
    )

    media_properties = dict()
    if args.duration is not None:
        media_properties['DURATION'] = str(args.duration)

    filetype = guess_type(args.filepath)[0].split('/')[0]
    if filetype == 'audio':
        if args.fps is not None:
            parser.error('FPS not used when processing audio files.')
    elif filetype == 'video':
        if args.fps is None:
            parser.error('FPS must be provided when passing video files.')
        media_properties['FPS'] = str(args.fps)
        if args.frame_count is not None:
            media_properties['FRAME_COUNT'] = str(args.frame_count)
    else:
        parser.error(f'Filetype must be video or audio (got {filetype})')

    job_name = args.job_name
    recording_id = args.filepath.split('/')[-1]

    if filetype == 'audio':
        job = mpf.AudioJob(
            job_name=job_name,
            data_uri=args.filepath,
            start_time=args.start_time,
            stop_time=args.stop_time,
            job_properties=properties,
            media_properties=media_properties,
            feed_forward_track=None
        )
    else:
        job = mpf.VideoJob(
            job_name=job_name,
            data_uri=args.filepath,
            start_frame=args.start_frame,
            stop_frame=args.stop_frame,
            job_properties=properties,
            media_properties=media_properties,
            feed_forward_track=None
        )

    comp = AcsSpeechComponent()
    job_config = AzureJobConfig(job)
    comp.processor.acs.update_acs(job_config.server_info)

    api_version = job_config.server_info.api_version
    job_url = f"{transcriptions_url}/{job_name}"
    base_local_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)),
        'test_data'
    )

    # Write locales response using the component-computed locales URL
    req = request.Request(
        url=comp.processor.acs._locales_url,
        headers=comp.processor.acs.acs_headers,
        method='GET'
    )
    response = request.urlopen(req)
    supported_locales = json.load(response)
    path = os.path.join(base_local_path, 'transcriptions', 'locales') + '.json'
    with open(path, 'w') as fout:
        json.dump(supported_locales, fout, indent=4)

    recording_url = comp.processor.acs.upload_file_to_blob(
        audio_bytes=util.transcode_to_wav(
            args.filepath,
            start_time=job_config.start_time,
            stop_time=job_config.stop_time),
        recording_id=recording_id,
        blob_access_time=args.blob_access_time,
    )
    try:
        output_loc = comp.processor.acs.submit_batch_transcription(
            recording_url=recording_url,
            job_name=job_name,
            diarize=job_config.diarize,
            language=job_config.language,
            expiry=args.transcription_expiration
        )
    except Exception:
        comp.processor.acs.delete_blob(recording_id)
        raise

    transcription = None
    try:
        result = comp.processor.acs.poll_for_result(output_loc)
        if result['status'] == 'Succeeded':
            results_uri = result['links']['files']
            req = request.Request(
                url=results_uri,
                headers=comp.processor.acs.acs_headers,
                method='GET'
            )
            response = request.urlopen(req)
            files = json.load(response)

            transcript_uri = next(
                v['links']['contentUrl'] for v in files['values']
                if v['kind'] == 'Transcription'
            )
            response = request.urlopen(transcript_uri)
            transcription = json.load(response)
    finally:
        comp.processor.acs.delete_blob(recording_id)
        comp.processor.acs.delete_transcription(output_loc)

    result['self'] = f"{job_url}.json?api-version={api_version}"
    result['model']['self'] = f"{models_url}/base/modelhash"
    result['links']['files'] = f"{job_url}/files.json?api-version={api_version}"
    path = os.path.join(base_local_path, 'transcriptions', job_name) + '.json'
    with open(path, 'w') as fout:
        json.dump(result, fout, indent=4)

    if transcription is not None:
        jobdir = os.path.join(base_local_path, 'transcriptions', job_name)
        outdir = os.path.join(base_local_path, 'outputs')
        os.makedirs(jobdir, exist_ok=True)
        os.makedirs(outdir, exist_ok=True)

        for i, value in enumerate(files['values']):
            value['self'] = f"{job_url}/files/hash{i}.json?api-version={api_version}"
            value['links']['contentUrl'] = 'dummy_file.json'
            if value['kind'] == 'Transcription':
                value['links']['contentUrl'] = f"{outputs_url}/{job_name}.json"

        transcription['source'] = f"{blobs_url}/{recording_id}"

        files_path = os.path.join(jobdir, 'files.json')
        with open(files_path, 'w') as fout:
            json.dump(files, fout, indent=4)

        transcription_path = os.path.join(outdir, job_name) + '.json'
        with open(transcription_path, 'w') as fout:
            json.dump(transcription, fout, indent=4)
