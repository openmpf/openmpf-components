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

import argparse
import json
import os
import sys

from google import genai
from google.genai import types
from google.genai.types import Part
from google.cloud import storage
from google.genai.errors import ClientError


def main():
    parser = argparse.ArgumentParser(description='Sends image and prompt to Gemini Client for processing.')

    parser.add_argument("--model", "-m", type=str, default="gemini-2.5-flash", help="The name of the Gemini model to use.")
    parser.add_argument("--data_uri", "-d", type=str, required=True, help="Path to the media file to process with Gemini.")
    parser.add_argument("--prompt", "-p", type=str, help="The prompt you want to use with the video.")
    parser.add_argument("--google_application_credientials", "-c", type=str, required=True, help="The JSON file to your credentials to use Vertex AI.")
    parser.add_argument("--project_id", "-i", type=str, required=True, help="Name of your GCP project.")
    parser.add_argument("--bucket_name", "-b", type=str, required=True, help="Name of the GCP bucket.")
    parser.add_argument("--label_prefix", "-l", type=str, required=True, help="Label prefix to use when uploading the video to GCP.")
    parser.add_argument("--label_user", "-u", type=str, required=True, help="User of whom is accessing the GCP resources.")
    parser.add_argument("--label_purpose", "-r", type=str, required=True, help="Purpose of accessing the GCP resources.")
    parser.add_argument("--segment_start", "-s", type=str, required=True, help="Start time of the current segment.")
    parser.add_argument("--segment_stop", "-e", type=str, required=True, help="End time of the current segment.")
    parser.add_argument("--fps", "-f", type=str, default="1.0", help="Specifies the number of frames per second of video to process.")

    args = parser.parse_args()

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_application_credientials

        # GCP resources
        USER = args.label_user
        PURPOSE = args.label_purpose
        LABEL_PREFIX = args.label_prefix
        PROJECT_ID = args.project_id
        BUCKET_NAME = args.bucket_name

        PROMPT = args.prompt
        MODEL = args.model

        # Video segment storage information
        FILE_PATH = args.data_uri
        FILE_NAME = os.path.basename(FILE_PATH)
        STORAGE_PATH = USER + "/" + FILE_NAME

        SEGMENT_START = int(float(args.segment_start))
        SEGMENT_STOP = int(float(args.segment_stop))
        FPS = float(args.fps)

        # Automatically uses ADC to authenticate
        client = storage.Client(
            project=PROJECT_ID
        )

        # Uploads file to GCP bucket
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_PATH)

        # There is no way to set a time-to-live (TTL) for a file in Google Storage.
        # The file will be deleted manually at the end of this script.
        # If you want to set a TTL, you can use the `lifecycle` configuration in the bucket settings.
        # See: https://cloud.google.com/storage/docs/lifecycle
        # For example, you can set a rule to delete files older than 30 days.
        blob.upload_from_filename(FILE_PATH)

        file_uri = f"gs://{BUCKET_NAME}/{STORAGE_PATH}"

        # Automatically uses ADC to authenticate
        client = genai.Client(
            project=PROJECT_ID,
            location="global",
            vertexai=True
        )

        content_config = None
        if(USER != "" and LABEL_PREFIX != "" and PURPOSE != ""):
            # Use labels to help track billing and usage
            content_config = types.GenerateContentConfig(
                labels={
                    LABEL_PREFIX + "user": USER,
                    LABEL_PREFIX + "purpose": PURPOSE,
                    LABEL_PREFIX + "modality": "video"
                }
            )

        response = client.models.generate_content(
            model=MODEL,
            contents=types.Content(
                role='user',
                parts=[
                    Part(
                        file_data=types.FileData(
                            file_uri=file_uri,
                            mime_type='video/mp4'
                        ),
                        video_metadata=types.VideoMetadata(
                            start_offset=str(SEGMENT_START) + "s", 
                            end_offset=str(SEGMENT_STOP) + "s", 
                            fps=FPS
                        )
                    ),
                    Part(
                        text=PROMPT
                    )
                ],
            ),
            config=content_config
        )
        print(f"\n{response.text}")

        # Set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to delete is aborted if the object's
        # generation number does not match your precondition.
        blob.reload()  # fetch blob metadata to use in generation_match_precondition.
        generation_match_precondition = blob.generation

        blob.delete(if_generation_match=generation_match_precondition)

        sys.exit(0)
    except ClientError as e:
        if hasattr(e, 'code') and e.code == 429:
            print("Caught a ResourceExhausted error (429 Too Many Requests)", file=sys.stderr)
            sys.exit(1)
        raise
    except Exception as e:
        err_str = str(e)
        print(err_str, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()