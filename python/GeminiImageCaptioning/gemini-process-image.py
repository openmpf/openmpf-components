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
import base64
import json
import os
import sys
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from openai import OpenAI, RateLimitError

from gemini_component.resource_tracker_monkeypatch import remove_shm_from_resource_tracker


def get_openai_api_key(application_credentials):
    environment_value = os.environ.get(application_credentials)
    if environment_value:
        return environment_value
    if application_credentials and os.path.exists(application_credentials):
        with open(application_credentials, "r") as api_key_file:
            return api_key_file.read().strip() or "Empty"
    return "Empty"


def main():
    parser = argparse.ArgumentParser(
        description="Sends an image and prompt to an OpenAI-compatible or Google API.")
    parser.add_argument("--api", choices=("OpenAI", "Google"), default="OpenAI")
    parser.add_argument("--model", "-m", default="gemma-3-27b-it")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--application-credentials", default="")
    parser.add_argument("--project-id", default="")
    parser.add_argument("--shm-name", required=True)
    parser.add_argument("--shm-shape", required=True)
    parser.add_argument("--shm-dtype", required=True)
    parser.add_argument("--prompt", "-p", required=True)
    args = parser.parse_args()

    remove_shm_from_resource_tracker()
    shared_memory = None
    try:
        shape = tuple(json.loads(args.shm_shape))
        dtype = np.dtype(args.shm_dtype)
        shared_memory = SharedMemory(name=args.shm_name)
        image = np.ndarray(shape, dtype=dtype, buffer=shared_memory.buf)
        encoded, image_buffer = cv2.imencode(".jpg", image)
        if not encoded:
            raise RuntimeError("Failed to encode image as JPEG.")
        image_bytes = image_buffer.tobytes()

        if args.api == "OpenAI":
            client = OpenAI(
                api_key=get_openai_api_key(args.application_credentials),
                base_url=args.base_url or None)
            image_data = base64.b64encode(image_bytes).decode("ascii")
            response = client.chat.completions.create(
                model=args.model,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": args.prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }])
            print(response.choices[0].message.content or "")
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.application_credentials
            client = genai.Client(
                project=args.project_id, location="global", vertexai=True)
            response = client.models.generate_content(
                model=args.model,
                contents=types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(
                            data=image_bytes, mime_type="image/jpeg"),
                        types.Part.from_text(text=args.prompt)
                    ]))
            print(response.text or "")
    except (RateLimitError, ClientError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    except Exception as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    finally:
        if shared_memory:
            shared_memory.close()


if __name__ == "__main__":
    main()
