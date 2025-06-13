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
import sys
import numpy as np
from google import genai
from multiprocessing.shared_memory import SharedMemory
from google.genai.errors import ClientError
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description='Sends image and prompt to Gemini Client for processing.')

    parser.add_argument("--model", "-m", type=str, default="gemma-3-27b-it", help="The name of the Gemini model to use.")
    parser.add_argument("--shm-name", type=str, required=True, help="Shared memory name for image data.")
    parser.add_argument("--shm-shape", type=str, required=True, help="Shape of the image in shared memory (JSON list).")
    parser.add_argument("--shm-dtype", type=str, required=True, help="Numpy dtype of the image in shared memory.")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="The prompt you want to use with the image.")
    parser.add_argument("--api_key", "-a", type=str, required=True, help="Your API key for Gemini.")
    args = parser.parse_args()

    shm = None

    try:
        shape = tuple(json.loads(args.shm_shape))
        dtype = np.dtype(args.shm_dtype)
        shm = SharedMemory(name=args.shm_name)
        np_img = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        image = Image.fromarray(np_img)
        client = genai.Client(api_key=args.api_key)
        content = client.models.generate_content(model=args.model, contents=[args.prompt, image])
        print(content.text)
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
    finally:
        if shm:
            shm.close()
            shm.unlink()

if __name__ == "__main__":
    main()