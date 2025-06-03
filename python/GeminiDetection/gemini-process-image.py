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
from google import genai
from PIL import Image
import sys

def main():
    parser = argparse.ArgumentParser(description='Sends image and prompt to Gemini Client for processing.')

    parser.add_argument("--model", "-m", type=str, default="gemma-3-27b-it", help="The name of the Gemini model to use.")
    parser.add_argument("--filepath", "-f", type=str, required=True, help="Path to the media file to process with Gemini.")
    parser.add_argument("--prompt", "-p", type=str, required=True, help="The prompt you want to use with the image.")
    parser.add_argument("--api_key", "-a", type=str, required=True, help="Your API key for Gemini.")
    args = parser.parse_args()
    
    try:
        client = genai.Client(api_key=args.api_key)
        content = client.models.generate_content(model=args.model, contents=[args.prompt, Image.open(args.filepath)])
        print(content.text)
        sys.exit(0)
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print("Caught a ResourceExhausted error (429 Too Many Requests", file=sys.stderr)
        else:
            print(err_str, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()