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

import io
import os
import pickle
import socket
import torch

from transformers import AutoModelForCausalLM, AutoProcessor
from typing import Mapping

DEVICE = "cuda:0"
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"

class VideoProcessor:

    model = None
    processor = None

    def __init__(self, job_properties: Mapping[str, str]):
        print("Loading model/processor...")

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            device_map={"": DEVICE},
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

        print("Loaded model/processor")


    def process(video_path, fps, max_frames, max_new_tokens, generation_prompt_path, system_prompt_path):

        print(f"Processing \"{video_path}\" with fps: {fps}, max_frames: {max_frames}, max_new_tokens: {max_new_tokens},")

        if system_prompt == generation_prompt:
            print(f"system_prompt/generation_prompt:\n\n{system_prompt}\n")
        else:
            print(f"system_prompt:\n\n{system_prompt}\n")
            print(f"generation_prompt:\n\n{generation_prompt}\n")

        conversation = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": max_frames}},
                    # {"type": "video", "video": {"video_path": "/llama/data/camera_23.mp4", "fps": 1, "max_frames": 180}},
                    # {"type": "text", "text": "How many animals are in the video?"},
                    {"type": "text", "text": generation_prompt},
                ]
            },
        ]

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"Response:\n\n{response}")


def get_request_socket() -> io.BufferedRWPair:

    if fd := os.getenv('MPF_SOCKET_FD'):
        with socket.socket(fileno=int(fd)) as sock:
            return sock.makefile('rwb')
    raise ValueError('MPF_SOCKET_FD not set')


if __name__ == '__main__':

    print("HELLO WORLD") # DEBUG

    request_socket = get_request_socket()
    request_length = int.from_bytes(request_socket.read(4), 'little')
    print(f'{request_length=}')

    request = pickle.loads(request_socket.read(request_length))
    print(f'{request=}')


    response = "RESPONSE"
    response_bytes = pickle.dumps(response)
    request_socket.write(len(response_bytes).to_bytes(4, 'little'))
    request_socket.write(response_bytes)
    request_socket.flush()


    # TODO: Create instance of VideoProcessor
    # TODO: Loop over socket for jobs

    # parser = argparse.ArgumentParser(description='Process some arguments.')

    # # Valid ranges? https://replicate.com/lucataco/videollama3-7b/api/schema

    # parser.add_argument('videoPath', type=str, help='path to video file')
    # parser.add_argument('--fps', type=int, default=1, help='frames per second to process')
    # # TODO: Does this setting actually do anything?
    # parser.add_argument('--maxFrames', type=int, default=180, help='maximum number of frames to process')
    # parser.add_argument('--maxNewTokens', type=int, default=1024, help='maximum number of tokens to generate, ignoring the number of tokens in the prompt')
    # parser.add_argument('--generationPromptPath', type=str, help='path to text file containing the generation prompt')
    # parser.add_argument('--systemPromptPath', type=str, help='path to text file containing the system (role) prompt; if omitted, the generation prompt is used')

    # args = parser.parse_args()

    # if not args.videoPath:
    #     parser.print_help()
    #     parser.exit()

    # process(args.videoPath, args.fps, args.maxFrames, args.maxNewTokens, args.generationPromptPath, args.systemPromptPath)
