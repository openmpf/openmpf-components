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

import logging
import os
import pickle
import signal
import socket
import sys
import torch

from transformers import AutoModelForCausalLM, AutoProcessor 

DEVICE = "cuda:0"
MODEL_PATH = "DAMO-NLP-SG/VideoLLaMA3-7B"
MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")

LOG_FORMAT = '%(levelname)-5s %(process)d [%(filename)s:%(lineno)d] - %(message)s'

logging.basicConfig(stream=sys.stdout, format=LOG_FORMAT, level=logging.DEBUG)
log = logging.getLogger('LlamaVideoSummarizationSubprocess')

class VideoProcessor:

    def __init__(self):
        log.info("Loading model/processor...")

        self._model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            revision=MODEL_REVISION,
            local_files_only=True,
            trust_remote_code=True,
            device_map={"": DEVICE},
            torch_dtype=torch.bfloat16,
            tempurature=0.7, # default 0.7
            repetition_penalty=1.05, # default 1.05
            attn_implementation="flash_attention_2"
        )
        
        self._processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True)

        log.info("Loaded model/processor")


    def process(self, job: dict) -> str:

        log.info(f"Processing \"{job['video_path']}\" with "
            f"fps: {job['process_fps']}, "
            f"max_frames: {job['max_frames']}, "
            f"max_new_tokens: {job['max_new_tokens']}, "
            f"start_time: {job['segment_start_time']}, "
            f"end_time: {job['segment_stop_time']}")

        if job['system_prompt'] == job['generation_prompt']:
            log.debug(f"system_prompt/generation_prompt:\n\n{job['system_prompt']}\n")
        else:
            log.debug(f"system_prompt:\n\n{job['system_prompt']}\n")
            log.debug(f"generation_prompt:\n\n{job['generation_prompt']}\n")

        conversation = [
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": job['system_prompt']},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": job['video_path'],
                                                "fps": job['process_fps'], "max_frames": job['max_frames'],
                                                "start_time": job['segment_start_time'], "end_time": job['segment_stop_time']}},
                    {"type": "text", "text": job['generation_prompt']},
                ]
            },
        ]

        inputs = self._processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        output_ids = self._model.generate(**inputs, max_new_tokens=job['max_new_tokens'])
        response = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        log.debug(f"Response:\n\n{response}")

        return response


class SocketHandler:

    def __init__(self):
        if fd := os.getenv('MPF_SOCKET_FD'):
            self._socket = socket.socket(fileno=int(fd))
            self._socket_stream = self._socket.makefile('rwb')
        else:
            raise ValueError('MPF_SOCKET_FD not set')

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self._socket:
            self._socket.close()
            self._socket = None

    def get_next_job(self):
        request_length = int.from_bytes(self._socket_stream.read(4), 'little')
        # An empty read indicates the socket is closed
        if request_length == 0:
            return None
        return pickle.loads(self._socket_stream.read(request_length))

    def send_response(self, response: str):
        response_bytes = pickle.dumps(response)
        self._socket_stream.write(len(response_bytes).to_bytes(4, 'little'))
        self._socket_stream.write(response_bytes)
        self._socket_stream.flush()

    def send_error(self, error: str):
        error_bytes = pickle.dumps(error)
        self._socket_stream.write((0).to_bytes(4, 'little')) # indicate an error
        self._socket_stream.write(len(error_bytes).to_bytes(4, 'little'))
        self._socket_stream.write(error_bytes)
        self._socket_stream.flush()


socket_hander : SocketHandler

def signal_handler(signal, frame):
    if socket_handler:
        socket_handler.cleanup()

if __name__ == '__main__':

    if len(sys.argv) > 1:
        log_level = int(sys.argv[1])
        log.setLevel(log_level)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    socket_handler = SocketHandler()
    video_processor = VideoProcessor()

    while job := socket_handler.get_next_job():
        try:
            response = video_processor.process(job)
            socket_handler.send_response(response)
        except Exception as e:
            log.error(f'Failed to summarize video: {e}')
            socket_handler.send_error(str(e))
