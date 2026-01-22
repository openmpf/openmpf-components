#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2023 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2023 The MITRE Corporation                                      #
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

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from typing import Sequence, Mapping

from openai import OpenAI
from transformers import AutoTokenizer
from jinja2 import Environment, FileSystemLoader
import os, sys

# No local model loading; using remote API
from .schema import response_format, StructuredResponse

from .llm_util.classifiers import get_classifier_lines
from .llm_util.slapchop import split_csv_into_chunks, summarize_summaries
from .llm_util.input_cleanup import convert_tracks_to_csv

from pkg_resources import resource_filename

logger = logging.getLogger('QwenSpeechSummaryComponent')

class QwenSpeechSummaryComponent:
    
    def get_output(self, classifiers, input):
        prompt = self.template.render(input = input, classifiers=classifiers)
        with self.client_factory() as client:
            stream = client.chat.completions.create(
                model=self.client_model_name, #model_name ## for ollama
                # reasoning_effort='none',
                messages=[
                    {"role": "user", "content": prompt, "reasponse_format": response_format}
                ],
                temperature=0,
                stream=True,
                max_tokens=0.95 * (self.max_model_len - self.chunk_size - self.overlap),
                timeout=300,
            )
            content = ""
            for event in stream:
                if event.choices[0].finish_reason != None:
                    break
                if event.object == "chat.completion.chunk":
                    if hasattr(event.choices[0].delta, 'reasoning'):
                        print(event.choices[0].delta.reasoning, end="", file=sys.stderr)
                    if len(event.choices[0].delta.content) > 0:
                        content += event.choices[0].delta.content
        return content

    @staticmethod
    def get_video_track_for_classifier(video_job: mpf.VideoJob, classifier):
        detection_properties = {'CLASSIFIER': classifier.classifier, 'REASONING': classifier.reasoning}
        # TODO: translate utterance start to frame number based on fps
        return mpf.VideoTrack(video_job.start_frame, video_job.stop_frame, classifier.confidence, {0: mpf.ImageLocation(0, 0, 0, 0, classifier.confidence, detection_properties)}, detection_properties)

    def get_classifier_track(self, video_job):
        func = lambda classifier: QwenSpeechSummaryComponent.get_video_track_for_classifier(video_job, classifier)
        return func

    def setup_client(self, config):
        self.model_name_hf = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

        # max_model_len (must match vllm container) >> chunk_size + overlap + completion max_tokens (above)
        self.max_model_len = int(os.environ.get('MAX_MODEL_LEN', 45000))
        self.chunk_size = int(os.environ.get('INPUT_TOKEN_CHUNK_SIZE', 10000))
        self.overlap = int(os.environ.get('INPUT_CHUNK_TOKEN_OVERLAP', 500))

        # TODO: warn if chunk_size is TOO LARGE of a proportion of max_model_len

        # vllm
        self.client_model_name = self.model_name_hf

        logger.debug(f"Using VLLM URI: {config.vllm_uri}")  ## DEBUG

        # Set OpenAI API base URL
        self.client_factory = lambda: OpenAI(base_url=config.vllm_uri, api_key="whatever")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_hf)
        self.tokenizer.add_special_tokens({'sep_token': '<|newline|>'})

    def get_detections_from_all_video_tracks(self, video_job: mpf.AllVideoTracksJob) -> Sequence[mpf.VideoTrack]:
        print(f'Received feed forward video job.')

        #print('Received all tracks video job: {video_job}')

        config = JobConfig(video_job.job_properties)
        self.setup_client(config)

        if config.prompt_template:
            self.env = Environment(loader = FileSystemLoader(os.path.dirname(config.prompt_template)))
            self.template = self.env.get_template(os.path.basename(config.prompt_template))
        else:
            self.env = Environment(loader = FileSystemLoader(os.path.realpath(resource_filename(__name__, 'templates'))))
            self.template = self.env.get_template('prompt.jinja')


        if video_job.feed_forward_tracks is not None:
            classifiers = get_classifier_lines(config.classifiers_path, config.enabled_classifiers)

            input = convert_tracks_to_csv(video_job.feed_forward_tracks)

            summaries = []
            chunks = split_csv_into_chunks(self.tokenizer, input, self.chunk_size, self.overlap)
            nchunks = len(chunks)
            for idx,chunk in enumerate(chunks):
                print(f"chunk [{idx+1} / {nchunks}] ({round(100.0 * (idx+1) / nchunks)}%)", flush=True)
                content = self.get_output(classifiers, chunk)
                summaries += [StructuredResponse.model_validate_json(content)] # type: ignore
            if nchunks == 1:
                final_summary = summaries[0]
            else:
                final_summary = summarize_summaries(StructuredResponse, self.tokenizer, lambda input: self.get_output(classifiers, input), self.chunk_size, self.overlap, summaries)
            if config.debug:
                print(final_summary.json())
            main_detection_properties = {
                'TEXT': final_summary.summary,
                'PRIMARY TOPIC': final_summary.primary_topic,
                'OTHER TOPICS': ', '.join(final_summary.other_topics),
                **{k.upper(): ', '.join(v) for (k,v) in final_summary.entities.__dict__.items()}
            }
            results = [mpf.VideoTrack(
                    video_job.start_frame,
                    video_job.stop_frame,
                    -1,
                    {
                        # TODO: translate utterance start to frame number based on fps
                        0: mpf.ImageLocation(0, 0, 0, 0, -1, main_detection_properties)
                    },
                    main_detection_properties
                )]
            results += list(
                map(
                    self.get_classifier_track(video_job), final_summary.classifiers
                )
            )
            print(f'get_detections_from_all_video_tracks found: {len(results)} detections')
            if config.debug:
                print(f'get_detections_from_all_video_tracks results: {results}')
            return results

        else:
            the_roof = Exception("Received no feed forward tracks")
            raise the_roof

    def get_detections_from_audio(self, job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        print(f'Received audio job.')

        raise Exception('Getting 1 track at a time is going to be rough')

class JobConfig:
    def __init__(self, props: Mapping[str, str]):
        # if debug is true will return which corpus sentences triggered the match
        self.debug = mpf_util.get_property(props, 'ENABLE_DEBUG', False)

        self.prompt_template = mpf_util.get_property(props, 'PROMPT_TEMPLATE', None)

        self.vllm_uri = \
            mpf_util.get_property(props, 'VLLM_URI', "http://qwen-speech-summarization-server:11434/v1")

        self.enabled_classifiers = \
            mpf_util.get_property(props, 'ENABLED_CLASSIFIERS', "ALL")

        self.classifiers_file = \
            mpf_util.get_property(props, 'CLASSIFIERS_FILE', "classifiers.json")

        if "$" not in self.classifiers_file and '/' not in self.classifiers_file:
            self.classifiers_path = os.path.realpath(resource_filename(__name__, self.classifiers_file))
        else:
            self.classifiers_path = os.path.expandvars(self.classifiers_file)

        if not os.path.exists(self.classifiers_path):
            print('Failed to complete job due incorrect file path for the qwen classifiers path: '
                             f'"{self.classifiers_path}"')
            raise mpf.DetectionException(
                'Invalid path provided for qwen classifiers path: '
                f'"{self.classifiers_path}"',
                mpf.DetectionError.COULD_NOT_READ_DATAFILE)

def run_component_test():
    qsc = QwenSpeechSummaryComponent()
    input = None
    with open(os.path.join(os.path.dirname(__file__), 'test_data', 'test.txt')) as f:
        input = f.read()
    input = input.replace("\r\n", "\n")

    job = mpf.AllVideoTracksJob('Test Job', '/dev/null', 0, 9000, {}, {}, [
        mpf.VideoTrack(0, 1, -100, {}, {
            "DEFAULT_LANGUAGE": "eng",
            "LANGUAGE": "eng",
            "SPEAKER_ID": None,
            "GENDER": None,
            "TRANSCRIPT": x
        }) for x in input.split('\n\n') # type: ignore
    ])

    print('About to call get_detections_from_all_video_tracks')
    return qsc.get_detections_from_all_video_tracks(job)


if __name__ == '__main__':
    run_component_test()
