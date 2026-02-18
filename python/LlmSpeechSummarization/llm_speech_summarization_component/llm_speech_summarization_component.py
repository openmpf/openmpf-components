#############################################################################
# NOTICE                                                                    #
#                                                                           #
# This software (or technical data) was produced for the U.S. Government    #
# under contract, and is subject to the Rights in Data-General Clause       #
# 52.227-14, Alt. IV (DEC 2007).                                            #
#                                                                           #
# Copyright 2025 The MITRE Corporation. All Rights Reserved.                #
#############################################################################

#############################################################################
# Copyright 2025 The MITRE Corporation                                      #
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

import importlib.resources
import logging
import math
import os
from openai.types import ResponseFormatJSONSchema
import requests
import sys
import time

from jinja2 import Environment, FileSystemLoader
from typing import Sequence, Mapping

from openai import OpenAI

if not os.environ.get("HF_HUB_OFFLINE"): os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

import mpf_component_api as mpf
import mpf_component_util as mpf_util

# If import fails with ., assume we are running in pydebug and the CWD is proper
try:
    from .schema import response_format_json_schema, StructuredResponse

    from .llm_util.classifiers import get_classifier_lines
    from .llm_util.slapchop import split_csv_into_chunks, summarize_summaries
    from .llm_util.input_cleanup import convert_tracks_to_csv
except:
    from schema import response_format_json_schema, StructuredResponse

    from llm_util.classifiers import get_classifier_lines
    from llm_util.slapchop import split_csv_into_chunks, summarize_summaries
    from llm_util.input_cleanup import convert_tracks_to_csv

logger = logging.getLogger('LLMSpeechSummaryComponent')


class JobConfig:
    def __init__(self, props: Mapping[str, str]):
        # if debug is true will return which corpus sentences triggered the match
        self.debug = mpf_util.get_property(props, 'ENABLE_DEBUG', False)

        self.vllm_model = mpf_util.get_property(props, 'VLLM_MODEL', "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")

        self.max_model_len = int(mpf_util.get_property(props, 'MAX_MODEL_LEN', 45000))
        self.chunk_size = int(mpf_util.get_property(props, 'INPUT_TOKEN_CHUNK_SIZE', 10000))
        self.overlap = int(mpf_util.get_property(props, 'INPUT_CHUNK_TOKEN_OVERLAP', 500))

        self.api_token = mpf_util.get_property(props, 'API_TOKEN', "Must be set for anonymous VLLM, but can be anything")

        self.prompt_template = self._get_file_path(mpf_util.get_property(props, 'PROMPT_TEMPLATE', 'templates/prompt.jinja'))

        self.vllm_uri = \
            mpf_util.get_property(props, 'VLLM_URI', "http://qwen-speech-summarization-server:11434/v1")

        self.vllm_health_uri = \
            mpf_util.get_property(props, 'VLLM_HEALTH_URI', "../health")
        if '://' not in self.vllm_health_uri:
            self.vllm_health_uri = os.path.join(self.vllm_uri, self.vllm_health_uri)

        self.enabled_classifiers = \
            mpf_util.get_property(props, 'CLASSIFIERS_LIST', "ALL")

        # exclude classifiers from output if their confidence is below this threshold
        self.classifier_confidence_minimum = \
            mpf_util.get_property(props, 'CLASSIFIER_CONFIDENCE_MINIMUM', 0.3)

        self.classifiers_path = \
            self._get_file_path(mpf_util.get_property(props, 'CLASSIFIERS_FILE', "classifiers.json"))

    @staticmethod
    def _get_file_path(path: str) -> str:
        expanded_path = os.path.expandvars(path)
        if os.path.exists(expanded_path):
            return expanded_path
        resource = importlib.resources.files(__name__) / expanded_path
        if resource.is_file():
            return str(importlib.resources.as_file(resource).__enter__())
        raise mpf.DetectionError.COULD_NOT_READ_DATAFILE.exception(
            f"{path} does not exist.")

class LlmSpeechSummaryComponent:
    def _get_output(self, config: JobConfig, template, classifiers, input):
        if self.client_factory:
            client_factory = self.client_factory
        else:
            client_factory = lambda: LlmSpeechSummaryComponent._get_openai_api_client_when_server_is_ready(config, base_url=config.vllm_uri, api_key=config.api_token)
        prompt = template.render(input=input, classifiers=classifiers, response_format_json_schema=response_format_json_schema)
        with client_factory() as client:
            stream = client.chat.completions.create(
                model=config.vllm_model, #model_name ## for ollama
                # reasoning_effort='none',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=True,
                max_completion_tokens=int(math.floor(0.95 * (config.max_model_len - config.chunk_size - config.overlap))),
                timeout=300,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "StructuredResponse",
                        "schema": response_format_json_schema,
                        "strict": True
                    }
                }
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
    def _get_video_track_for_classifier(video_job: mpf.VideoJob, classifier):
        detection_properties = {'CLASSIFIER': classifier.classifier, 'REASONING': classifier.reasoning}
        # TODO: translate utterance start to frame number based on fps
        return mpf.VideoTrack(video_job.start_frame, video_job.stop_frame, classifier.confidence, {0: mpf.ImageLocation(0, 0, 0, 0, classifier.confidence, detection_properties)}, detection_properties)

    @staticmethod
    def _get_classifier_track(video_job):
        func = lambda classifier: LlmSpeechSummaryComponent._get_video_track_for_classifier(video_job, classifier)
        return func

    @staticmethod
    def _get_openai_api_client_when_server_is_ready(config, timeout_seconds=300, retry_delay_seconds=5, **kwargs):
        if config.vllm_health_uri:
            start_time = time.time()
            success = False
            failed_ever = False
            last_error = None
            while time.time() - start_time < timeout_seconds:
                try:
                    response = requests.get(config.vllm_health_uri, timeout=retry_delay_seconds)
                    if response.status_code == 200:
                        if failed_ever:
                            logger.info("VLLM is now available")
                        success = True
                        break
                    else:
                        failed_ever = True
                        logger.warning(f"Received HTTP{response.status_code} from {config.vllm_health_uri}")
                except Exception as e:
                    failed_ever = True
                    logger.info(f"Waiting up to {timeout_seconds}s for VLLM at {config.vllm_health_uri} to be healthy. {int(math.floor(time.time() - start_time))}s passed so far")
                    last_error = e
                time.sleep(retry_delay_seconds)

            if not success:
                if last_error:
                    raise last_error
                raise Exception("Timed out waiting for VLLM to be healthy")

        return OpenAI(**kwargs)

    def __init__(self, clientFactory=None):
        self.client_factory = clientFactory

    def get_detections_from_all_video_tracks(self, video_job: mpf.AllVideoTracksJob) -> Sequence[mpf.VideoTrack]:
        """
        Process:
        1. Convert all input tracks to a CSV format
        2. Split CSV into chunks to fit in context window, retaining header row on each chunk
        3. Run each chunk through the LLM, templating classifiers and each chunk into the prompt, receiving its summary
        4. Recursively summarize summaries until only 1 remains
        5. Convert to final summary VideoTracks for output
        """
        logger.info(f'Received feed forward video job.')

        config = JobConfig(video_job.job_properties)

        tokenizer = AutoTokenizer.from_pretrained(config.vllm_model, local_files_only=(os.environ["HF_HUB_OFFLINE"] == "1"))
        tokenizer.add_special_tokens({'sep_token': '<|newline|>'})

        env = Environment(loader = FileSystemLoader(os.path.dirname(config.prompt_template)))
        template = env.get_template(os.path.basename(config.prompt_template))

        if video_job.feed_forward_tracks is not None:
            classifiers = get_classifier_lines(config.classifiers_path, config.enabled_classifiers)

            input = convert_tracks_to_csv(video_job.feed_forward_tracks)

            summaries = []
            chunks = split_csv_into_chunks(tokenizer, input, config.chunk_size, config.overlap)
            nchunks = len(chunks)
            for idx,chunk in enumerate(chunks):
                logger.debug(f"chunk [{idx+1} / {nchunks}] ({round(100.0 * (idx+1) / nchunks)}%)")
                content = self._get_output(config, template, classifiers, chunk)
                summaries += [StructuredResponse.model_validate_json(content)] # type: ignore
            if nchunks == 1:
                final_summary = summaries[0]
            else:
                # TODO: rip out the entities from all of the summaries, combine them manually, and glue them onto the final summary
                final_summary = summarize_summaries(StructuredResponse, tokenizer, lambda input: self._get_output(config, template, classifiers, input), config.chunk_size, config.overlap, summaries)
            if config.debug:
                logger.debug(final_summary.model_dump_json())
            main_detection_properties = {
                'TEXT': final_summary.summary
            }
            if hasattr(final_summary, 'primary_topic'):
                main_detection_properties['PRIMARY_TOPIC'] = final_summary.primary_topic
            if hasattr(final_summary, 'other_topics'):
                main_detection_properties['OTHER_TOPICS'] = ', '.join(final_summary.other_topics)
            if hasattr(final_summary, 'entities'):
                for (k,v) in final_summary.entities.__dict__.items():
                    main_detection_properties[k.upper()] = ', '.join(v)
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
            classifier_confidence_minimum = config.classifier_confidence_minimum
            results += list(
                map(
                    self._get_classifier_track(video_job),
                    filter(
                        lambda classifier: classifier.confidence >= classifier_confidence_minimum,
                        final_summary.classifiers)
                )
            )
            logger.info(f'get_detections_from_all_video_tracks found: {len(results)} detections')
            if config.debug:
                logger.debug(f'get_detections_from_all_video_tracks results: {results}')
            return results

        else:
            raise mpf.DetectionError.MPF_OTHER_DETECTION_ERROR_TYPE.exception('Received no feed forward tracks')

    def get_detections_from_audio(self, job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        logger.info(f'Received audio job.')

        raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(f'Audio detection not supported.')

def run_component_test(clientFactory = None):
    qsc = LlmSpeechSummaryComponent(clientFactory)
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
