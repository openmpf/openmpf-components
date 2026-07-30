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
import requests
import time
import re
import json

from jinja2 import Environment, FileSystemLoader
from typing import Mapping

from openai import OpenAI, AzureOpenAI

if not os.environ.get("HF_HUB_OFFLINE"): os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoTokenizer

import mpf_component_api as mpf
import mpf_component_util as mpf_util

# If import fails with ., assume we are running in pydebug and the CWD is proper
try:
    from .schema import StructuredResponseClassFactory

    from .llm_util.classifiers import get_classifier_lines, get_classifier_dict
    from .llm_util.slapchop import split_csv_into_chunks, summarize_summaries, BOUNDARY_TOKEN_FOR_COUNTING
    from .llm_util.input_cleanup import convert_feed_forward_inputs_to_csv
except:
    from schema import StructuredResponseClassFactory

    from llm_util.classifiers import get_classifier_lines, get_classifier_dict
    from llm_util.slapchop import split_csv_into_chunks, summarize_summaries, BOUNDARY_TOKEN_FOR_COUNTING
    from llm_util.input_cleanup import convert_feed_forward_inputs_to_csv

class JobConfig:
    def __init__(self, props: Mapping[str, str]):
        # if debug is true will output extra information AND log response stream
        self.debug = mpf_util.get_property(props, 'ENABLE_DEBUG', False)

        self.allow_partial_response = mpf_util.get_property(props, 'ALLOW_PARTIAL_RESPONSE', False)
        self.allow_refusal_response = mpf_util.get_property(props, 'ALLOW_REFUSAL_RESPONSE', False)

        self.llm_model = mpf_util.get_property(props, 'LLM_MODEL', "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
        self.tokenizer_model = mpf_util.get_property(props, 'TOKENIZER_MODEL', self.llm_model)

        self.max_model_len = int(mpf_util.get_property(props, 'MAX_MODEL_LEN', 45000))
        self.chunk_size = int(mpf_util.get_property(props, 'INPUT_TOKEN_CHUNK_SIZE', 1500))
        self.overlap = int(mpf_util.get_property(props, 'INPUT_CHUNK_TOKEN_OVERLAP', 300))

        self.api_token = mpf_util.get_property(props, 'API_TOKEN', "Must be set for anonymous VLLM, but can be anything")

        self.prompt_template = self._get_file_path(mpf_util.get_property(props, 'PROMPT_TEMPLATE', 'templates/prompt.jinja'))

        self.api_uri = \
            mpf_util.get_property(props, 'API_URI', "http://llm-speech-summarization-server:11434/v1")

        self.api_health_uri = \
            mpf_util.get_property(props, 'API_HEALTH_URI', "../health")
        if len(self.api_health_uri) == 0:
            self.api_health_uri = None
        elif '://' not in self.api_health_uri:
            from urllib.parse import urljoin
            self.api_health_uri = urljoin(self.api_uri, self.api_health_uri)

        self.enabled_classifiers = \
            mpf_util.get_property(props, 'CLASSIFIERS_LIST', "ALL")

        # exclude classifiers from output if their confidence is below this threshold
        self.classifier_confidence_minimum = \
            mpf_util.get_property(props, 'CLASSIFIER_CONFIDENCE_MINIMUM', 0.3)

        self.classifiers_path = \
            self._get_file_path(mpf_util.get_property(props, 'CLASSIFIERS_FILE', "classifiers.json"))

        self.azure_client = mpf_util.get_property(props, 'AZURE_CLIENT', False)
        self.azure_api_version = mpf_util.get_property(props, 'AZURE_API_VERSION', "")

    @staticmethod
    def _get_file_path(path: str) -> str:
        expanded_path = os.path.expandvars(path)
        if os.path.exists(expanded_path):
            return expanded_path
        resource = importlib.resources.files(__name__) / expanded_path
        if resource.is_file():
            with importlib.resources.as_file(resource) as f:
                return str(f)
        raise mpf.DetectionError.COULD_NOT_READ_DATAFILE.exception(
            f"{path} does not exist.")

logger = logging.getLogger('LLMSpeechSummaryComponent')

def _log_exception(err: mpf.DetectionError, msg: str|None):
    if msg:
        logger.error(msg)
    return err.exception(msg)

class LlmSpeechSummaryComponent:
    def _get_output(self, config: JobConfig, template, classifiers, input):
        if self.client_factory:
            client_factory = self.client_factory
        else:
            # Determine args based on azure or non-azure client
            if config.azure_client:
                client_factory = lambda: LlmSpeechSummaryComponent._get_openai_api_client_when_server_is_ready(config, base_url=config.api_uri, api_key=config.api_token, api_version=config.azure_api_version)
            else:
                client_factory = lambda: LlmSpeechSummaryComponent._get_openai_api_client_when_server_is_ready(config, base_url=config.api_uri, api_key=config.api_token)
        StructuredResponse = StructuredResponseClassFactory(classifiers)
        response_format_json_schema = StructuredResponse.model_json_schema()
        prompt = template.render(input=input, classifiers=get_classifier_lines(classifiers), response_format_json_schema=response_format_json_schema)
        with client_factory() as client:
            stream = client.chat.completions.create(
                model=config.llm_model, #model_name ## for ollama
                # reasoning_effort='none',
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                stream=True,
                max_completion_tokens=int(math.floor(0.9 * (config.max_model_len - config.chunk_size - config.overlap))),
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
            debug_buf = ""
            success = False
            for event in stream:
                if len(event.choices) == 0:
                    continue
                if event.choices[0].finish_reason == "stop":
                    success = True
                    break
                if event.choices[0].finish_reason == 'content_filter':
                    if event.choices[0].model_extra != None and type(event.choices[0].model_extra) is dict:
                        if 'content_filter_results' in event.choices[0].model_extra.keys():
                            raise _log_exception(mpf.DetectionError.DETECTION_FAILED, f"Received LLM content_filter error - reason: {json.dumps(event.choices[0].model_extra['content_filter_results'])}")
                        raise _log_exception(mpf.DetectionError.DETECTION_FAILED, f"Received LLM content_filter error - model_extra: {json.dumps(event.choices[0].model_extra)}")
                    raise _log_exception(mpf.DetectionError.DETECTION_FAILED, "Received LLM content_filter error - unspecified")
                if event.object == "chat.completion.chunk":
                    if event.choices[0].delta.refusal and len(event.choices[0].delta.refusal) > 0:
                        if not config.allow_refusal_response:
                            raise _log_exception(mpf.DetectionError.DETECTION_FAILED, f"Received LLM refusal: {event.choices[0].delta.refusal}")
                        logger.error(f"Received LLM refusal: {event.choices[0].delta.refusal}")
                    if event.choices[0].delta.content and len(event.choices[0].delta.content) > 0:
                        content += event.choices[0].delta.content
                        if config.debug:
                            debug_buf += event.choices[0].delta.content
                            match = re.search(r"(.*)\n(.*)", debug_buf)
                            while match:
                                # nibble whole lines off the front of debug_buf
                                logger.debug(f'COMPLETION: {match.group(1)}')
                                debug_buf = match.group(2)
                                match = re.search(r"(.*)\n(.*)", debug_buf)
            if debug_buf:
                logger.debug(f'COMPLETION: {debug_buf}')
            if not success:
                if not config.allow_partial_response:
                    raise _log_exception(mpf.DetectionError.DETECTION_FAILED, "LLM completion did not terminate successfully")
                logger.error("LLM completion seems to have not completed successfully")
        return content

    @staticmethod
    def _get_openai_api_client_when_server_is_ready(config, timeout_seconds=300, retry_delay_seconds=5, **kwargs):
        if config.api_health_uri:
            start_time = time.time()
            success = False
            failed_ever = False
            last_error: str|None = None
            while time.time() - start_time < timeout_seconds:
                try:
                    response = requests.get(config.api_health_uri, timeout=retry_delay_seconds)
                    if response.status_code == 200:
                        if failed_ever:
                            logger.info("OpenAI API is now available")
                        success = True
                        break
                    else:
                        failed_ever = True
                        logger.warning(f"Received HTTP{response.status_code} from {config.api_health_uri}")
                except Exception as e:
                    failed_ever = True
                    logger.info(f"Waiting up to {timeout_seconds}s for OpenAI API at {config.api_health_uri} to be healthy. {int(math.floor(time.time() - start_time))}s passed so far")
                    last_error = str(e)
                time.sleep(retry_delay_seconds)

            if not success:
                if last_error:
                    raise _log_exception(mpf.DetectionError.NETWORK_ERROR, last_error)
                raise _log_exception(mpf.DetectionError.NETWORK_ERROR, "Timed out waiting for OpenAI API to be healthy")

        return (AzureOpenAI if config.azure_client else OpenAI)(**kwargs)

    def __init__(self, clientFactory=None):
        self.client_factory = clientFactory

    def get_detections_from_all_image_locations(self, image_job):
        logger.info('Received feed forward image job.')
        config = JobConfig(image_job.job_properties)

        final_summary = self._get_final_summary(image_job, config)
        summary_detection_properties = LlmSpeechSummaryComponent._get_summary_detection_properties(final_summary)

        results = [
            mpf.ImageLocation(
                0, 0, 0, 0, -1, summary_detection_properties
            )
        ]

        if hasattr(final_summary, 'classifiers'):
            for classifier in final_summary.classifiers:
                if classifier[1].confidence >= config.classifier_confidence_minimum:
                    detection_properties = {'CLASSIFIER': classifier[0], 'REASONING': classifier[1].reasoning}
                    results.append(
                        mpf.ImageLocation(
                            0, 0, 0, 0, classifier[1].confidence, detection_properties
                        )
                    )

        logger.info(f'get_detections_from_all_image_locations generated {len(results)} detections')
        return results

    def get_detections_from_all_video_tracks(self, video_job):
        logger.info('Received feed forward video job.')
        config = JobConfig(video_job.job_properties)

        final_summary = self._get_final_summary(video_job, config)
        summary_detection_properties = LlmSpeechSummaryComponent._get_summary_detection_properties(final_summary)

        results = [
            mpf.VideoTrack(
                video_job.start_frame,
                video_job.stop_frame,
                -1,
                {0: mpf.ImageLocation(0, 0, 0, 0, -1, summary_detection_properties)},
                summary_detection_properties
            )
        ]

        if hasattr(final_summary, 'classifiers'):
            for classifier in final_summary.classifiers:
                if classifier[1].confidence >= config.classifier_confidence_minimum:
                    detection_properties = {'CLASSIFIER': classifier[0], 'REASONING': classifier[1].reasoning}
                    results.append(
                        mpf.VideoTrack(
                            video_job.start_frame,
                            video_job.stop_frame,
                            classifier[1].confidence,
                            {0: mpf.ImageLocation(0, 0, 0, 0, classifier[1].confidence, detection_properties)},
                            detection_properties
                        )
                    )

        logger.info(f'get_detections_from_all_video_tracks generated {len(results)} tracks')
        return results

    def get_detections_from_all_audio_tracks(self, audio_job):
        logger.info('Received feed forward audio job.')
        config = JobConfig(audio_job.job_properties)

        final_summary = self._get_final_summary(audio_job, config)
        summary_detection_properties = LlmSpeechSummaryComponent._get_summary_detection_properties(final_summary)

        results = [
            mpf.AudioTrack(
                audio_job.start_time,
                audio_job.stop_time,
                -1,
                summary_detection_properties
            )
        ]

        if hasattr(final_summary, 'classifiers'):
            for classifier in final_summary.classifiers:
                if classifier[1].confidence >= config.classifier_confidence_minimum:
                    detection_properties = {'CLASSIFIER': classifier[0], 'REASONING': classifier[1].reasoning}
                    results.append(
                        mpf.AudioTrack(
                            audio_job.start_time,
                            audio_job.stop_time,
                            classifier[1].confidence,
                            detection_properties
                        )
                    )

        logger.info(f'get_detections_from_all_audio_tracks generated {len(results)} tracks')
        return results

    def get_detections_from_all_generic_tracks(self, generic_job):
        logger.info('Received feed forward generic job.')
        config = JobConfig(generic_job.job_properties)

        final_summary = self._get_final_summary(generic_job, config)
        summary_detection_properties = LlmSpeechSummaryComponent._get_summary_detection_properties(final_summary)

        results = [
            mpf.GenericTrack(
                -1,
                summary_detection_properties
            )
        ]

        if hasattr(final_summary, 'classifiers'):
            for classifier in final_summary.classifiers:
                if classifier[1].confidence >= config.classifier_confidence_minimum:
                    detection_properties = {'CLASSIFIER': classifier[0], 'REASONING': classifier[1].reasoning}
                    results.append(
                        mpf.GenericTrack(
                            classifier[1].confidence,
                            detection_properties
                        )
                    )

        logger.info(f'get_detections_from_all_generic_tracks generated {len(results)} tracks')
        return results

    def _get_final_summary(self, job, config):
        tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_model,
            local_files_only=(os.environ["HF_HUB_OFFLINE"] == "1")
        )
        tokenizer.add_special_tokens({'sep_token': BOUNDARY_TOKEN_FOR_COUNTING})

        env = Environment(loader=FileSystemLoader(os.path.dirname(config.prompt_template)))
        template = env.get_template(os.path.basename(config.prompt_template))

        ff_inputs = None
        if isinstance(job, mpf.AllImageLocationsJob):
            if not job.feed_forward_locations:
                raise _log_exception(mpf.DetectionError.OTHER_DETECTION_ERROR_TYPE, 'Received no feed forward locations')
            ff_inputs = job.feed_forward_locations;
        else:
            if not job.feed_forward_tracks:
                raise _log_exception(mpf.DetectionError.OTHER_DETECTION_ERROR_TYPE, 'Received no feed forward tracks')
            ff_inputs = job.feed_forward_tracks;

        classifiers = get_classifier_dict(config.classifiers_path, config.enabled_classifiers)
        StructuredResponse = StructuredResponseClassFactory(classifiers)

        csv_input = convert_feed_forward_inputs_to_csv(ff_inputs)

        summaries = []
        chunks = split_csv_into_chunks(tokenizer, csv_input, config.chunk_size, config.overlap)
        nchunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            logger.debug(f"chunk [{idx+1} / {nchunks}] ({round(100.0 * (idx+1) / nchunks)}%)")
            content = self._get_output(config, template, classifiers, chunk)
            summaries += [StructuredResponse.model_validate_json(content)]

        if nchunks == 1:
            return summaries[0]

        return summarize_summaries(
            StructuredResponse, tokenizer,
            lambda input: self._get_output(config, template, classifiers, input),
            config.chunk_size, config.overlap, summaries
        )

    @staticmethod
    def _get_summary_detection_properties(final_summary):
        summary_detection_properties = {'TEXT': final_summary.summary}
        if hasattr(final_summary, 'primary_topic'):
            summary_detection_properties['PRIMARY_TOPIC'] = final_summary.primary_topic
        if hasattr(final_summary, 'other_topics'):
            summary_detection_properties['OTHER_TOPICS'] = ', '.join(final_summary.other_topics)
        if hasattr(final_summary, 'entities'):
            for k, v in final_summary.entities.model_dump().items():
                summary_detection_properties[k.upper()] = ', '.join(v)
        return summary_detection_properties
