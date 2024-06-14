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

import mpf_component_api as mpf
import mpf_component_util as mpf_util

from sentence_transformers import SentenceTransformer, util

from typing import Sequence, Dict, Mapping
import os
import time

from pkg_resources import resource_filename
from nltk.tokenize.punkt import PunktSentenceTokenizer
import pandas as pd

logger = logging.getLogger('TransformerTaggingComponent')

class TransformerTaggingComponent:

    def __init__(self):
        self._cached_model = SentenceTransformer('/models/all-mpnet-base-v2')
        self._cached_corpuses: Dict[str, Corpus] = {}


    def get_detections_from_video(self, job: mpf.VideoJob) -> Sequence[mpf.VideoTrack]:
        logger.info(f'Received video job.')

        return self._get_feed_forward_detections(job, job.feed_forward_track, video_job=True)


    def get_detections_from_image(self, job: mpf.ImageJob) -> Sequence[mpf.ImageLocation]:
        logger.info(f'Received image job.')

        return self._get_feed_forward_detections(job, job.feed_forward_location)


    def get_detections_from_audio(self, job: mpf.AudioJob) -> Sequence[mpf.AudioTrack]:
        logger.info(f'Received audio job.')

        return self._get_feed_forward_detections(job, job.feed_forward_track)


    def get_detections_from_generic(self, job: mpf.GenericJob) -> Sequence[mpf.GenericTrack]:
        logger.info(f'Received generic job.')

        if job.feed_forward_track:
            return self._get_feed_forward_detections(job, job.feed_forward_track)
        else:
            logger.info('Job did not contain a feed forward track. Assuming '
                        'media file is a plain text file containing the text to '
                        'be tagged.')

            # preserve line endings in the original text, such as '\r\n'
            with open(job.data_uri, 'r', newline='') as f:
                text = f.read()

            new_ff_props = dict(TEXT=text)
            ff_track = mpf.GenericTrack(detection_properties=new_ff_props)

            new_job_props = {
                **job.job_properties,
                'FEED_FORWARD_PROP_TO_PROCESS': 'TEXT'
            }

            config = JobConfig(new_job_props)
            corpus = self._get_corpus(config.corpus_path)
            self._add_tags(config, corpus, new_ff_props)

            return [ff_track]


    def _get_feed_forward_detections(self, job, job_feed_forward, video_job=False):
        try:
            if job_feed_forward is None:
                raise mpf.DetectionError.UNSUPPORTED_DATA_TYPE.exception(
                    f'Component can only process feed forward '
                    ' jobs, but no feed forward track provided. ')

            config = JobConfig(job.job_properties)
            corpus = self._get_corpus(config.corpus_path)

            self._add_tags(config, corpus, job_feed_forward.detection_properties)

            if video_job:
                for ff_location in job.feed_forward_track.frame_locations.values():
                    self._add_tags(config, corpus, ff_location.detection_properties)

            return [job_feed_forward]

        except Exception:
            logger.exception(
                f'Failed to complete job due to the following exception:')
            raise


    def _get_corpus(self, corpus_path):
        if not corpus_path in self._cached_corpuses:
            self._cached_corpuses[corpus_path] = Corpus(corpus_path, self._cached_model)

        return self._cached_corpuses[corpus_path]


    def _add_tags(self, config, corpus, ff_props: Dict[str, str]):
        input_texts = {}
        for prop_to_tag in config.props_to_process:
            input_text = ff_props.get(prop_to_tag, None)
            if input_text:
                input_texts[prop_to_tag] = input_text

        if not input_texts:
            logger.warning("Feed forward element missing one of the following properties: "
                           + ", ".join(config.props_to_process))
            return
            
        for prop_to_tag, input_text in input_texts.items():
            self._add_tags_for_prop(config, corpus, ff_props, prop_to_tag, input_text)


    def _add_tags_for_prop(self, config, corpus, ff_props: Dict[str, str], prop_to_tag, input_text):
        all_tag_results = []

        # for each sentence in input
        for start, end in PunktSentenceTokenizer().span_tokenize(input_text):
            probe_str = input_text[start:end]

            # split input sentence further on newline or carriage return if flag is set
            if (config.split_on_newline):
                probe_list = probe_str.splitlines(keepends=True)
            else:
                probe_list = [probe_str]

            # an offset counter to track character offset start
            offset_counter = start

            for probe in probe_list:
                # strip probe of leading and trailing whitespace
                stripped_probe = probe.strip()
                
                # determine probe character offsets
                num_leading_chars = len(probe) - len(probe.lstrip())
                offset_start = offset_counter + num_leading_chars 
                offset_end = offset_start + len(stripped_probe) - 1

                # set character offset counter for next iteration
                offset_counter += len(probe)

                if stripped_probe == "":
                    continue

                # get similarity scores for the input sentence with each corpus sentence
                embed_probe = self._cached_model.encode(stripped_probe, convert_to_tensor=True, show_progress_bar=False)
                scores = [float(util.cos_sim(embed_probe, corpus_entry)) for corpus_entry in corpus.embed]

                probe_df = pd.DataFrame({
                    "input text": stripped_probe,
                    "corpus text": corpus.json["text"],
                    "tag": corpus.json["tag"].str.upper(),
                    "score": scores,
                    "offset": str(offset_start) + "-" + str(offset_end)
                })

                # sort by score then group by tag so each group will be sorted highest to lowest score,
                # then take top row for each group
                probe_df = probe_df.sort_values(by=['score'], ascending=False)
                top_per_tag = probe_df.groupby(['tag'], sort=False).head(1)

                # filter out results that are below threshold
                top_per_tag_threshold = top_per_tag[top_per_tag["score"] >= config.threshold]
                all_tag_results.append(top_per_tag_threshold)

        # if no tags found in text return
        if not all_tag_results:
            return

        all_tag_results = pd.concat(all_tag_results)

        ff_tags = set()
        if "TAGS" in ff_props:
            ff_tags = {t.strip().upper() for t in ff_props["TAGS"].split(';')}

        new_tags = set(all_tag_results["tag"].unique())

        ff_props["TAGS"] = "; ".join(sorted(ff_tags.union(new_tags))) # lexicographic order

        # create detection properties for each tag found in the text
        # detection properties formatted as <input property> <tag> TRIGGER SENTENCES...
        for tag in new_tags: 
            tag_df = all_tag_results[all_tag_results["tag"] == tag]

            sents = []
            offsets = []
            scores = []
            matches = []

            for input_text in tag_df["input text"].unique():
                input_text_df = tag_df[tag_df["input text"] == input_text]

                sents.append(input_text.replace(';', '[;]'))
                offsets.append(", ".join(input_text_df["offset"]))
                # all entries should have the same score, so just use the first
                scores.append(input_text_df["score"].values[0].astype(str))

                if config.debug:
                    # all entries should have the same match, so just use the first
                    matches.append(input_text_df["corpus text"].values[0].replace(';', '[;]'))

            prop_name_sent = prop_to_tag + " " + tag + " TRIGGER SENTENCES"
            prop_name_offset = prop_name_sent + " OFFSET"
            prop_name_score = prop_name_sent + " SCORE"

            ff_props[prop_name_sent] = "; ".join(sents)
            ff_props[prop_name_offset] = "; ".join(offsets)
            ff_props[prop_name_score] = "; ".join(scores)

            if config.debug:
                prop_name_matches = prop_name_sent + " MATCHES"
                ff_props[prop_name_matches] = "; ".join(matches)


class Corpus:
    def __init__(self, corpus_path, model):
        self.json = pd.read_json(corpus_path)

        start = time.time()
        self.embed= model.encode(self.json["text"], convert_to_tensor=True, show_progress_bar=False)
        elapsed = time.time() - start
        logger.info(f"Successfully encoded corpus in {elapsed} seconds.")


class JobConfig:
    def __init__(self, props: Mapping[str, str]):

        self.props_to_process = [
            prop.strip() for prop in
            mpf_util.get_property(
                properties=props,
                key='FEED_FORWARD_PROP_TO_PROCESS',
                default_value='TEXT,TRANSCRIPT,TRANSLATION',
                prop_type=str
            ).split(',')
        ]

        self.threshold = mpf_util.get_property(props, 'SCORE_THRESHOLD', .3)

        # if debug is true will return which corpus sentences triggered the match
        self.debug = mpf_util.get_property(props, 'ENABLE_DEBUG', False)

        # if split on newline is true will split input on newline and carriage returns
        self.split_on_newline = mpf_util.get_property(props, 'ENABLE_NEWLINE_SPLIT', False)

        self.corpus_file = \
            mpf_util.get_property(props, 'TRANSFORMER_TAGGING_CORPUS', "transformer_text_tags_corpus.json")

        self.corpus_path = ""
        if "$" not in self.corpus_file and "/" not in self.corpus_file:
            self.corpus_path = os.path.realpath(resource_filename(__name__, self.corpus_file))
        else:
            self.corpus_path = os.path.expandvars(self.corpus_file)

        if not os.path.exists(self.corpus_path):
            logger.exception('Failed to complete job due incorrect file path for the transformer tagging corpus: '
                             f'"{self.corpus_file}"')
            raise mpf.DetectionException(
                'Invalid path provided for transformer tagging corpus: '
                f'"{self.corpus_file}"',
                mpf.DetectionError.COULD_NOT_READ_DATAFILE)
