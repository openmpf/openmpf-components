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

import json
from typing import Any, List
import pandas as pd
import io
from math import inf

def _chunk_within_limits(total_count: int, chunk_size: int, overlap: int, token_count_at_boundaries: List[int], min_grouping: int|None, get_partial_chunk = None, convert_chunk_for_output = lambda x: x):
    if not min_grouping:
        min_grouping = -1
    chunks = []
    chunk_data = []
    chunk_tokens = 0
    overlap_items = 0

    for i in range(0, total_count):
        if token_count_at_boundaries[i] + chunk_tokens <= chunk_size or len(chunk_data) - overlap_items < min_grouping:
            chunk_data.append(get_partial_chunk(i)) # type: ignore
            chunk_tokens += token_count_at_boundaries[i]
        else:
            # When the limit is hit, finalize the current chunk
            if chunk_data:
                chunks.append(convert_chunk_for_output(chunk_data))

            # Start the new chunk with overlap
            # Determine how many rows from the end of the last chunk to include in the new one
            overlap_rows = []
            overlap_count = 0
            overlap_items = 0
            for overlap_row in reversed(chunk_data):
                # Approximation for row overlap token count
                overlap_count += token_count_at_boundaries[i]
                if overlap_count < overlap:
                    overlap_rows.insert(0, overlap_row)
                    overlap_items += 1
                else:
                    break
            
            chunk_data = overlap_rows + [get_partial_chunk(i)] # type: ignore
            chunk_tokens = overlap_count
    if chunk_data:
        chunks.append(convert_chunk_for_output(chunk_data))

    return chunks

def split_csv_into_chunks(tokenizer, text: str, chunk_size: int = 10000, overlap: int = 500, min_grouping=-1):
    newline_token_id = tokenizer.encode('<|newline|>')[0]
    token_ids = tokenizer.encode(text.replace('\r\n', '\n').replace('\n', '<|newline|>'))
    # find all the newlines in the tokenized text
    token_count_before_line = [index for index, element in enumerate(token_ids) if element == newline_token_id]
    token_count_at_line = [x for x in token_count_before_line]
    for i in range(1, len(token_count_at_line)):
        token_count_at_line[i] -= token_count_at_line[i-1]

    df = pd.read_csv(io.StringIO(tokenizer.decode(token_ids).replace('<|newline|>', '\n')),sep='|')
    
    total_rows = len(df)

    def convert_chunk_to_csv(chunk_data):
        chunk_buffer = io.StringIO()
        pd.DataFrame(chunk_data).to_csv(chunk_buffer, index=False, sep='|')
        return chunk_buffer.getvalue()

    return _chunk_within_limits(total_rows, chunk_size, overlap, token_count_at_line, min_grouping, lambda i: df.iloc[i], convert_chunk_to_csv) # type: ignore

def split_array_into_chunks(tokenizer, arr: List[Any], chunk_size: int = 10000, overlap: int = 500, min_grouping=-1):
    for i in range(0, len(arr)):
        if type(arr[i]) is not str:
            arr[i] = arr[i].json() if hasattr(arr[i], 'json') else json.dumps(arr[i])
    # serialize each object separately so we can insert newline tokens to facilitate letting the tokenizer
    # count for us

    newline_token_id = tokenizer.encode('<|newline|>')[0]
    token_ids = tokenizer.encode('[' + (',<|newline|>'.join(arr)) + ',<|newline|>{}]')
    # find all the newlines in the tokenized text
    token_count_before_obj = [index for index, element in enumerate(token_ids) if element == newline_token_id]
    token_count_at_obj = token_count_before_obj
    for i in range(1, len(token_count_at_obj)):
        token_count_at_obj[i] -= token_count_at_obj[i-1]

    total_objects = len(arr)

    return _chunk_within_limits(total_objects, chunk_size, overlap, token_count_at_obj, min_grouping, lambda i: arr[i])

def split_into_chunks(tokenizer, text: str, chunk_size: int = 10000, overlap: int = 500):
    chunks = []
    token_ids = tokenizer.encode(text)
    for i in range(0, len(token_ids), chunk_size - overlap):
        chunk_token_ids = token_ids[i:i + chunk_size]
        chunks.append(chunk_token_ids)

    decoded = [tokenizer.decode(chunk) for chunk in chunks]
    return decoded

def summarize_summaries(model, tokenizer, get_output, chunk_size, overlap, summaries):
    print(f'Summarizing {len(summaries)} summaries...')

    # bisecting or n-secting the chunks is probably a smarter way to handle this... but greedy for now

    # based
    if len(summaries) == 1:
        return summaries[0]

    # TODO: evaluate minimum grouping factors?
    chunks = split_array_into_chunks(tokenizer, summaries, chunk_size, overlap, min_grouping=2)
    results = []
    for chunk in chunks:
        if not model:
            results.append(json.loads(get_output(chunk)))
        else:
            results.append(model.model_validate_json(get_output(chunk))) # type: ignore
    return summarize_summaries(model, tokenizer, get_output, chunk_size, overlap, results)