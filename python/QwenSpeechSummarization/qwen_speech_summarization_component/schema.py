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

from pydantic import BaseModel
from typing import List

class EntitiesObject(BaseModel):
    names_of_people: List[str]
    places: List[str]
    companies: List[str]
    body_parts: List[str]
    organs: List[str]
    emotions: List[str]

class Classification(BaseModel):
    Classifier: str
    Confidence: float
    Reasoning: str

class StructuredResponse(BaseModel):
    summary: str
    primary_topic: str
    other_topics: List[str]
    classifications: List[Classification]
    entities: EntitiesObject

response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "StructuredResponse",
        "schema": StructuredResponse.model_json_schema(),
        "strict": True
    }
}