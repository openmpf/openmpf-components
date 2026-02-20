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

from pydantic import BaseModel, Field
from typing import List

class EntitiesObject(BaseModel):
    """
    An object containing lists of entities of interest

    COMBINATION_INSTRUCTION: In your output, lists of strings (entities, topics, etc.) should each be the union of the corresponding fields in your input objects.
    """
    names_of_people: List[str] = Field(
        default=[],
        title="A list of people's names",
        description="CLARIFICATION: only include people referred to in the conversation. Unless the speakers use each others' names or refer to each other somehow in an utterance, do not include the speakers."
    )
    places: List[str] = Field(default=[], title="Names of or references to specific places")
    companies: List[str] = Field(default=[], title="Names of or references to companies, businesses, and/or institutions")
    body_parts: List[str] = Field(default=[], title="Parts of the human body: organs or otherwise")
    emotions: List[str] = Field(default=[], title="Emotions, feelings, and/or sentiments")

class Classifier(BaseModel):
    """
    One classifier object

    COMBINATION_INSTRUCTION: You must combine classifiers with the same name, such that classifier names are unique in your output. Combine confidences and reasonings, with higher confidence inputs (and the corresponding reasonings) receiving precedence.
    """
    classifier: str = Field(title='name', description="the name of this classifier")
    confidence: float = Field(title='confidence', description='How confident you are in the presence or absence of this classifier in the input you are summarizing', ge=0, le=1)
    reasoning: str = Field(title='reasoning', description="INSTRUCTION: If the definition of this classifier included a 'Specific Items of Interest' appendage, please make sure to note the presence of any of those specific items of interest in this field, independent of their inclusion or exclusion in any entities category. COMBINATION INSTRUCTION: include the union of your inputs' items of interest in your output's reasoning.")

class StructuredResponse(BaseModel):
    summary: str = Field(title='summary of conversation', description="INSTRUCTION: summarize the conversation with one or more precise, declarative statements about the gestalt of the conversation")
    primary_topic: str = Field(title='The primary topic of conversation')
    other_topics: List[str] = Field(title='Other topics of conversation', description="INSTRUCTION: do not include the primary_topic in this list")
    classifiers: List[Classifier] = Field(title='A list of classifier results', description="INSTRUCTION: produce based on the Classifiers between <classifiers></classifiers>. Do not create or infer new classifier categories that are not specified below. Include all classifier categories in your response, even those that have very low confidence.")
    entities: EntitiesObject

response_format_json_schema = StructuredResponse.model_json_schema()