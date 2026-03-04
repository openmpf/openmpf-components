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

from pydantic import BaseModel, Field, ConfigDict, create_model
from typing import List

class EntitiesObject(BaseModel):
    """
    An object containing lists of entities of interest that are present in your input.

    COMBINATION_INSTRUCTION: In your output, lists of strings (entities, topics, etc.) should each be the union of the corresponding fields in your input objects.
    COMBINATION_INSTRUCTION: when combining multiple EntitiesObject objects, treat each field on it as a set of unique strings")
    """
    model_config = ConfigDict(extra='forbid')
    names_of_people: List[str] = Field(
        title="Unique names of people referred to in your input",
        description="CLARIFICATION: only include people referred to in the conversation. Unless the speakers use each others' names or refer to each other somehow in an utterance, do not include the speakers."
    )
    places: List[str] = Field(title="Unique places referred to in your input")
    companies: List[str] = Field(title="Unique names of companies, businesses, and/or institutions in your input")
    body_parts: List[str] = Field(title="Unique parts of the human body named in your input")
    emotions: List[str] = Field(title="Unique emotions, feelings, and/or sentiments referred to in your input")

class Classifier(BaseModel):
    """
    One classifier object

    COMBINATION_INSTRUCTION: Combine confidences and reasonings, with higher confidence inputs (and the corresponding reasonings) receiving precedence.
    """
    model_config = ConfigDict(extra='forbid')
    confidence: float = Field(title='confidence', description='How confident you are that there is a TRUE POSITIVE for this classifier in the input you are summarizing. 0 indicates a confident TRUE NEGATIVE.', ge=0, le=1)
    reasoning: str = Field(title='reasoning', description="INSTRUCTION: If the definition of this classifier included a 'Specific Items of Interest' appendage, please make sure to note the presence of any of those specific items of interest in this field, independent of their inclusion or exclusion in any entities category. COMBINATION INSTRUCTION: include the union of your inputs' items of interest in your output's reasoning.")

def StructuredResponseClassFactory(classifiers):
    classifier_fields = {x: (Classifier, Field()) for x in classifiers.keys()}
    config = ConfigDict(extra='forbid', strict=True)
    Classifiers = create_model('Classifiers', __config__=config, **classifier_fields) # type: ignore
    fields = {
        'summary': (str, Field(title='summary of conversation', description="INSTRUCTION: summarize the conversation with one or more precise, declarative statements about the gestalt of the conversation. COMBINATION_INSTRUCTION: only combine the summaries of your input. Do not cross-contaminate your summary with any other pieces of your input objects.")),
        'primary_topic': (str, Field(title='The primary topic of conversation')),
        'other_topics': (List[str], Field(title='Other topics of conversation', description="INSTRUCTION: do not include the primary_topic in this list")),
        'classifiers': (Classifiers, Field()),
        'entities': (EntitiesObject, Field())
    }
    return create_model('StructuredResponse', __config__=config, **fields)