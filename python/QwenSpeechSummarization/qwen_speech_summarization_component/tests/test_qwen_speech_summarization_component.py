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

from qwen_speech_summarization_component.qwen_speech_summarization_component import run_component_test

class FakeClass():
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            self.__dict__[k] = v

class FakeCompletions():
    # builds an array that emulates the streaming event from the LLM
    def create(self, *args, **kwargs):
        return [
            FakeClass(choices=[FakeClass(finish_reason=None,
                                 delta=FakeClass(content="""{
  "summary": "The conversation centers on the experience of switching between languages during communication, particularly focusing on the comfort and cognitive effort involved when speaking in different languages. One speaker reflects on how language use depends on context and the person they are speaking with, noting that they adapt their language based on familiarity and environment. The other speaker confirms that they always speak English with this person, while using other languages with others. They discuss the challenges of translating jokes or culturally specific expressions, emphasizing that some ideas or humor do not translate well. The speakers also reflect on the novelty of recording this conversation in a multilingual format, acknowledging it as a unique and potentially more challenging experience than expected.",
  "primary_topic": "Language switching and communication comfort in multilingual interactions",
  "other_topics": [
    "Cultural and linguistic adaptation in personal relationships",
    "Challenges of translating humor and idiomatic expressions",
    "The cognitive effort of multilingual conversation"
  ],
  "classifications": [
    {
      "classification": "Major League Baseball",
      "reasoning": "No mention of Major League Baseball, professional baseball players, or baseball stadiums was made in the conversation.",
      "confidence": 0.0
    }
  ],
  "entities": {
    "people": [],
    "places": [],
    "companies": [],
    "businesses": [],
    "body_parts": [],
    "organs": [],
    "emotions": [
      "awkwardness",
      "comfort",
      "confusion",
      "curiosity"
    ]
  }
}"""))], object="chat.completion.chunk"),
            FakeClass(choices=[FakeClass(finish_reason=True)]),
        ]

class FakeChat():
    def __init__(self):
        self.completions = FakeCompletions()

class FakeLLM():
    def __init__(self):
        self.chat = FakeChat()

def test_invocation_with_fake_client():
    result = run_component_test(FakeLLM)
    assert len(result) == 2
    main_detection = result[0]
    classifier_detection = result[1]
    assert main_detection.detection_properties['TEXT'] == "The conversation centers on the experience of switching between languages during communication, particularly focusing on the comfort and cognitive effort involved when speaking in different languages. One speaker reflects on how language use depends on context and the person they are speaking with, noting that they adapt their language based on familiarity and environment. The other speaker confirms that they always speak English with this person, while using other languages with others. They discuss the challenges of translating jokes or culturally specific expressions, emphasizing that some ideas or humor do not translate well. The speakers also reflect on the novelty of recording this conversation in a multilingual format, acknowledging it as a unique and potentially more challenging experience than expected."
    assert classifier_detection.detection_properties['CLASSIFICATION'] == 'Major League Baseball'
    assert classifier_detection.confidence == 0.0