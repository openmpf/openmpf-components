from pydantic import BaseModel
from typing import List

class EntitiesObject(BaseModel):
    names_of_people: List[str]
    places: List[str]
    companies: List[str]
    businesses: List[str]
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