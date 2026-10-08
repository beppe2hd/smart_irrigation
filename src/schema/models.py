from schema.enum import Features
from typing import List

from pydantic import BaseModel


class TrainingConfiguration(BaseModel):
    feature: List[Features]


class InferenceConfiguration(BaseModel):
    feature: List[Features]
