from pydantic import BaseModel
from typing import List

class ChurnRequest(BaseModel):
    """
    Input features must match training-time order & meaning.
    """
    features: List[float]

class ChurnResponse(BaseModel):
    prediction: str
    probability: float
    threshold: float