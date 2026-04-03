from pydantic import BaseModel
from typing import Union

class QuestionAnswer(BaseModel):
    question: str
    feedback: str | int
    complies: bool