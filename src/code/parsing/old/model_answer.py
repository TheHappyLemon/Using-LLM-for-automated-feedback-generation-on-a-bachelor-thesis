from pydantic import BaseModel, model_validator
import logging

logger = logging.getLogger(__name__)

class QuestionAnswer(BaseModel):
    question: str
    feedback: str | int
    complies: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_missing_complies(cls, data):
        if isinstance(data, dict) and "complies" not in data:
            logger.warning(f"Question '{data['question']}' is missing 'complies'. Default value of False will be used")
        return data
