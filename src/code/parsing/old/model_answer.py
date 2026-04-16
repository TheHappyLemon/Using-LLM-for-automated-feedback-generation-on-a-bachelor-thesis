from pydantic import BaseModel, model_validator
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class QuestionAnswer(BaseModel):
    question: str
    feedback: Optional[str | int] = None
    complies: bool = False

    @model_validator(mode="before")
    @classmethod
    def check_missing_fields(cls, data):
        if isinstance(data, dict):
            if "complies" not in data:
                logger.warning(
                    f"Question '{data.get('question')}' is missing 'complies'. Default value of False will be used"
                )
            if "feedback" not in data:
                logger.warning(
                    f"Question '{data.get('question')}' is missing 'feedback'. Default value of None will be used"
                )
        return data