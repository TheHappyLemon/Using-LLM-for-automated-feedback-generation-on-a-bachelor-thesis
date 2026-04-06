from src.code.parsing.old.model_answer import QuestionAnswer
import re
import logging

logger = logging.getLogger(__name__)

class Part:

    def __init__(self) -> None:
        self.questions_mapping: dict[str, str] = {}

    # because LLM may output dash not as dash
    def normalize_dashes(self, text: str) -> str:
        pattern = r"[‐-‒–—−]"
        
        matches = re.findall(pattern, text)
        if matches:
            normalized = re.sub(pattern, "-", text)
            logger.warning(
                f"Dashes {set(matches)} replaced in '{text}' -> '{normalized}'"
            )
            return normalized
        
        return text

    def get_question_by_text(self, LLM_text: str) -> str | None:
        LLM_text = LLM_text.strip().lower()
        for key, value in self.questions_mapping.items():
            val = value.strip().lower()
            # exact match OR substring match
            if LLM_text == val or LLM_text in val:
                return key
        return None

    def __extract_quantity__(self, text : str) -> int:

        lowered = text.lower()
        allowed_groups = [
            (r"\b(0|zero)\b", 0),
            (r"\b(1|one)\b", 1),
            (r"\b(2|two)\b", 2),
            (r"\b(3|three)\b", 3),
            (r"\b(4|four)\b", 4),
            (r"\b(5|five)\b", 5),
            (r"\b(6|six)\b", 6),
            (r"\b(7|seven)\b", 7),
            (r"\b(8|eight)\b", 8),
            (r"\b(9|nine)\b", 9),
            (r"\b(10|ten)\b", 10),
            (r"\b(11|eleven)\b", 11),
            (r"\b(12|twelve)\b", 12),
            (r"\b(13|thirteen)\b", 13),
            (r"\b(14|fourteen)\b", 14),
            (r"\b(15|fifteen)\b", 15),
            (r"\b(16|sixteen)\b", 16),
            (r"\b(17|seventeen)\b", 17),
            (r"\b(18|eighteen)\b", 18),
            (r"\b(19|nineteen)\b", 19),
            (r"\b(20|twenty)\b", 20)
        ]
    
        matched_numbers = [num for pattern, num in allowed_groups if re.search(pattern, lowered)]
        if len(matched_numbers) != 1:
            logger.info("It had several or zero integers. Falling back to 'complies' value")
            return 0
        return matched_numbers[0]

    def parse_answer(self, keyword : str, value : str, feedback : str):
        # some specific handlers 
        if keyword.lower() == "quantity":
            value_int = None
            try:
                # if model responded only with number - good. Take it as response.
                value_int = int(feedback)
                value = value_int
            except ValueError as e:
                # if it had other text, but only one number - take this number as response.
                logger.warning(f"Failed to parse 'quantity' question. Field '{feedback}' is not a single integer.")
                value_int = self.__extract_quantity__(str(feedback))
                if value_int != 0:
                    logger.info(f"Found a single integer - '{value_int}'. Using that value as an anwswer.")
                    value = value_int
                # Otherwise leave default 'complies' property
                else:
                    if value:
                        logger.info(f"Complies value was {value} -> setting response to question to 5")
                        value = 5
                    else:
                        logger.info(f"Complies value was '{value}' -> setting response to question to 0")

        elif self.normalize_dashes(keyword.lower()) == "state-of-the-art":
            keyword = "State_of_the_art"
        elif keyword.lower() == "sequence":
            # hack for old results. At some point we removed question 'Sequence' for Tasks part
            raise Exception("Skipping 'sequence' question answer")
        return keyword, value, feedback

    def load(self, json_arr : list[QuestionAnswer], **kwargs):
        for answer in json_arr:
            keyword = answer.question
            value = answer.complies
            feedback = answer.feedback
            try:
                keyword, value, feedback = self.parse_answer(keyword, value, feedback)
                setattr(getattr(self, keyword), "value", value)
                setattr(getattr(self, keyword), "feedback", feedback)
            except AttributeError as ae:
                # Here we handle cases when LLM incorrectly name question it is answering.
                # 1. Check if instead of the question name, e.g. "Significance" it generated the whole question itself.
                logger.error(f"START: {str(ae)}")
                logger.error("Will try to find question by text")
                keyword = self.get_question_by_text(keyword) # type: ignore cant really happen to be None
                if not keyword is None:
                    logger.error(f"END: Question by text found! It is '{keyword}'")
                    keyword, value, feedback = self.parse_answer(keyword, value, feedback)
                    setattr(getattr(self, keyword), "value", value) # type: ignore
                    setattr(getattr(self, keyword), "feedback", feedback) # type: ignore
                else:
                    logger.error("END: Question by text not found :(. It will be set to zero")
            except Exception as e:
                logger.error(str(e))

        mandatory_questions = kwargs.get("mandatory_questions", [])
        if mandatory_questions == []:
            mandatory_questions = list(self.__dict__.keys())
        self.validate_questions(mandatory_questions)

    def validate_questions(self, mandatory_questions : list[str]):
        for q in mandatory_questions:
            if self.__dict__.get(q, None) is None:
                raise Exception(f"Question '{q}' is not in answer!")

    def to_bool(self):
        proprties_to_ignore = ["questions_mapping"]

        for key, obj in self.__dict__.items():
            if key == "Quantity":
                continue
            if key in proprties_to_ignore:
                continue
            setattr(getattr(self, key), "value", bool(obj.value))

    def to_int(self):
        proprties_to_ignore = ["questions_mapping"]

        for key, obj in self.__dict__.items():
            if key in proprties_to_ignore:
                continue
            setattr(getattr(self, key), "value", int(obj.value))

    def __str__(self) -> str:
        proprties_to_ignore = ["questions_mapping"]
        res = str(self.__class__) + "\n"

        for key, obj in self.__dict__.items():
            if key in proprties_to_ignore:
                continue
            res = res + "\t" + key + " : " + str(obj) + "\n"
        return res

    