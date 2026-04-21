from src.code.functions import prompt, get_prompt, get_texts_not_divided, get_topics
from src.data.constants import BASE_PATH
from . import generating_logging
import logging
import os

logging.info("TEST LOG MESSAGE DONT MIND ME")

single_prompt = get_prompt("REFINE_IN_A_SINGLE_PROMPT")
texts = get_texts_not_divided()
topics = get_topics()
prompts = {}

my_prompt = "Make a fictional story about cat named kitty. Two sentences only. Format the story as a JSON object"

resp = prompt(system='You are a helpful tutor', user=my_prompt, model='gemma4:26b-a4b-it-q4_K_M', temperature=0, num_ctx=8196, to_think=False, use_schema=True, save_to=os.path.join(BASE_PATH, "src", "code", "generating", "response_raw.json"))

with open(os.path.join(BASE_PATH, "src", "code", "generating", "response.json"), 'w', encoding='utf-8') as f:
    f.write(resp)

# python -m src.code.generating.test
