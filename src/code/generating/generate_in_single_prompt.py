from src.data.constants import BASE_PATH
import os
from . import generating_logging
import logging
from src.code.functions import get_prompt, get_texts_not_divided, get_topics, save_used_prompts_not_divided, make_prompt

logger = logging.getLogger(__name__)
logger.info("STARTED GENERATING RESULTS FOR REFINING IN A SINGLE PROMPT")

single_prompt = get_prompt("REFINE_IN_A_SINGLE_PROMPT")
texts = get_texts_not_divided()
topics = get_topics()
prompts = {}

for text_id in texts:
    prompts[text_id] = single_prompt.replace("{THESIS_TOPIC}", topics[text_id]).replace("{TEXT}", texts[text_id])

# -----------------  DEFINE CONSTANTS  -----------------

PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", "single_prompt_testing_01", "prompts")       + os.path.sep
RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", "single_prompt_testing_01", "responses")     + os.path.sep
RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "single_prompt_testing_01", "raw_responses") + os.path.sep
MODEL_ROLE         = "a very helpful tutor"
MODELS = {
	"gemma4:26b-a4b-it-q4_K_M": "gemma4-26b-q4"
}
TEMPERATURES = [0]
ITERATIONS = 1

os.makedirs(PROMPTS_PATH      , exist_ok=True)
os.makedirs(RAW_RESPONSES_PATH, exist_ok=True)
os.makedirs(RESPONSES_PATH    , exist_ok=True)

save_used_prompts_not_divided(PROMPTS_PATH, prompts)

logger.info("START PROMPTING ")

for model in MODELS:
  logger.info(f"Working on model '{MODELS[model]}'.")
  for t in TEMPERATURES:
    logger.info(f"Temperature = {t}.")
    for i in range(ITERATIONS):
      iter = str(i + 1).zfill(2)
      logger.info(f"Iteration Nr. {iter:2}")
      for p in sorted(prompts):
        logger.info(f"Text Nr. {p}")

        t_string = f't{str(t).replace('.', '-')}'
        raw_responses_dir = os.path.join(RAW_RESPONSES_PATH, MODELS[model], t_string, iter) + os.path.sep
        responses_dir     = os.path.join(RESPONSES_PATH    , MODELS[model], t_string, iter) + os.path.sep

        make_prompt(
          text=prompts[p],
          model=model,
          model_role=MODEL_ROLE,
          temperature=t,
          raw_response_model_path=raw_responses_dir,
          response_model_path=responses_dir,
          fname=f"{p}_full_{MODELS[model]}_{t_string}_{iter}.json",
          to_think=False, # disable for gemma4:26b-a4b-it-q4_K_M
          num_ctx=8196
        )

# python -m src.code.generating.generate_in_single_prompt

# JUST a workaround to start not from zero if session ends abnormally
#if (iter == 9 and t == 0.5) and p < 59 or (t == 0.5 and iter < 9):
#  continue
