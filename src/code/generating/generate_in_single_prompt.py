from src.data.constants import BASE_PATH
import os
from . import generating_logging
import logging
from src.code.functions import get_prompt, get_texts_not_divided, get_topics, prepare_prompts_single, save_used_prompts_not_divided, make_prompt

logger = logging.getLogger(__name__)
logger.info("STARTED GENERATING RESULTS FOR REFINING IN A SINGLE PROMPT")

REFINE_EVERYTHING = get_prompt("REFINE_EVERYTHING_AT_ONCE")
introduction_texts_divided = get_texts_not_divided()
introduction_topics = get_topics()
prompts_single_refinement = prepare_prompts_single(
    introduction_texts_divided, introduction_topics,
    REFINE_EVERYTHING
)

# ----------------- 4. EXPERIMENTS -----------------
# ----------------- 4.1 DEFINE CONSTANTS  -----------------

PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", "single_prompt_testing_01", "prompts")       + os.path.sep
RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", "single_prompt_testing_01", "responses")     + os.path.sep
RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "single_prompt_testing_01", "raw_responses") + os.path.sep
MODEL_ROLE         = "a very helpful tutor"
MODELS = {
	"gpt-oss:20b": "gpt-oss-20b-thinking"
}
TEMPERATURES = [1]
ITERATIONS = 2

os.makedirs(PROMPTS_PATH      , exist_ok=True)
os.makedirs(RAW_RESPONSES_PATH, exist_ok=True)
os.makedirs(RESPONSES_PATH    , exist_ok=True)

save_used_prompts_not_divided(PROMPTS_PATH, prompts_single_refinement)

logger.info("START PROMPTING ")

for model in MODELS:
  logger.info(f"Working on model '{MODELS[model]}'.")
  for t in TEMPERATURES:
    logger.info(f"Temperature = {t}.")
    for i in range(ITERATIONS):
      iter = str(i + 1).zfill(2)
      logger.info(f"Iteration Nr. {iter:2}")
      for p in sorted(prompts_single_refinement):
        logger.info(f"Text Nr. {p}")

        t_string = f't{str(t).replace('.', '-')}'
        raw_responses_dir = os.path.join(RAW_RESPONSES_PATH, MODELS[model], t_string, iter) + os.path.sep
        responses_dir     = os.path.join(RESPONSES_PATH    , MODELS[model], t_string, iter) + os.path.sep

        make_prompt(
          text=prompts_single_refinement[p],
          model=model,
          model_role=MODEL_ROLE,
          temperature=t,
          raw_response_model_path=raw_responses_dir,
          response_model_path=responses_dir,
          fname=f"{p}_full_{MODELS[model]}_{t_string}_{iter}.json"
        )

# JUST a workaround to start not from zero if session ends abnormally
#if (iter == 9 and t == 0.5) and p < 59 or (t == 0.5 and iter < 9):
#  continue
