from src.data.constants import BASE_PATH
import os
from . import generating_logging
import logging
from src.code.functions import get_prompt, get_texts, get_topics, prepare_prompts, save_used_prompts, make_prompt

logger = logging.getLogger(__name__)
logger.info("STARTED")

REFINE_BEFORE_GOAL            = get_prompt("REFINE_BEFORE_GOAL")
REFINE_GOAL_WITH_PRECEDING    = get_prompt("REFINE_GOAL_WITH_PRECEDING_TEXT")
REFINE_GOAL_WITHOUT_PRECEDING = get_prompt("REFINE_GOAL_WITHOUT_PRECEDING_TEXT")
REFINE_TASKS_WITH_GOAL        = get_prompt("REFINE_TASKS_WITH_GOAL")
REFINE_TASKS_WITHOUT_GOAL     = get_prompt("REFINE_TASKS_WITHOUT_GOAL")
REFINE_AFTER_TASKS            = get_prompt("REFINE_AFTER_TASKS")

introduction_texts_divided = get_texts()
introduction_topics = get_topics()
prompts_part_refinement = prepare_prompts(
    introduction_texts_divided, introduction_topics,
    REFINE_BEFORE_GOAL, REFINE_GOAL_WITH_PRECEDING,
    REFINE_GOAL_WITHOUT_PRECEDING, REFINE_TASKS_WITH_GOAL,
    REFINE_TASKS_WITHOUT_GOAL, REFINE_AFTER_TASKS
)

# ----------------- 4. EXPERIMENTS -----------------
# ----------------- 4.1 DEFINE CONSTANTS  -----------------

PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "prompts")       + os.path.sep
RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "responses")     + os.path.sep
RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "raw_responses") + os.path.sep
MODEL_ROLE         = "a very helpful tutor"
MODELS = {
  "qwen3.5:9b-q8_0": "qwen3.5-9b-q8",
  "gemma4:26b-a4b-it-q4_K_M": "gemma4-26b-q4",
  "ministral-3:14b-instruct-2512-q8_0": "ministral3-14b-q8"
}
TEMPERATURES = [0]
ITERATIONS = 10

os.makedirs(PROMPTS_PATH, exist_ok=True)
os.makedirs(RAW_RESPONSES_PATH, exist_ok=True)
os.makedirs(RESPONSES_PATH    , exist_ok=True)

save_used_prompts(PROMPTS_PATH, prompts_part_refinement)

logger.info("START PROMPTING ")

for model in MODELS:
  logger.info(f"Working on model '{MODELS[model]}'.")
  for t in TEMPERATURES:
    logger.info(f"Temperature = {t}.")
    for i in range(ITERATIONS):
      iter = str(i + 1).zfill(2)
      logger.info(f"Iteration Nr. {iter:2}")
      for p in sorted(prompts_part_refinement):
        logger.info(f"Text Nr. {p}")

        t_string = f't{str(t).replace('.', '-')}'
        raw_responses_dir = os.path.join(RAW_RESPONSES_PATH) + os.path.sep
        responses_dir     = os.path.join(RESPONSES_PATH    ) + os.path.sep
        
        logger.info("BeforeGoal")
        make_prompt(
          text=prompts_part_refinement[p]["BeforeGoal"],
          model=model,
          model_role=MODEL_ROLE,
          temperature=t,
          raw_response_model_path=raw_responses_dir,
          response_model_path=responses_dir,
          fname=f"{p}_BeforeGoal_{MODELS[model]}_{t_string}_{iter}.json"
        )
        logger.info("Goal")
        make_prompt(
          text=prompts_part_refinement[p]["Goal"],
          model=model,
          model_role=MODEL_ROLE,
          temperature=t,
          raw_response_model_path=raw_responses_dir,
          response_model_path=responses_dir,
          fname=f"{p}_Goal_{MODELS[model]}_{t_string}_{iter}.json"
        )
        logger.info("Tasks")
        make_prompt(
          text=prompts_part_refinement[p]["Tasks"],
          model=model,
          model_role=MODEL_ROLE,
          temperature=t,
          raw_response_model_path=raw_responses_dir,
          response_model_path=responses_dir,
          fname=f"{p}_Tasks_{MODELS[model]}_{t_string}_{iter}.json"
        )
        logger.info("AfterTasks")
        make_prompt(
          text=prompts_part_refinement[p]["AfterTasks"],
          model=model,
          model_role=MODEL_ROLE,
          temperature=t,
          raw_response_model_path=raw_responses_dir,
          response_model_path=responses_dir,
          fname=f"{p}_AfterTasks_{MODELS[model]}_{t_string}_{iter}.json"
        )

# JUST a workaround to start not from zero if session ends abnormally
#if (iter == 9 and t == 0.5) and p < 59 or (t == 0.5 and iter < 9):
#  continue
