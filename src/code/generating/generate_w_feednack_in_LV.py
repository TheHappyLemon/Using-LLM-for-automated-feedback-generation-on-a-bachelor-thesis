from src.data.constants import BASE_PATH
import os
from . import generating_logging
import logging
from src.code.functions import get_prompt, get_texts, get_topics, prepare_prompts, save_used_prompts, make_prompt

logger = logging.getLogger(__name__)
logger.info("STARTED GENERATING RESPONSES WITH FEEDBACK IN LATVIAN LANGUGE")

LV_PROMPTS_PATH = os.path.join(BASE_PATH, "src", "data", "prompts", "feedback_in_lv")

REFINE_BEFORE_GOAL            = get_prompt("REFINE_BEFORE_GOAL", path=LV_PROMPTS_PATH)
REFINE_GOAL_WITH_PRECEDING    = get_prompt("REFINE_GOAL_WITH_PRECEDING_TEXT", path=LV_PROMPTS_PATH)
REFINE_GOAL_WITHOUT_PRECEDING = get_prompt("REFINE_GOAL_WITHOUT_PRECEDING_TEXT", path=LV_PROMPTS_PATH)
REFINE_TASKS_WITH_GOAL        = get_prompt("REFINE_TASKS_WITH_GOAL", path=LV_PROMPTS_PATH)
REFINE_TASKS_WITHOUT_GOAL     = get_prompt("REFINE_TASKS_WITHOUT_GOAL", path=LV_PROMPTS_PATH)
REFINE_AFTER_TASKS            = get_prompt("REFINE_AFTER_TASKS", path=LV_PROMPTS_PATH)

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

PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", "lv_feedback_testing_01", "prompts")       + os.path.sep
RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", "lv_feedback_testing_01", "responses")     + os.path.sep
RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "lv_feedback_testing_01", "raw_responses") + os.path.sep
MODEL_ROLE         = "a very helpful tutor"
MODELS = {
	"gemma4:26b-a4b-it-q4_K_M": "gemma4-26b-q4"
}
TEMPERATURE = 0
ITERATIONS = 1
DISABLED_THINKING = ['qwen3.5-9b-q8', 'gemma4-26b-q4', 'ministral3-14b-q8']

os.makedirs(PROMPTS_PATH, exist_ok=True)
os.makedirs(RAW_RESPONSES_PATH, exist_ok=True)
os.makedirs(RESPONSES_PATH    , exist_ok=True)

save_used_prompts(PROMPTS_PATH, prompts_part_refinement)

logger.info("START PROMPTING ")

# python -m src.code.generating.generate_w_feednack_in_LV

for model in MODELS:
    logger.info(f"Working on model '{MODELS[model]}'.")
    to_think = False

    for p in sorted(prompts_part_refinement):
        logger.info(f"Text Nr. {p}")

        raw_responses_dir = os.path.join(RAW_RESPONSES_PATH, MODELS[model]) + os.path.sep
        responses_dir     = os.path.join(RESPONSES_PATH    , MODELS[model]) + os.path.sep

        logger.info("BeforeGoal")
        make_prompt(
            text=prompts_part_refinement[p]["BeforeGoal"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=TEMPERATURE,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_BeforeGoal_{MODELS[model]}.json",
            to_think=to_think
        )
        logger.info("Goal")

        make_prompt(
            text=prompts_part_refinement[p]["Goal"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=TEMPERATURE,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_Goal_{MODELS[model]}.json",
            to_think=to_think
        )
        logger.info("Tasks")
        make_prompt(
            text=prompts_part_refinement[p]["Tasks"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=TEMPERATURE,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_Tasks_{MODELS[model]}.json",
            to_think=to_think
        )
        logger.info("AfterTasks")
        make_prompt(
            text=prompts_part_refinement[p]["AfterTasks"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=TEMPERATURE,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_AfterTasks_{MODELS[model]}.json",
            to_think=to_think
        )

# JUST a workaround to start not from zero if session ends abnormally
#if (iter == 9 and t == 0.5) and p < 59 or (t == 0.5 and iter < 9):
#  continue
