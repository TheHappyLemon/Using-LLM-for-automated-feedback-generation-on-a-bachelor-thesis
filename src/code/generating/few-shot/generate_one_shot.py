from src.data.constants import BASE_PATH
import os
import os
import argparse
from src.code.generating import generating_logging
import logging
from src.code.functions import get_prompt, get_texts, get_topics, save_used_prompts_not_divided, make_prompt, prepare_prompts, save_used_prompts

logger = logging.getLogger(__name__)
logger.info("STARTED GENERATING RESULTS FOR REFINING IN ONE-SHOT")
CONTEXT = None

def is_positive_context():
  return CONTEXT == "POSITIVE"

def is_negative_context():
  return CONTEXT == "NEGATIVE"

def main(run_id : str):

  logger.info(f"Running with {CONTEXT} context & id {run_id}")

  if is_negative_context():
    one_shot_path = os.path.join(BASE_PATH, "src", "data", "prompts", "few-shot", "1-shot", "negative")
  elif is_positive_context():
    one_shot_path = os.path.join(BASE_PATH, "src", "data", "prompts", "few-shot", "1-shot", "positive")
  else:
    logger.error(f"UNKNOWN CONTEXT {CONTEXT}")

  REFINE_BEFORE_GOAL            = get_prompt("REFINE_BEFORE_GOAL", path=one_shot_path)
  REFINE_GOAL_WITH_PRECEDING    = get_prompt("REFINE_GOAL_WITH_PRECEDING_TEXT", path=one_shot_path)
  REFINE_GOAL_WITHOUT_PRECEDING = get_prompt("REFINE_GOAL_WITHOUT_PRECEDING_TEXT", path=one_shot_path)
  REFINE_TASKS_WITH_GOAL        = get_prompt("REFINE_TASKS_WITH_GOAL", path=one_shot_path)
  REFINE_TASKS_WITHOUT_GOAL     = get_prompt("REFINE_TASKS_WITHOUT_GOAL", path=one_shot_path)
  REFINE_AFTER_TASKS            = get_prompt("REFINE_AFTER_TASKS", path=one_shot_path)

  introduction_texts_divided = get_texts()
  topics = get_topics()

  prompts_part_refinement = prepare_prompts(
      introduction_texts_divided, topics,
      REFINE_BEFORE_GOAL, REFINE_GOAL_WITH_PRECEDING,
      REFINE_GOAL_WITHOUT_PRECEDING, REFINE_TASKS_WITH_GOAL,
      REFINE_TASKS_WITHOUT_GOAL, REFINE_AFTER_TASKS
  )

  for p in prompts_part_refinement:
    if is_positive_context():
      prompts_part_refinement[p]["BeforeGoal"] = prompts_part_refinement[p]["BeforeGoal"].replace('{BEFORE_GOAL_43}', introduction_texts_divided[43]['BeforeGoal']).replace('{THESIS_TOPIC_43}', topics[43])
      prompts_part_refinement[p]["Goal"]       = prompts_part_refinement[p]["Goal"].replace('{GOAL_5}', introduction_texts_divided[5]['Goal']).replace('{THESIS_TOPIC_5}', topics[5]).replace('{BEFORE_GOAL_5}', introduction_texts_divided[5]['BeforeGoal'])
      prompts_part_refinement[p]["Tasks"]      = prompts_part_refinement[p]["Tasks"].replace('{TASKS_1}', introduction_texts_divided[1]['Tasks']).replace('{THESIS_TOPIC_1}', topics[1]).replace('{GOAL_1}', introduction_texts_divided[1]['Goal'])
      prompts_part_refinement[p]["AfterTasks"] = prompts_part_refinement[p]["AfterTasks"].replace('{AFTER_TASKS_5}', introduction_texts_divided[5]['AfterTasks']).replace('{THESIS_TOPIC_5}', topics[5])
    elif is_negative_context():
      prompts_part_refinement[p]["BeforeGoal"] = prompts_part_refinement[p]["BeforeGoal"].replace('{BEFORE_GOAL_57}', introduction_texts_divided[57]['BeforeGoal']).replace('{THESIS_TOPIC_57}', topics[57])
      prompts_part_refinement[p]["Goal"]       = prompts_part_refinement[p]["Goal"].replace('{GOAL_57}', introduction_texts_divided[57]['Goal']).replace('{THESIS_TOPIC_57}', topics[57]).replace('{BEFORE_GOAL_57}', introduction_texts_divided[57]['BeforeGoal'])
      prompts_part_refinement[p]["Tasks"]      = prompts_part_refinement[p]["Tasks"].replace('{TASKS_57}', introduction_texts_divided[57]['Tasks']).replace('{THESIS_TOPIC_57}', topics[57]).replace('{GOAL_57}', introduction_texts_divided[57]['Goal'])
      prompts_part_refinement[p]["AfterTasks"] = prompts_part_refinement[p]["AfterTasks"].replace('{AFTER_TASKS_7}', introduction_texts_divided[7]['AfterTasks']).replace('{THESIS_TOPIC_7}', topics[7])

  print(introduction_texts_divided[43]['BeforeGoal'])

  # delete 'shots' from evaluation set
  if is_positive_context():
    del prompts_part_refinement[1] # used for Tasks example
    del prompts_part_refinement[5] # used for Goal,AfterTasks example
    del prompts_part_refinement[43] # used for BeforeGoal example
  elif is_negative_context():
    del prompts_part_refinement[57]
    del prompts_part_refinement[7]


  # -----------------  DEFINE CONSTANTS  -----------------

  if is_positive_context():
    PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", "one_shot_testing_01", "positive", "prompts", run_id)       + os.path.sep
    RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", "one_shot_testing_01", "positive", "responses", run_id)     + os.path.sep
    RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "one_shot_testing_01", "positive", "raw_responses", run_id) + os.path.sep
  elif is_negative_context():
    PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", "one_shot_testing_01", "negative", "prompts", run_id)       + os.path.sep
    RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", "one_shot_testing_01", "negative", "responses", run_id)     + os.path.sep
    RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "one_shot_testing_01", "negative", "raw_responses", run_id) + os.path.sep

  MODEL_ROLE = "a very helpful tutor"
  MODELS = {
    "gemma4:26b-a4b-it-q4_K_M": "gemma4-26b-q4"
  }
  TEMPERATURES = [0]
  ITERATIONS = 1

  os.makedirs(PROMPTS_PATH      , exist_ok=True)
  os.makedirs(RAW_RESPONSES_PATH, exist_ok=True)
  os.makedirs(RESPONSES_PATH    , exist_ok=True)

  save_used_prompts(PROMPTS_PATH, prompts_part_refinement)

  logger.info("START PROMPTING ")

  for model in MODELS:
    logger.info(f"Working on model '{MODELS[model]}'.")
    to_think = False
    for t in TEMPERATURES:
      logger.info(f"Temperature = {t}.")
      for i in range(ITERATIONS):
        iter = str(i + 1).zfill(2)
        logger.info(f"Iteration Nr. {iter:2}")
        for p in sorted(prompts_part_refinement):
          logger.info(f"Text Nr. {p}")

          t_string = f't{str(t).replace('.', '-')}'
          raw_responses_dir = os.path.join(RAW_RESPONSES_PATH) + os.path.sep
          responses_dir     = os.path.join(RESPONSES_PATH) + os.path.sep

          logger.info("BeforeGoal")
          make_prompt(
            text=prompts_part_refinement[p]["BeforeGoal"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=t,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_BeforeGoal_{MODELS[model]}_{t_string}_{iter}.json",
            to_think=to_think,
            num_ctx=8196
          )
          logger.info("Goal")
          make_prompt(
            text=prompts_part_refinement[p]["Goal"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=t,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_Goal_{MODELS[model]}_{t_string}_{iter}.json",
            to_think=to_think,
            num_ctx=8196
          )
          logger.info("Tasks")
          make_prompt(
            text=prompts_part_refinement[p]["Tasks"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=t,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_Tasks_{MODELS[model]}_{t_string}_{iter}.json",
            to_think=to_think,
            num_ctx=8196
          )
          logger.info("AfterTasks")
          make_prompt(
            text=prompts_part_refinement[p]["AfterTasks"],
            model=model,
            model_role=MODEL_ROLE,
            temperature=t,
            raw_response_model_path=raw_responses_dir,
            response_model_path=responses_dir,
            fname=f"{p}_AfterTasks_{MODELS[model]}_{t_string}_{iter}.json",
            to_think=to_think,
            num_ctx=8196
          )

# python -m src.code.generating.few-shot.generate_one_shot
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context selector")
    parser.add_argument(
        "--context",
        type=str,
        choices=["POSITIVE", "NEGATIVE"],
        required=True,
        help="Set the context type"
    )
    parser.add_argument(
        "--run_id",
        type=str,
        required=True,
        help="Run identifier(01, 02, ... 0n)"
    )
    args = parser.parse_args()
    CONTEXT = args.context
    run_id = args.run_id
    main(run_id)