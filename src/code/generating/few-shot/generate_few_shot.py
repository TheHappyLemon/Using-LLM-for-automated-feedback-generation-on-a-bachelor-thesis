from src.data.constants import BASE_PATH
from src.code.parsing.old.EvaluationDataset import EvaluationDataset
import os
import os
import argparse
from pathlib import Path
from src.code.generating import generating_logging
import logging
from src.code.functions import get_prompt, get_texts, get_topics, save_used_prompts_not_divided, make_prompt, prepare_prompts, save_used_prompts

logger = logging.getLogger(__name__)
logger.info("STARTED GENERATING RESULTS FOR REFINING IN FEW-SHOT")

HUMANS = ["human1", "human2", "human3"]
SHOTS = ["2-shot", "3-shot", "4-shot"]
PARTS = {"BeforeGoal" : "BEORE_GOAL", "Goal" : "GOAL", "Tasks" : "TASKS", "AfterTasks" : "AFTER_TASKS"}
EXAMPLES = [1, 5, 7, 43]
FEW_SHOTS_PATH = os.path.join(BASE_PATH, "src", "data", "prompts", "few-shot")

def main(run_id : str):

  HUMAN_RESPONSES_DIR = Path("src/results/human")
  human1_ds = EvaluationDataset("human1")
  human2_ds = EvaluationDataset("human2")
  human3_ds = EvaluationDataset("human3")
  human1_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human1_orig.csv")
  human2_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human2_orig.csv")
  human3_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human3_orig.csv")
  human1_ds.to_bool()
  human2_ds.to_bool(quantity_already_bool=True)
  human3_ds.to_bool()

  for SHOT in SHOTS:

    SHOT_PATH = os.path.join(FEW_SHOTS_PATH, SHOT)

    REFINE_BEFORE_GOAL            = get_prompt("REFINE_BEFORE_GOAL", path=SHOT_PATH)
    REFINE_GOAL_WITH_PRECEDING    = get_prompt("REFINE_GOAL_WITH_PRECEDING_TEXT", path=SHOT_PATH)
    REFINE_GOAL_WITHOUT_PRECEDING = get_prompt("REFINE_GOAL_WITHOUT_PRECEDING_TEXT", path=SHOT_PATH)
    REFINE_TASKS_WITH_GOAL        = get_prompt("REFINE_TASKS_WITH_GOAL", path=SHOT_PATH)
    REFINE_TASKS_WITHOUT_GOAL     = get_prompt("REFINE_TASKS_WITHOUT_GOAL", path=SHOT_PATH)
    REFINE_AFTER_TASKS            = get_prompt("REFINE_AFTER_TASKS", path=SHOT_PATH)

    introduction_texts_divided = get_texts()
    topics = get_topics()

    prompts_part_refinement = prepare_prompts(
        introduction_texts_divided, topics,
        REFINE_BEFORE_GOAL, REFINE_GOAL_WITH_PRECEDING,
        REFINE_GOAL_WITHOUT_PRECEDING, REFINE_TASKS_WITH_GOAL,
        REFINE_TASKS_WITHOUT_GOAL, REFINE_AFTER_TASKS
    )

    for h in HUMANS:
      for p in prompts_part_refinement:
        for e in EXAMPLES:
          for part in PARTS:
            THESIS_TOPIC = "{THESIS_TOPIC_" + str(e) + "}" # THESIS_TOPIC_X
            PART_X = "{" + PARTS[part] + "_" + str(e) + "}"
            CONGRUENT_OR = "{CONGRUENT_OR_NOT_" + str(e) + "}"
            RELEVANT_OR = "{RELEVANT_OR_NOT_GOAL_" + str(e) + "}"
            prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(THESIS_TOPIC, topics[e]).replace(PART_X, introduction_texts_divided[e][part])
            if part == "Goal":
              if h == "human1":
                IS_CONGRUENT = human1_ds.rows[e].goal.Congruence.value > 0
              elif h == "human2":
                IS_CONGRUENT = human2_ds.rows[e].goal.Congruence.value > 0
              elif h == "human3":
                IS_CONGRUENT = human3_ds.rows[e].goal.Congruence.value > 0
              prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(CONGRUENT_OR, " congruent " if IS_CONGRUENT else " not congruent ")
            elif part == "Tasks":
              if h == "human1":
                IS_RELEVANT = human1_ds.rows[e].tasks.Relevance.value > 0
              elif h == "human2":
                IS_RELEVANT = human2_ds.rows[e].tasks.Relevance.value > 0
              elif h == "human3":
                IS_RELEVANT = human3_ds.rows[e].tasks.Relevance.value > 0
              prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(RELEVANT_OR, " relevant " if IS_RELEVANT else " not relevant ")

    del prompts_part_refinement[1]
    del prompts_part_refinement[5]
    del prompts_part_refinement[7]
    del prompts_part_refinement[43]

    exit()

    # -----------------  DEFINE CONSTANTS  -----------------

    PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", "prompts", run_id)       + os.path.sep
    RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", "responses", run_id)     + os.path.sep
    RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", "raw_responses", run_id) + os.path.sep

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

            '''
  
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
          '''
# python -m src.code.generating.few-shot.generate_few_shot
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