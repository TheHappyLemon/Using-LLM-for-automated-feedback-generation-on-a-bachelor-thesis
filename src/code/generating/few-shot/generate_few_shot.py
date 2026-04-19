from src.data.constants import BASE_PATH
from src.code.parsing.old.EvaluationDataset import EvaluationDataset
from .json_answer_examples import *
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
PARTS = {"BeforeGoal" : "BEFORE_GOAL", "Goal" : "GOAL", "Tasks" : "TASKS", "AfterTasks" : "AFTER_TASKS"}
EXAMPLES = [1, 5, 7, 43]
FEW_SHOTS_PATH = os.path.join(BASE_PATH, "src", "data", "prompts", "few-shot")

HUMAN_RESPONSES_DIR = Path("src/results/human")
human1_ds = EvaluationDataset("human1")
human2_ds = EvaluationDataset("human2")
human3_ds = EvaluationDataset("human3")
human1_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human1_orig.csv")
human2_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human2_orig.csv")
human3_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human3_orig.csv")


def main(run_id : str):
  
  for h in HUMANS:
    logger.info(f"Working on few-shots for human {h}")
    for SHOT in SHOTS:

      logger.info(f"Current few-shot: {SHOT}")

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

      for p in prompts_part_refinement:
        for e in EXAMPLES:
          for part in PARTS:

            IS_CONGRUENT = False
            IS_RELEVANT = False
            row = None

            THESIS_TOPIC = "{THESIS_TOPIC_" + str(e) + "}" # THESIS_TOPIC_X
            PART_X = "{" + PARTS[part] + "_" + str(e) + "}"
            BEFORE_GOAL_X = "{BEFORE_GOAL_" + str(e) + "}"
            GOAL_X = "{GOAL_" + str(e) + "}"
            CONGRUENT_OR = "{CONGRUENT_OR_NOT_" + str(e) + "}"
            RELEVANT_OR = "{RELEVANT_OR_NOT_GOAL_" + str(e) + "}"

            PART_X_ANSWER_TMPL = "{ANSWER_" + PARTS[part] + "_" + str(e) + "}"
            PART_X_ANSWER = ""

            if part == "BeforeGoal":
              PART_X_ANSWER = BEFORE_GOAL
            if part == "Goal":
              if prompts_part_refinement[p]["BeforeGoal"] != "":
                PART_X_ANSWER = GOAL_WITH_PRECEDING
              else:
                PART_X_ANSWER = GOAL_WITHOUT_PRECEDING
            if part == "Tasks":
              if prompts_part_refinement[p]["Goal"] != "":
                PART_X_ANSWER = TASKS_WITH_GOAL
              else:
                PART_X_ANSWER = TASKS_WITHOUT_GOAL
            if part == "AfterTasks":
              PART_X_ANSWER = AFTER_TASKS

            if h == "human1":
              row = next((r for r in human1_ds.rows if r.Nr == e), None)
            elif h == "human2":
              row = next((r for r in human2_ds.rows if r.Nr == e), None)
            elif h == "human3":
              row = next((r for r in human3_ds.rows if r.Nr == e), None)

            PART_X_ANSWER = PART_X_ANSWER.replace("{SIGNIFICANCE_COMPLIES}", str(row.beforeGoal.Significance.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{STATEOFTHEART_COMPLIES}", str(row.beforeGoal.State_of_the_art.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{GAP_COMPLIES}", str(row.beforeGoal.Gap.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{PROBLEM_COMPLIES}", str(row.beforeGoal.Problem.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{REFERENCES_COMPLIES}", str(row.beforeGoal.References.value > 0).lower())

            PART_X_ANSWER = PART_X_ANSWER.replace("{PURPOSE_COMPLIES}", str(row.goal.Purpose.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{INTENTION_COMPLIES}", str(row.goal.Intention.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{STRUCTURE_GOAL_COMPLIES}", str(row.goal.Structure.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{CONGRUENCE_COMPLIES}", str(row.goal.Congruence.value > 0).lower())

            PART_X_ANSWER = PART_X_ANSWER.replace("{OUTLOOK_COMPLIES}", str(row.tasks.Outlook.value > 0).lower())
            if h == "human2":
              if row.tasks.Quantity.value > 0:
                PART_X_ANSWER = PART_X_ANSWER.replace("{QUANTITY}", "5")
                PART_X_ANSWER = PART_X_ANSWER.replace("{QUANTITY_COMPLIES}", "true")
              else:
                PART_X_ANSWER = PART_X_ANSWER.replace("{QUANTITY}", "0")
                PART_X_ANSWER = PART_X_ANSWER.replace("{QUANTITY_COMPLIES}", "false")
            else:
              PART_X_ANSWER = PART_X_ANSWER.replace("{QUANTITY}", str(row.tasks.Quantity.value).lower())
              PART_X_ANSWER = PART_X_ANSWER.replace("{QUANTITY_COMPLIES}", str(row.tasks.Quantity.value >= 5 and row.tasks.Quantity.value <= 7 ).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{COMPLETENESS_COMPLIES}", str(row.tasks.Completeness.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{FORMAT_COMPLIES}", str(row.tasks.Format.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{STRUCTURE_TASKS_COMPLIES}", str(row.tasks.Structure.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{CLARITY_COMPLIES}", str(row.tasks.Clarity.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{RELEVANCE_COMPLIES}", str(row.tasks.Relevance.value > 0).lower())

            PART_X_ANSWER = PART_X_ANSWER.replace("{CHAPTERS_COMPLIES}", str(row.afterTasks.Chapters.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{DESCRIPTION_COMPLIES}", str(row.afterTasks.Description.value > 0).lower())
            PART_X_ANSWER = PART_X_ANSWER.replace("{STRUCTURE_AFTER_TASKS_COMPLIES}", str(row.afterTasks.Structure.value > 0).lower())

            prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(THESIS_TOPIC, topics[e]).replace(PART_X, introduction_texts_divided[e][part])
            prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(BEFORE_GOAL_X, introduction_texts_divided[e]["BeforeGoal"])
            prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(GOAL_X, introduction_texts_divided[e]["Goal"])
            prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(PART_X_ANSWER_TMPL, PART_X_ANSWER)
            if part == "Goal":
              IS_CONGRUENT = row.goal.Congruence.value > 0
              prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(CONGRUENT_OR, " congruent " if IS_CONGRUENT else " not congruent ")
            elif part == "Tasks":
              IS_RELEVANT = row.tasks.Relevance.value > 0
              prompts_part_refinement[p][part] = prompts_part_refinement[p][part].replace(RELEVANT_OR, " relevant " if IS_RELEVANT else " not relevant ")

      if SHOT == "2-shot":
        del prompts_part_refinement[1]
        del prompts_part_refinement[5]
        logger.info(f"Removed from test set texts 1, 5")
      elif SHOT == "3-shot":
        del prompts_part_refinement[1]
        del prompts_part_refinement[5]
        del prompts_part_refinement[7]
        logger.info(f"Removed from test set texts 1, 5, 7")
      elif SHOT == "4-shot":
        del prompts_part_refinement[1]
        del prompts_part_refinement[5]
        del prompts_part_refinement[7]
        del prompts_part_refinement[43]
        logger.info(f"Removed from test set texts 1, 5, 7, 43")

      # -----------------  DEFINE CONSTANTS  -----------------

      PROMPTS_PATH       = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", h, "prompts", run_id)       + os.path.sep
      RESPONSES_PATH     = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", h, "responses", run_id)     + os.path.sep
      RAW_RESPONSES_PATH = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", h, "raw_responses", run_id) + os.path.sep

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
        "--run_id",
        type=str,
        required=True,
        help="Run identifier(01, 02, ... 0n)"
    )
    args = parser.parse_args()
    run_id = args.run_id
    main(run_id)