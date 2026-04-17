import csv
import os
import json
import re
import logging
import src.code.parsing.old.logging_config as logging_config 
import argparse

from src.data.constants import BASE_PATH
from json.decoder import JSONDecodeError
from src.code.parsing.old.AfterTasks import AfterTasks
from src.code.parsing.old.BeforeGoal import BeforeGoal
from src.code.parsing.old.Tasks import Tasks
from src.code.parsing.old.Goal import Goal
from src.code.parsing.old.Full import Full
from src.code.parsing.old.model_answer import QuestionAnswer
from itertools import groupby
from operator import itemgetter
from src.code.parsing.old.EvaluationRow import EvaluationRow
from src.code.parsing.old.EvaluationDataset import EvaluationDataset
from pydantic import ValidationError
from pathlib import Path

logger = logging.getLogger(__name__)
logger.info("STARTED")

ARRAY_REGEX = r'\[\s*(\{(?:[^{}]|\{[^{}]*\})*\}\s*,?\s*)+\]'

def answer_keys_ok(json_obj : dict, required_keys : set):
    difference = set(json_obj.keys()).difference(required_keys)
    required_keys.copy()
    return len(difference) == 0, list(difference)

# model_validation_OK = if could not extract data, just create empty object.

def parse_Goal(json_arr : list[QuestionAnswer], model_validation_OK : bool, **kwargs) -> Goal:
    part = Goal()
    if not model_validation_OK:
        return part
    part.load(json_arr)
    return part 

def parse_Tasks(json_arr : list[QuestionAnswer], model_validation_OK : bool, has_goal : bool) -> Tasks:
    part = Tasks()
    if not model_validation_OK:
        return part
    part.load(json_arr, has_goal = has_goal)
    return part 

def parse_BeforeGoal(json_arr : list[QuestionAnswer], model_validation_OK : bool, **kwargs) -> BeforeGoal:
    part = BeforeGoal()
    if not model_validation_OK:
        return part
    part.load(json_arr)
    return part 

def parse_AfterTasks(json_arr : list[QuestionAnswer], model_validation_OK : bool, **kwargs) -> AfterTasks:
    part = AfterTasks()
    if not model_validation_OK:
        return part
    part.load(json_arr)
    return part  

def parse_full(json_arr : list[QuestionAnswer], model_validation_OK : bool, **kwargs) -> Full:
    logger.info("Parsing FULL entry")
    part = Full()
    if not model_validation_OK:
        return part
    part.load(json_arr)
    return part

def get_json(json_str : str) -> list:
    try:
        return json.loads(json_str)
    except JSONDecodeError as e:
        return []

# ministral3-14b-q8 forgets to escape newlines
def fix_json_newlines(bad_json: str) -> str:
    def replacer(match):
        s = match.group(0)
        return s.replace('\n', '\\n')

    return re.sub(r'"(\\.|[^"\\])*"', replacer, bad_json, flags=re.DOTALL)

def parse_llm_response(llm_response : str, part_type : str, has_goal : bool):
    # returns object of one of four types: Goal, Tasks, BeforeGoal, AfterTasks
    
    is_valid_json_originally = False
    removing_markdown_helped = False
    extracting_JSON_with_regex_helped = False

    json_arr = get_json(llm_response)
    if json_arr != []:
        is_valid_json_originally = True
        removing_markdown_helped = False
        extracting_JSON_with_regex_helped = False
    else:
        llm_response = llm_response.lstrip("```json").rstrip("```</end_of_turn>").rstrip("```")
        json_arr = get_json(llm_response)
        if json_arr != []:
            is_valid_json_originally = False
            removing_markdown_helped = True
            extracting_JSON_with_regex_helped = False
        else:
            match = re.search(ARRAY_REGEX, llm_response, re.DOTALL)
            if match:
                json_arr = get_json(match.group(0))
                if json_arr != []:
                    is_valid_json_originally = False
                    removing_markdown_helped = False
                    extracting_JSON_with_regex_helped = True
                else:
                    llm_response = fix_json_newlines(llm_response)
                    json_arr = get_json(llm_response)
                    if json_arr != []:
                        is_valid_json_originally = False
                        removing_markdown_helped = False
                        extracting_JSON_with_regex_helped = True
                    else:
                        logger.error("Failed to parse JSON (1)")
                        return False, False, False, False, None
            else:
                logger.error("Failed to parse JSON (2)")
                return False, False, False, False, None

    model_validation_OK = True
    answers : list[QuestionAnswer] = []

    try:
        answers = [QuestionAnswer.model_validate(item) for item in json_arr]
    except ValidationError as e:
        model_validation_OK = False
        logger.error(f"Pydantic model 'QuestionAnswer' failed validation: {e.errors()}")

    if part_type != "full":
        part = globals()[f"parse_{part_type}"](answers, has_goal=has_goal, model_validation_OK=model_validation_OK)
    else:
        part = parse_full(answers, model_validation_OK=model_validation_OK)
    return is_valid_json_originally, removing_markdown_helped, extracting_JSON_with_regex_helped, model_validation_OK, part

def main(path_answer : str, path_source, dump_feedback : bool = False, postfix : str = "", skipped_rows : list = None) -> int:

    with open(os.path.join(path_answer, f"stats_{postfix}.csv"), 'w', encoding='utf-8', newline='') as stats_f:

        writer = csv.writer(stats_f)
        writer.writerow(['ID', 'part', 'model', 'valid_json_originally', 'removing_markdown_helped', 'regex_helped', 'pydantic_model_validation_OK', 'error_occured'])
        entries = []

        for file in sorted(os.listdir(path_answer)):
            # Take only JSON files
            full_file = os.path.join(path_answer, file)
            if not os.path.isfile(full_file):
                continue
            params = file.split(".")
            if params[-1] != "json":
                continue
            
            params = file.split(".json")[0].split("_")
            text_id = params[0]
            part = params[1]
            model = params[2]
            has_goal = False

            if part.lower() == "tasks":
                with open(os.path.join(path_source, text_id + ".json"), 'r', encoding='utf-8') as source_f:
                    source_json = get_json(source_f.read())
                    if source_json is None:
                        has_goal = False
                    else:
                        has_goal = (source_json.get("Goal", "") != "") # type: ignore

            with open(full_file, 'r', encoding='utf-8') as f:
                logger.info(f"Processing: '{file}'")
                try:
                    res = parse_llm_response(f.read(), part, has_goal)
                    writer.writerow([text_id, part, model, res[0], res[1], res[2], res[3], False])
                    entries.append((text_id, model, part, res[4])) # type: ignore
                except Exception as e:
                    logger.error(f"UNKNOWN EXCEPTION: {str(e)}")
                    writer.writerow([text_id, part, model, False, False, False, False])
                    continue


    entries.sort(key=lambda x: (int(x[0]), x[1]))
    datasets: dict[str, EvaluationDataset] = {}

    logger.info("Start creatting datasets")

    for id_, id_group in groupby(entries, key=lambda x: int(x[0])):
        for model, model_group in groupby(list(id_group), key=itemgetter(1)):
            before_goal = None
            goal = None
            tasks = None
            after_tasks = None
            # individual case
            full = None

            for entry in model_group:
                if entry[2] == "BeforeGoal":
                    before_goal = entry[3]
                elif entry[2] == "Goal":
                    goal = entry[3]
                elif entry[2] == "Tasks":
                    tasks = entry[3]
                elif entry[2] == "AfterTasks":
                    after_tasks = entry[3]
                elif entry[2] == "full":
                    full = entry[3]

            row = EvaluationRow(model)
            if full == None:
                row.load_2(id_, before_goal, goal, tasks, after_tasks)
            else:
                logger.info("load 3 called")
                row.load_3(id_, full)
            row.to_int()
            if datasets.get(model, None) is None:
                datasets[model] = EvaluationDataset(model)
            datasets[model].append(row)

    for ds in datasets:
        datasets[ds].dump_to_csv(os.path.join(path_answer, f"{ds}_as_int{("_" + postfix) if postfix != "" else ""}.csv"))
    
    # Now prepare a special csv where for each model each answer corresponding feedback is shown
    if dump_feedback:
        HUMAN_RESPONSES_DIR = Path("src/results/human")
        human1_ds = EvaluationDataset("human1")
        human2_ds = EvaluationDataset("human2")
        human3_ds = EvaluationDataset("human3")
        human1_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human1_orig.csv")
        human2_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human2_orig.csv")
        human3_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human3_orig.csv")
        human_datasets = [human1_ds, human2_ds, human3_ds]
        EvaluationDataset.dump_to_csv_feedback(os.path.join(path_answer, f"feedback_{postfix}.csv"), human_datasets, list(datasets.values()), skipped_rows=skipped_rows)

    return 0

# python -m src.code.parsing.old.LLM_to_CSV "src/results/llm/initial_testing_01/responses" "src/data/texts/divided" --feedback --run_id="08"
# python -m src.code.parsing.old.LLM_to_CSV "src/results/llm/single_prompt_testing_01/responses/gemma4-26b-q4/t0/01" "src/data/texts/divided" --feedback
# python -m src.code.parsing.old.LLM_to_CSV "src/results/llm/one_shot_testing_01/responses" "src/data/texts/divided" --feedback --skip_rows 1 5 43
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", help="Path to folder with model responses")
    parser.add_argument("data_path"   , help="Path to original divided texts")
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="dump models feedback to CSV"
    )
    parser.add_argument(
        "--run_id",
        default="",
        help="Run identifier (e.g., 08)"
    )
    parser.add_argument(
        "--skip_rows",
        type=int,
        nargs="+",   # one or more integers
        help="Texts ids to be skipped"
    )
    args = parser.parse_args()

    main(
        path_answer=os.path.join(BASE_PATH, args.results_path),
        path_source=os.path.join(BASE_PATH, args.data_path),
        dump_feedback=args.feedback,
        postfix=args.run_id,
        skipped_rows=args.skip_rows
    )