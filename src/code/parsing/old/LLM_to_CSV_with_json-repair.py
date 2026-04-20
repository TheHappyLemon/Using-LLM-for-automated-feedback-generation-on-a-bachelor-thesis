import csv
import os
import json
import json_repair
import re
import logging
from src.code.parsing.old.logging_config_new import setup_logging
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

ARRAY_REGEX = r'\[\s*(\{(?:[^{}]|\{[^{}]*\})*\}\s*,?\s*)+\]'
JSON_SCHEMA = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "feedback": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                        {"type": "null"}
                    ],
                    "default": None
                },
                "complies": {
                    "type": "boolean",
                    "default": False
                }
            },
            "required": ["question"],
            "additionalProperties": False
        }
    }

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

def get_json(json_str : str, part : str = "") -> tuple:
    # returns object, bool_was_json_good_initially, bool_did_json_repair_help, amount_of_skipped_questions
    try:
        result = json.loads(json_str)
        return result, True, False, 0
    # ANTI-PATTERN according to docs, but I want to catch cases, where initial JSON was valid for statistics
    except JSONDecodeError as e:
        try:
            skipped_questions = 0
            repaired_json, logs = json_repair.repair_json(json_str, return_objects=True, skip_json_loads=True, logging=True, schema=JSON_SCHEMA, schema_repair_mode='salvage')
            logger.info("Fixed JSON with json_repair.")
            logger.debug("Logs:")
            for log in logs:
                logger.debug(log)
                if log['text'] == "Dropped invalid array item while salvaging":
                    skipped_questions = skipped_questions + 1
                elif log['text'] == "Inserted default value for missing property" and "complies" in log['context']:
                    skipped_questions = skipped_questions + 1
            return repaired_json, False, True, skipped_questions
        except ValueError as ve:
            logger.error("Even json_repair could not fix this mess")
            if part != "BeforeGoal":
                skipped_questions = 5
            elif part == "Goal":
                skipped_questions = 4
            elif part == "Tasks":
                skipped_questions = 7
            elif part == "AfterTasks":
                skipped_questions = 3
            else:
                skipped_questions = 0
            return [], False, False, skipped_questions
            
# ministral3-14b-q8 forgets to escape newlines
def fix_json_newlines(bad_json: str) -> str:
    def replacer(match):
        s = match.group(0)
        return s.replace('\n', '\\n')

    return re.sub(r'"(\\.|[^"\\])*"', replacer, bad_json, flags=re.DOTALL)

def parse_llm_response(llm_response : str, part_type : str, has_goal : bool):
    # returns object of one of four types: Goal, Tasks, BeforeGoal, AfterTasks
    
    skipped_questions = 0
    (json_arr, was_json_good_initially, did_json_repair_help, skipped_questions) = get_json(llm_response, part_type)
    if json_arr == []:
        return was_json_good_initially, did_json_repair_help, skipped_questions, None
    
    answers : list[QuestionAnswer] = []
    model_validation_OK = True # redundant leftover

    for item in json_arr:
        try:
            validated = QuestionAnswer.model_validate(item)
            answers.append(validated)
        except ValidationError as e:
            #model_validation_OK = False
            skipped_questions = skipped_questions + 1
            logger.error(f"Invalid item skipped: {e.errors()} | item={item}")

    if part_type != "full":
        part = globals()[f"parse_{part_type}"](answers, has_goal=has_goal, model_validation_OK=model_validation_OK)
    else:
        part = parse_full(answers, model_validation_OK=model_validation_OK)
    return was_json_good_initially, did_json_repair_help, skipped_questions, part

def main(path_answer : str, path_source, dump_feedback : bool = False, postfix : str = "", skipped_rows : list = None) -> int:

    with open(os.path.join(path_answer, f"stats_{postfix}.csv"), 'w', encoding='utf-8', newline='') as stats_f:

        writer = csv.writer(stats_f)
        writer.writerow(['ID', 'part', 'model', 'valid_json_originally', 'json_repair_helped', 'skipped_questions'])
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
                    source_json = get_json(source_f.read())[0]
                    if source_json is None:
                        has_goal = False
                    else:
                        has_goal = (source_json.get("Goal", "") != "")

            with open(full_file, 'r', encoding='utf-8') as f:
                logger.info(f"Processing: '{file}'")
                try:
                    res = parse_llm_response(f.read(), part, has_goal)
                    writer.writerow([text_id, part, model, res[0], res[1], res[2]])
                    entries.append((text_id, model, part, res[3]))
                except Exception as e:
                    logger.error(f"UNKNOWN EXCEPTION: {str(e)}")
                    writer.writerow([text_id, part, model, False, False, 'all'])
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

# python -m src.code.parsing.old.LLM_to_CSV_with_json-repair "src/results/llm/single_prompt_testing_01/responses/gemma4-26b-q4/t0/02" "src/data/texts/divided" --feedback --logfile_postfix="_single-prompt_to-csv_json-repair" --run_id="json-repair"
# python -m src.code.parsing.old.LLM_to_CSV_with_json-repair "src/results/llm/initial_testing_01/responses" "src/data/texts/divided" --feedback --logfile_postfix="initial-testing_to-csv_json-repair" --run_id="json-repair"

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
        "--logfile_postfix",
        default="",
        help="Postfix added to logfile. Just to distinct them later"
    )
    parser.add_argument(
        "--skip_rows",
        type=int,
        nargs="+",   # one or more integers
        help="Texts ids to be skipped"
    )
    args = parser.parse_args()

    setup_logging(args.logfile_postfix)
    logger = logging.getLogger(__name__)
    logger.info("STARTED")

    main(
        path_answer=os.path.join(BASE_PATH, args.results_path),
        path_source=os.path.join(BASE_PATH, args.data_path),
        dump_feedback=args.feedback,
        postfix=args.run_id,
        skipped_rows=args.skip_rows
    )