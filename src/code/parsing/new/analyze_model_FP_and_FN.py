from src.code.parsing.old.EvaluationDataset import EvaluationDataset
from src.data.constants import BASE_PATH
import os
import operator
import pandas as pd
import csv

def main():
    human1_ds = EvaluationDataset("human1")
    human2_ds = EvaluationDataset("human2")
    human3_ds = EvaluationDataset("human3")

    human1_ds.load_from_csv(os.path.join(BASE_PATH, "src", "results", "human", "human1_orig.csv"))
    human2_ds.load_from_csv(os.path.join(BASE_PATH, "src", "results", "human", "human2_orig.csv"))
    human3_ds.load_from_csv(os.path.join(BASE_PATH, "src", "results", "human", "human3_orig.csv"))

    human1_ds.to_bool()
    human2_ds.to_bool(quantity_already_bool=True)
    human3_ds.to_bool()

    model_ds = EvaluationDataset("gemma4")
    model_ds.load_from_csv(
        os.path.join(
            BASE_PATH,
            "src", "results", "llm", "actionable_feedback_01",
            "responses", "gemma4-26b-q4",
            "gemma4-26b-q4_as_int_json-repair.csv"
        )
    )
    model_ds.to_bool()

    # Initialize stats for all questions (ensures zero-count inclusion)
    stats = {
        f"{attr_path}_{label}": {"question" : label, "Part": attr_path, "FP": 0, "FN": 0}
        for label, attr_path in EvaluationDataset.questions
    }
    stats_detailed = []

    for model_row in model_ds.rows:
        human1_row = next((row for row in human1_ds.rows if row.Nr == model_row.Nr), None)
        human2_row = next((row for row in human2_ds.rows if row.Nr == model_row.Nr), None)
        human3_row = next((row for row in human3_ds.rows if row.Nr == model_row.Nr), None)

        if human1_row is None or human2_row is None or human3_row is None:
            raise ValueError(f"Could not find human Nr '{model_row.Nr}'")

        for label, attr_path in EvaluationDataset.questions:
            getter = operator.attrgetter(f"{attr_path}.{label}")

            model_answer  = getter(model_row).value
            human1_answer = getter(human1_row).value
            human2_answer = getter(human2_row).value
            human3_answer = getter(human3_row).value
            model_feedback = getter(model_row).feedback

            # Only cases where all humans agree
            if not ((human1_answer == human2_answer) and (human2_answer == human3_answer)):
                continue
            # Only disagreements
            if model_answer == human1_answer:
                continue

            stats_detailed_row = {
                "Nr"   : model_row.Nr,
                "part" : attr_path,
                "question" : label,
                "H1" : human1_answer,
                "H2" : human2_answer,
                "H3" : human3_answer,
                "LLM": model_answer,
                "class": '',
                "feedback": model_feedback
            }

            # Classify
            if model_answer and not human1_answer:
                stats[f"{attr_path}_{label}"]["FP"] += 1
                stats_detailed_row["class"] = "FP"
            else:
                stats[f"{attr_path}_{label}"]["FN"] += 1
                stats_detailed_row["class"] = "FN"
            stats_detailed.append(stats_detailed_row)

    output_path = os.path.join(
            BASE_PATH,
            "src", "results", "llm", "actionable_feedback_01",
            "gemma4_FP_FN_analysis"
        )
    df = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    df.to_csv(os.path.join(output_path, "FP_FN_stats.csv"), index=False, encoding="utf-8-sig", columns=['question','Part','FP','FN'])

    # Stupid. I should have just started going from this feedback csv. But I already went from _as_int version.
    # So here just extract feedback
    path_feedback = os.path.join(BASE_PATH, "src", "results", "llm", "actionable_feedback_01", "responses", "gemma4-26b-q4", "feedback_json-repair.csv")
    feedback_rows = []
    with open(path_feedback, 'r', encoding='utf-8-sig') as f:
        dict_reader = csv.DictReader(f)
        feedback_rows = list(dict_reader)

    for elem in stats_detailed:
        Nr = elem['Nr']
        question = elem['question']
        part = elem['part']
        for f_row in feedback_rows:
            #print(f_row)
            if f_row['Nr'] == str(Nr) and f_row['part'] == part and f_row['Question'] == question:
                elem['feedback'] = f_row['gemma4-26b-q4_feedback']

    with open(os.path.join(output_path, "FP_FN_stats_detailed.csv"), 'w', encoding='utf-8-sig', newline='') as out_f:
        csv_writer = csv.DictWriter(out_f, fieldnames=['Nr', "part", "question", 'H1', 'H2', 'H3', 'LLM', 'class', 'feedback'])
        csv_writer.writeheader()
        csv_writer.writerows(stats_detailed)


if __name__ == "__main__":
    main()