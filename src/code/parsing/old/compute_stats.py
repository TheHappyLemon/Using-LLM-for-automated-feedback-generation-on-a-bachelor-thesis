from src.code.parsing.old.EvaluationDataset import EvaluationDataset
from src.data.constants import BASE_PATH
import csv
import os
import pandas as pd
from pathlib import Path

PATH_RESULTS_NEW = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01")
PATH_RESULTS_OLD = os.path.join(BASE_PATH, "results_04")
postfix_new = "08"
postfix_old = "04"
evaluators = [
	"gemma2-9b-q8",
	"gemma3-12b-qat",
	"gemma3-12b-q8",
	"gemma3-27b-qat",
	"llama3.1-8b-q8",
	"llama3.1-8b-fp16",
	"mistral-nemo-12b-q8",
	"mistral-small-22b-q6",
	"mistral-small-24b-q4",
	"qwen3-14b-q8-thinking",
	"qwen3-30b-q4-thinking",
	"deepseek-r1-14b-q8",
	"gpt-oss-20b-thinking",
	"eurollm-9b-q8",
    "magistral-24b-q4"
]

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
datasets : list[EvaluationDataset] = []

def calculate_new_results():
    for evaluator in evaluators:
        dataset = EvaluationDataset(author=evaluator)
        dataset.load_from_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_int_{postfix_new}.csv"))
        dataset.to_bool()
        dataset.dump_to_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_bool_{postfix_new}.csv"))
        datasets.append(dataset)
        EvaluationDataset.compute_metrics(baseline_ds=human1_ds, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human1_ds.author}_results_{postfix_new}.csv"))
        EvaluationDataset.compute_metrics(baseline_ds=human2_ds, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human2_ds.author}_results_{postfix_new}.csv"))
        EvaluationDataset.compute_metrics(baseline_ds=human3_ds, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human3_ds.author}_results_{postfix_new}.csv"))
    EvaluationDataset.compute_metrics_total_average(human1_ds, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human1_ds.author}_total_{postfix_new}.csv"))
    EvaluationDataset.compute_metrics_total_average(human2_ds, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human2_ds.author}_total_{postfix_new}.csv"))
    EvaluationDataset.compute_metrics_total_average(human3_ds, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human3_ds.author}_total_{postfix_new}.csv"))

def compare_results(new_evaluators : list[str], old_evaluators : list[str]):

    for new_evaluator in new_evaluators:
        for old_evaluator in old_evaluators:
            old = pd.read_csv(os.path.join(PATH_RESULTS_OLD, f"{old_evaluator}_results_{postfix_old}.csv"))
            new = pd.read_csv(os.path.join(PATH_RESULTS_NEW, f"{new_evaluator}_results_{postfix_new}.csv"))

            # Merge on "Question" column (common identifier)
            merged = pd.merge(old, new, on="Question", suffixes=("_old", "_new"))
            diff = pd.DataFrame()
            diff["Question"] = merged["Question"]
            for col in old.columns:
                if col == "Question":
                    continue
                diff[col + "_diff"] = merged[col + "_new"] - merged[col + "_old"]

            diff.to_csv(os.path.join(PATH_RESULTS_NEW, f"DIFF_{new_evaluator}_VS_{old_evaluator}_v{postfix_new}_VS_v{postfix_old}.csv"), index=False)

def compare_humans():
    h_datasets = [human1_ds, human2_ds, human3_ds]

    EvaluationDataset.compute_metrics_total_average(human1_ds, h_datasets, path=os.path.join(HUMAN_RESPONSES_DIR, f"human1_VS_humans_results_total.csv"))
    EvaluationDataset.compute_metrics_total_average(human2_ds, h_datasets, path=os.path.join(HUMAN_RESPONSES_DIR, f"human2_VS_humans_results_total.csv"))
    EvaluationDataset.compute_metrics_total_average(human3_ds, h_datasets, path=os.path.join(HUMAN_RESPONSES_DIR, f"human3_VS_humans_results_total.csv"))

if __name__ == "__main__":
    #calculate_new_results()
    compare_humans()
    #compare_results(evaluators, ["gemma3-27b-it", "mistral-small-24b-it"])


