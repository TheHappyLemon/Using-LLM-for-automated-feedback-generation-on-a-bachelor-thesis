from EvaluationDataset import EvaluationDataset
import csv
import os
import pandas as pd

PATH_BASE = "C:\\Univer\\work\\grading-with-AI\\data\\dati_new"
PATH_RESULTS_NEW = os.path.join(PATH_BASE, "results_08")
PATH_RESULTS_OLD = os.path.join(PATH_BASE, "results_04")
postfix_new = "08"
postfix_old = "04"
evaluators = [
	"gemma2-9b-q8",
	"gemma3-1b",
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

human_data = EvaluationDataset(author="human2")
human_data.load_from_csv(os.path.join(PATH_BASE, f"{human_data.author}_as_int.csv"))
human_data.to_bool(quantity_already_bool= (True if human_data.author == "human2" else False))
human_data.dump_to_csv(os.path.join(PATH_BASE, f"{human_data.author}_as_bool.csv"))
datasets : list[EvaluationDataset] = []

def calculate_new_results():
    for evaluator in evaluators:
        dataset = EvaluationDataset(author=evaluator)
        dataset.load_from_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_int_{postfix_new}.csv"))
        dataset.to_bool()
        dataset.dump_to_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_bool_{postfix_new}.csv"))
        datasets.append(dataset)
        EvaluationDataset.compute_metrics(baseline_ds=human_data, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human_data.author}_results_{postfix_new}.csv"))
    EvaluationDataset.compute_metrics_total_average(human_data, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human_data.author}_total_{postfix_new}.csv"))

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
    human1_data = EvaluationDataset(author="human1")
    human1_data.load_from_csv(os.path.join(PATH_BASE, f"human1_as_int.csv"))
    human1_data.to_bool()
    human1_data.dump_to_csv(os.path.join(PATH_BASE, f"{human1_data.author}_as_bool.csv"))

    human2_data = EvaluationDataset(author="human2")
    human2_data.load_from_csv(os.path.join(PATH_BASE, "human2_as_int.csv"))
    human2_data.to_bool(quantity_already_bool=True)
    human2_data.dump_to_csv(os.path.join(PATH_BASE, f"{human2_data.author}_as_bool.csv"))

    h_datasets = [human2_data]

    EvaluationDataset.compute_metrics(baseline_ds=human1_data, predicted_ds=human2_data, path=os.path.join(PATH_BASE, f"human1_VS_human2_results.csv"))
    EvaluationDataset.compute_metrics_total_average(human1_data, h_datasets, path=os.path.join(PATH_BASE, f"human1_VS_human2_results_total.csv"))

compare_humans()
#calculate_new_results()
#compare_results(evaluators, ["gemma3-27b-it", "mistral-small-24b-it"])


