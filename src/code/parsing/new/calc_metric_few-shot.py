from src.code.parsing.old.EvaluationDataset import EvaluationDataset
from src.data.constants import BASE_PATH
import os
import pandas as pd
from pathlib import Path
import logging
import src.code.parsing.old.logging_config as logging_config 

logger = logging.getLogger(__name__)
logger.info("STARTED COMPUTING STATS FOR FEW-SHOT")

SHOTS = ["2-shot", "3-shot", "4-shot"]
HUMANS = ["human1", "human2", "human3"]
evaluators = [
    "gemma4-26b-q4"
]
postfix_new = "better-parsing"

def main():

    for SHOT in SHOTS:

        logger.info(f"Working on shot {SHOT}")

        skipped_rows = [1, 5, 7, 43]
        # test set should always be the same
        '''
        if SHOT == "2-shot":
            skipped_rows = [1, 5]
        elif SHOT == "3-shot":
            skipped_rows = [1, 5, 7]
        elif SHOT == "4-shot":
            skipped_rows = [1, 5, 7, 43]
        '''
        logger.info(f"Skipped rows will be: {skipped_rows}")

        HUMAN_RESPONSES_DIR = Path("src/results/human")
        human1_ds = EvaluationDataset("human1")
        human2_ds = EvaluationDataset("human2")
        human3_ds = EvaluationDataset("human3")
        human1_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human1_orig.csv", skipped_rows=skipped_rows)
        human2_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human2_orig.csv", skipped_rows=skipped_rows)
        human3_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human3_orig.csv", skipped_rows=skipped_rows)
        human1_ds.to_bool()
        human2_ds.to_bool(quantity_already_bool=True)
        human3_ds.to_bool()
        
        # Also calculate metric for INITIAL RESULTS with test texts removed.
        PATH_RESULTS_INITIAL = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "responses")
        datasets : list[EvaluationDataset] = []
        for evaluator in evaluators:
            dataset = EvaluationDataset(author=evaluator)
            dataset.load_from_csv(os.path.join(PATH_RESULTS_INITIAL, f"{evaluator}_as_int_08.csv"), skipped_rows=skipped_rows)
            dataset.to_bool()
            datasets.append(dataset)
        EvaluationDataset.compute_metrics_total_average(human1_ds, datasets, path=os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", f"OG_VS_{human1_ds.author}_total_{postfix_new}.csv"))
        EvaluationDataset.compute_metrics_total_average(human2_ds, datasets, path=os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", f"OG_VS_{human2_ds.author}_total_{postfix_new}.csv"))
        EvaluationDataset.compute_metrics_total_average(human3_ds, datasets, path=os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", f"OG_VS_{human3_ds.author}_total_{postfix_new}.csv"))

        for HUMAN in HUMANS:
            PATH_RESULTS_NEW = os.path.join(BASE_PATH, "src", "results", "llm", f"{SHOT}_testing_01", HUMAN, "responses", "01")
            logger.info(f"Calculating results in: {PATH_RESULTS_NEW}")
            datasets : list[EvaluationDataset] = []
            for evaluator in evaluators:
                dataset = EvaluationDataset(author=evaluator)
                if postfix_new != "":
                    dataset.load_from_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_int_{postfix_new}.csv"), skipped_rows=skipped_rows)
                else:
                    dataset.load_from_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_int.csv"), skipped_rows=skipped_rows)
                dataset.to_bool()
                dataset.dump_to_csv(os.path.join(PATH_RESULTS_NEW, f"{evaluator}_as_bool_{postfix_new}.csv"))
                datasets.append(dataset)
                if HUMAN == "human1":
                    EvaluationDataset.compute_metrics(baseline_ds=human1_ds, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human1_ds.author}_results_{postfix_new}.csv"))
                elif HUMAN == "human2":
                    EvaluationDataset.compute_metrics(baseline_ds=human2_ds, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human2_ds.author}_results_{postfix_new}.csv"))
                elif HUMAN == "human3":
                    EvaluationDataset.compute_metrics(baseline_ds=human3_ds, predicted_ds=dataset, path=os.path.join(PATH_RESULTS_NEW, f"{evaluator}_VS_{human3_ds.author}_results_{postfix_new}.csv"))
            if HUMAN == "human1":
                EvaluationDataset.compute_metrics_total_average(human1_ds, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human1_ds.author}_total_{postfix_new}.csv"))
            elif HUMAN == "human2":
                EvaluationDataset.compute_metrics_total_average(human2_ds, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human2_ds.author}_total_{postfix_new}.csv"))
            elif HUMAN == "human3":
                EvaluationDataset.compute_metrics_total_average(human3_ds, datasets, path=os.path.join(PATH_RESULTS_NEW, f"results_VS_{human3_ds.author}_total_{postfix_new}.csv"))

# python -m src.code.parsing.new.calc_metric_few-shot
if __name__ == "__main__":
    main()


