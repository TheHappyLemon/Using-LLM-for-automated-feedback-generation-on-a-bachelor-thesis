from pathlib import Path
from src.code.parsing.old.EvaluationDataset import EvaluationDataset, EvaluationRow
from src.data.constants import BASE_PATH
import src.code.parsing.old.logging_config as logging_config
import logging
import os


logger = logging.getLogger(__name__)
logger.info("STARTED CALCULATING METRICS BASED ON TEMPERATURE TESTS")

BASE_RESPONSES_DIR = Path("src/results/llm/temperature_testing_01/responses/gpt-oss-20b-thinking")
HUMAN_RESPONSES_DIR = Path("src/results/human")
TEMPERATURE_FOLDERS = ["t0", "t0-5", "t1-0"]

def main() -> int:

    predicted_datasets = []

    for temp_folder in TEMPERATURE_FOLDERS:
        for i in range(10):
            iter = str(i + 1).zfill(2)
            input_dir = BASE_RESPONSES_DIR / temp_folder / iter
            if not input_dir.exists():
                logger.info(f"Skipping missing folder: {input_dir}")
                continue

            predicted_dataset = EvaluationDataset(f"gpt-oss-20b_{temp_folder}", iteration=i)
            predicted_dataset.load_from_csv(input_dir / "gpt-oss-20b-thinking_as_int.csv")
            predicted_dataset.to_bool()
            predicted_datasets.append(predicted_dataset)

        logger.info(f"Loaded {temp_folder} datasets")
    logger.info(f"Loaded all {len(temp_folder)} datasets")

    human1_ds = EvaluationDataset("human1")
    human2_ds = EvaluationDataset("human2")
    human3_ds = EvaluationDataset("human3")
    human1_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human1_orig.csv")
    human2_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human2_orig.csv")
    human3_ds.load_from_csv(HUMAN_RESPONSES_DIR / "human3_orig.csv")
    human1_ds.to_bool()
    human2_ds.to_bool(quantity_already_bool=True)
    human3_ds.to_bool()
    human_datasets = {human1_ds.author : human1_ds, human2_ds.author: human2_ds, human3_ds.author: human3_ds}
    logger.info("Loaded human datasets")
# compute_metrics_by_question_pooled
    for human in human_datasets:
        logger.info(f"Calculating metrics against {human}")
        EvaluationDataset.compute_metrics_total_average_by_iterations(
            human_datasets[human], predicted_datasets, os.path.join(BASE_PATH, "src", "results", "llm", "temperature_testing_01", f"results_vs_{human}_total.csv")
        )
        EvaluationDataset.compute_metrics_by_question_pooled(
            human_datasets[human], predicted_datasets, os.path.join(BASE_PATH, "src", "results", "llm", "temperature_testing_01", f"results_vs_{human}_by_question.csv")
        )

    return 0

if __name__ == "__main__":
    # python -m src.code.parsing.new.calculate_metrics_across_iterations
    raise SystemExit(main())