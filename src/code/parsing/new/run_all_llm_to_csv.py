from pathlib import Path
import logging

import src.code.parsing.old.logging_config as logging_config
from src.code.parsing.old import LLM_to_CSV

logger = logging.getLogger(__name__)
logger.info("STARTED TRANSFORMING EACH RESPONSE FROM TEMPERATURE TESTING TO CSV")

BASE_RESPONSES_DIR = Path("src/results/llm/temperature_testing_01/responses/gpt-oss-20b-thinking")
TEXTS_DIR = Path("src/data/texts/divided")
TEMPERATURE_FOLDERS = ["t0", "t0-5", "t1-0"]

def main() -> int:
    for temp_folder in TEMPERATURE_FOLDERS:
        for i in range(10):
            input_dir = BASE_RESPONSES_DIR / temp_folder / str(i + 1).zfill(2)

            if not input_dir.exists():
                logger.info(f"Skipping missing folder: {input_dir}")
                continue

            logger.info(f"Callin LLM_to_CSV for folder: {input_dir}")
            result = LLM_to_CSV.main(str(input_dir), str(TEXTS_DIR))

    logger.info("All folders processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())