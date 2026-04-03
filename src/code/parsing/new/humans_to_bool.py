from src.data.constants import BASE_PATH
from src.code.parsing.old.EvaluationDataset import EvaluationDataset
import os

# So that responses from all 3 humans are in same format. Becuase human1 and human3 for question
# 'quantity' in 'tasks' provided instead of 0 (false) or 1 (true) number of tasks.

HUMAN_PATH = os.path.join(BASE_PATH, "src", "results", "human") + os.path.sep

human1_data = EvaluationDataset(author="human1")
human2_data = EvaluationDataset(author="human2")
human3_data = EvaluationDataset(author="human3")
human1_data.load_from_csv(HUMAN_PATH + "human1_orig.csv")
human2_data.load_from_csv(HUMAN_PATH + "human2_orig.csv")
human3_data.load_from_csv(HUMAN_PATH + "human3_orig.csv")
human1_data.to_bool(quantity_already_bool=False)
human2_data.to_bool(quantity_already_bool=True)
human3_data.to_bool(quantity_already_bool=False)
human1_data.dump_to_csv(HUMAN_PATH + "human1.csv")
human2_data.dump_to_csv(HUMAN_PATH + "human2.csv")
human3_data.dump_to_csv(HUMAN_PATH + "human3.csv")