import pandas as pd
from src.data.constants import BASE_PATH
import os


FILE_PATH = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_negative_feedback_analysis", "01_w_rubrics", "template.csv") 

df = pd.read_csv(FILE_PATH)
grades = df.drop(columns=['ID'])

print(grades)
print()
print(grades.stack())
# Compute the average of all grades, ignoring NaN values
average_score = grades.stack().mean()

# python -m src.code.parsing.new.analyse_feedback_grading
print(f"Average score: {average_score:.4f}")