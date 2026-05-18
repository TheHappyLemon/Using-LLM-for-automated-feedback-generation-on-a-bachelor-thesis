import pandas as pd
from src.data.constants import BASE_PATH
import os
from statistics import mean

STUDENTS_PATH = os.path.join(BASE_PATH, "src", "results", "human", "students")

def get_average(author : str):
   
    df = pd.read_csv(os.path.join(STUDENTS_PATH, f"feedback_eval_{author}.csv"))
    grades = df.drop(columns=['ID'])
    #print(grades)
    #print()
    #print(grades.stack())
    average_score = grades.stack().mean()

    print(f"{author}: {average_score:.4f}")
    return average_score

# python -m src.code.parsing.new.analyse_feedback_grading
grade1 = get_average("student1")
grade2 = get_average("student2")
grade3 = get_average("student3")
grade4 = get_average("student4")

print(f"Total average = {mean([grade1, grade2, grade3, grade4])}")