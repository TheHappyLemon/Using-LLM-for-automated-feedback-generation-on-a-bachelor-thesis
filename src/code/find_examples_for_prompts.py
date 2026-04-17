from src.code.parsing.old.EvaluationDataset import EvaluationDataset
from src.data.constants import BASE_PATH
from pathlib import Path
import os
import csv
import json

def at_least_two(*args):
    return sum(args) >= 2

def main():

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

    with open(os.path.join(BASE_PATH, "other", "complies_between_humans.csv"), 'w', encoding='utf-8', newline='') as res_f:
        writer = csv.writer(res_f)
        before_goal_ids_all = ['BeforeGoal']
        goal_ids_all        = ['Goal      ']
        tasks_ids_all       = ['Tasks     ']
        after_tasks_ids_all = ['AfterTasks']
        
        before_goal_ids_2 = ['BeforeGoal']
        goal_ids_2        = ['Goal      ']
        tasks_ids_2       = ['Tasks     ']
        after_tasks_ids_2 = ['AfterTasks']

        before_goal_ids_all_no = ['BeforeGoal']
        goal_ids_all_no        = ['Goal      ']
        tasks_ids_all_no       = ['Tasks     ']
        after_tasks_ids_all_no = ['AfterTasks']

        before_goal_ids_2_no = ['BeforeGoal']
        goal_ids_2_no        = ['Goal      ']
        tasks_ids_2_no       = ['Tasks     ']
        after_tasks_ids_2_no = ['AfterTasks']

        for row_human1 in human1_ds.rows:
            
            nr = row_human1.Nr
            row_human2 = next((r for r in human2_ds.rows if r.Nr == nr), None)
            row_human3 = next((r for r in human3_ds.rows if r.Nr == nr), None)

            # find positive cases
            if row_human1.beforeGoal.fully_complies() and row_human2.beforeGoal.fully_complies() and row_human3.beforeGoal.fully_complies(): before_goal_ids_all.append(nr)
            if row_human1.goal.fully_complies() and row_human2.goal.fully_complies() and row_human3.goal.fully_complies(): goal_ids_all.append(nr)
            if row_human1.tasks.fully_complies() and row_human2.tasks.fully_complies() and row_human3.tasks.fully_complies(): tasks_ids_all.append(nr)
            if row_human1.afterTasks.fully_complies() and row_human2.afterTasks.fully_complies() and row_human3.afterTasks.fully_complies():  after_tasks_ids_all.append(nr)

            if at_least_two(row_human1.beforeGoal.fully_complies(), row_human2.beforeGoal.fully_complies(), row_human3.beforeGoal.fully_complies()): before_goal_ids_2.append(nr)
            if at_least_two(row_human1.goal.fully_complies(), row_human2.goal.fully_complies(), row_human3.goal.fully_complies()): goal_ids_2.append(nr)
            if at_least_two(row_human1.tasks.fully_complies(), row_human2.tasks.fully_complies(), row_human3.tasks.fully_complies()): tasks_ids_2.append(nr)
            if at_least_two(row_human1.afterTasks.fully_complies(), row_human2.afterTasks.fully_complies(), row_human3.afterTasks.fully_complies()): after_tasks_ids_2.append(nr)

            # find negative cases

            with open(os.path.join(BASE_PATH, "src", "data", "texts", "divided", str(nr) + ".json"), 'r', encoding='utf-8') as source_f:
                source_json = json.loads(source_f.read())



            if source_json.get("BeforeGoal") != "":
                if row_human1.beforeGoal.fully_not_complies() and row_human2.beforeGoal.fully_not_complies() and row_human3.beforeGoal.fully_not_complies(): before_goal_ids_all_no.append(nr)
                if at_least_two(row_human1.beforeGoal.fully_not_complies(), row_human2.beforeGoal.fully_not_complies(), row_human3.beforeGoal.fully_not_complies()): before_goal_ids_2_no.append(nr)
            if source_json.get("Goal") != "":
                if row_human1.goal.fully_not_complies() and row_human2.goal.fully_not_complies() and row_human3.goal.fully_not_complies(): goal_ids_all_no.append(nr)
                if at_least_two(row_human1.goal.fully_not_complies(), row_human2.goal.fully_not_complies(), row_human3.goal.fully_not_complies()): goal_ids_2_no.append(nr)
            if source_json.get("Tasks") != "":
                if row_human1.tasks.fully_not_complies() and row_human2.tasks.fully_not_complies() and row_human3.tasks.fully_not_complies(): tasks_ids_all_no.append(nr)
                if at_least_two(row_human1.tasks.fully_not_complies(), row_human2.tasks.fully_not_complies(), row_human3.tasks.fully_not_complies()): tasks_ids_2_no.append(nr)
            if source_json.get("AfterTasks") != "":
                if row_human1.afterTasks.fully_not_complies() and row_human2.afterTasks.fully_not_complies() and row_human3.afterTasks.fully_not_complies(): after_tasks_ids_all_no.append(nr)
                if at_least_two(row_human1.afterTasks.fully_not_complies(), row_human2.afterTasks.fully_not_complies(), row_human3.afterTasks.fully_not_complies()): after_tasks_ids_2_no.append(nr)

        writer.writerow(['complies_between_all_humans'])
        writer.writerow(before_goal_ids_all)
        writer.writerow(goal_ids_all)
        writer.writerow(tasks_ids_all)
        writer.writerow(after_tasks_ids_all)
        writer.writerow([])
        writer.writerow(['complies_between_2_humans'])
        writer.writerow(before_goal_ids_2)
        writer.writerow(goal_ids_2)
        writer.writerow(tasks_ids_2)
        writer.writerow(after_tasks_ids_2)
        writer.writerow([])
        writer.writerow(['not_complies_between_all_humans'])
        writer.writerow(before_goal_ids_all_no)
        writer.writerow(goal_ids_all_no)
        writer.writerow(tasks_ids_all_no)
        writer.writerow(after_tasks_ids_all_no)
        writer.writerow([])
        writer.writerow(['not_complies_between_2_humans'])
        writer.writerow(before_goal_ids_2_no)
        writer.writerow(goal_ids_2_no)
        writer.writerow(tasks_ids_2_no)
        writer.writerow(after_tasks_ids_2_no)

# python -m src.code.find_examples_for_prompts
if __name__ == "__main__":
    main()