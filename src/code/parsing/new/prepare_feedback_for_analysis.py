from src.data.constants import BASE_PATH
import json
import os
import csv
import random

path_feedback = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_feedback_for_analysis.csv")
path_source = os.path.join(BASE_PATH, "src", "data", "texts", "divided")
pats_answer = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_negative_feedback_analysis")
csv_path = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_negative_feedback_analysis", "template.csv")

def main():

    os.makedirs(pats_answer, exist_ok=True)
    fieldnames = [
        "Nr",
        "Question",
        "part",
        "human1",
        "human2",
        "human3",
        "gemma4-26b-q4",
        "gemma4-26b-q4_feedback"
    ]
    texts = {}

    with open(path_feedback, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        next(reader)

        for row in reader:
            
            Nr = row['Nr']
            part = row['part'].lower()
            question = row['Question']
            answer = row['gemma4-26b-q4']
            feedback = row['gemma4-26b-q4_feedback']
            # Not efficient but I do not care
            with open(os.path.join(path_source, f"{Nr}.json"), 'r', encoding='utf-8') as source:
                source_json = json.loads(source.read())
                before_goal = source_json.get("BeforeGoal", "")
                goal = source_json.get("Goal", "")
                tasks = source_json.get("Tasks", "")
                aftertasks = source_json.get("AfterTasks", "")

            if not Nr in texts:
                texts[Nr] = {}
                texts[Nr]['beforegoal'] = {}
                texts[Nr]['goal'] = {}
                texts[Nr]['tasks'] = {}
                texts[Nr]['aftertasks'] = {}
                texts[Nr]['beforegoal']['text'] = ''
                texts[Nr]['goal']['text'] = ''
                texts[Nr]['tasks']['text'] = ''
                texts[Nr]['aftertasks']['text'] = ''
                texts[Nr]['beforegoal']['feedback'] = []
                texts[Nr]['goal']['feedback'] = []
                texts[Nr]['tasks']['feedback'] = []
                texts[Nr]['aftertasks']['feedback'] = []

            if part == "beforegoal":
                texts[Nr][part]['text'] = before_goal
            elif part == "goal":
                texts[Nr][part]['text'] = goal
            elif part == "tasks":
                texts[Nr][part]['text'] = tasks
            elif part == "aftertasks":
                texts[Nr][part]['text'] = aftertasks

            # Skip positive answers
            if answer == '1' or (question == "Quantity" and (answer == "5" or answer == "6" or answer == "7")):
                continue

            texts[Nr][part]['feedback'].append(
                {
                    f"{question}" : feedback
                }
            )
    # for debug
    with open(os.path.join(pats_answer, "full.json"), 'w', encoding='utf-8') as f_w:
        f_w.write(json.dumps(texts, indent=3, ensure_ascii=True))

    # now prepare this in propert format for people to read
    # we will randomly shuffle all texts and take all feedbacks for each text untill we collect ~90 feedbacks
    ids = list(texts.keys())
    random.shuffle(ids)
    selected = []
    
    amount = 0
    for id in ids:
        if amount >= 90:
            break
        amount = amount + prepare_part(id, texts[id])
        selected.append(int(id))

    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        header = ["ID"] + [str(i) for i in range(1, 19)]
        writer.writerow(header)
        selected.sort()
        for select in selected:
            writer.writerow([select])

def prepare_part(id, data : dict):
    counter = 0
    text = ""
    text = "--- BEFORE GOAL ---" + "\n" + data["beforegoal"]['text'] + "\n--- GOAL ---\n" + data["goal"]['text'] + "\n--- TASKS ---\n" + data["tasks"]['text'] + "\n--- AFTER TASKS ---\n" + data["aftertasks"]['text']
    BG_feedbacks_1 = data["beforegoal"]["feedback"]
    G_feedbacks_1 = data["goal"]["feedback"]
    T_feedbacks_1 = data["tasks"]["feedback"]
    AT_feedbacks_1 = data["aftertasks"]["feedback"]

    BG_feedbacks = []
    G_feedbacks = []
    T_feedbacks = []
    AT_feedbacks = []
    
    for feedback in BG_feedbacks_1:
        question = list(feedback.keys())[0]
        response = feedback[question]
        if response == "":
            continue
        BG_feedbacks.append(feedback.copy())
    for feedback in G_feedbacks_1:
        question = list(feedback.keys())[0]
        response = feedback[question]
        if response == "":
            continue
        G_feedbacks.append(feedback.copy())
    for feedback in T_feedbacks_1:
        question = list(feedback.keys())[0]
        response = feedback[question]
        if question == "Quantity":
            continue
        if response == "":
            continue
        T_feedbacks.append(feedback.copy())
    for feedback in AT_feedbacks_1:
        question = list(feedback.keys())[0]
        response = feedback[question]
        if response == "":
            continue
        AT_feedbacks.append(feedback.copy())

    # early exit if student did not write anything
    if text == "":
        return counter
    # early exit if text perfect = no negative answers = no feedback
    if len(BG_feedbacks) + len(G_feedbacks) + len(T_feedbacks) + len(AT_feedbacks) == 0:
        return counter
    with open(os.path.join(pats_answer, f"{id}.txt"), 'w', encoding='utf-8') as f_w:
        f_w.write("STUDENT TEXT:\n\n" + text + "\n\n--- FEDBACK ---\n\n")
        if len(BG_feedbacks) > 0 and data["beforegoal"]['text'] != "":
            f_w.write("Feedback for text preceding goal of the thesis\n")
            for feedback in BG_feedbacks:
                counter = counter + 1
                f_w.write(f"{counter}) " + feedback[list(feedback.keys())[0]] + "\n")
        if len(G_feedbacks) > 0 and data["goal"]['text'] != "":
            f_w.write("\nFeedback for goal of the thesis\n")
            for feedback in G_feedbacks:
                counter = counter + 1
                f_w.write(f"{counter}) " + feedback[list(feedback.keys())[0]] + "\n")
        if len(T_feedbacks) > 0 and data["tasks"]['text'] != "":
            f_w.write("\nFeedback for tasks of the thesis\n")
            for feedback in T_feedbacks:
                counter = counter + 1
                f_w.write(f"{counter}) " + feedback[list(feedback.keys())[0]] + "\n")
        if len(AT_feedbacks) > 0 and data["aftertasks"]['text'] != "":
            f_w.write("\nFeedback for text after tasks of the thesis\n")
            for feedback in AT_feedbacks:
                counter = counter + 1
                f_w.write(f"{counter}) " + feedback[list(feedback.keys())[0]] + "\n")

    return counter
# python -m src.code.parsing.new.prepare_feedback_for_analysis
if __name__ == "__main__":
    main()
