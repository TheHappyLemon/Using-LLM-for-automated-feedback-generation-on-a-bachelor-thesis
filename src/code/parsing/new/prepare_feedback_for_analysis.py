from src.data.constants import BASE_PATH
import json
import os
import csv

path_feedback = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_feedback_for_analysis.csv")
path_source = os.path.join(BASE_PATH, "src", "data", "texts", "divided")
pats_answer = os.path.join(BASE_PATH, "src", "results", "llm", "initial_testing_01", "gemma4_negative_feedback_analysis")
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
mapping = {
    "BeforeGoal": [
        "Significance",
        "State_of_the_art",
        "Gap",
        "Problem",
        "References"
    ],
    "Goal": [
        "Purpose",
        "Intention",
        "Structure",
        "Congruence"
    ],
    "Tasks": [
        "Outlook",
        "Quantity",
        "Completeness",
        "Format",
        "Structure",
        "Clarity",
        "Relevance"
    ],
    "AfterTasks": [
        "Chapters",
        "Description",
        "Structure"
    ]
}

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
        has_beforegoal =  False
        has_goal = False
        has_tasks = False
        has_aftertasks =  False

        with open(os.path.join(path_source, f"{Nr}.json"), 'r', encoding='utf-8') as source:
            source_json = json.loads(source.read())
            has_beforegoal = source_json.get("BeforeGoal", "") != ""
            has_goal = source_json.get("Goal", "") != ""
            has_tasks = source_json.get("Tasks", "") != ""
            has_aftertasks = source_json.get("AfterTasks", "") != ""

        if answer != '1' and (answer != "5" and answer != "6" and answer != "7"):
            text = ""
            if part == "beforegoal" and has_beforegoal:
                text = source_json['BeforeGoal']
            elif part == "goal" and has_goal:
                text = source_json['Goal']
            elif part == "tasks" and has_tasks:
                text = source_json['Tasks']
            elif part == "aftertasks" and has_aftertasks:
                text = source_json['AfterTasks']
            if texts.get(Nr, "") == "":
                texts[Nr] = {}
            if texts[Nr].get(part, "") == "":
                texts[Nr][part] = {}
                texts[Nr][part]['lines'] = []
                texts[Nr][part]['text'] = ""
            #print(json.dumps(texts))
            texts[Nr][part]['lines'].append(f"question : {question} - {feedback}\n\n")
            texts[Nr][part]['text'] = text

os.makedirs(pats_answer, exist_ok=True)

for text in texts:
    for part in texts[text]:
        with open(os.path.join(pats_answer, f"feedback_analysis_{text}_{part}.txt"), 'w', encoding='utf-8') as f_w:
            f_w.write(f"# TEXT = {texts[text][part]['text']}\n\n")
            f_w.writelines(texts[text][part]['lines'])