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

#{
#    "1": {
#        "text": "la-la",
#        "feedback": [
#            {
#                "question": "feedback"
#            }
#        ]
#    }
#}

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

        texts[Nr]['aftertasks']['feedback'].append(
            {
                f"{question}" : feedback
            }
        )

os.makedirs(pats_answer, exist_ok=True)
with open(os.path.join(pats_answer, "full.json"), 'w', encoding='utf-8') as f_w:
    f_w.write(json.dumps(texts))

print(json.dumps(texts))