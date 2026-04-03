import requests
import json
import os
from src.data.constants import BASE_PATH, PROXY_KEY, PROXY_URL
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# ----------------- FUNCTION USED LATER ----------------- #
# return raw text from given file
def read_file(path : str) -> str:
  with open(path, 'r', encoding='utf-8') as f:
    return f.read().strip("\n").strip()

# call LLM provider API
def prompt(system, user, model='gemma3:1b', save_to : str = "", temperature : float = 0):
    response = requests.post(
        PROXY_URL,
        headers = {
            'Authorization': PROXY_KEY,
            'Content-Type': 'application/json; charset=utf-8',
            'User-Agent': 'RTU'
        },
        json = {
            'model': model,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user}
            ],
            'options': {
                'temperature': temperature,
                'num_predict': 8192
            }
        }
    )
    if response.status_code != 200:
      if save_to != "":
        with open(save_to, 'w', encoding='utf-8') as f:
          f.write(response.text)
      raise Exception(f"Status code: '{response.status_code}'. Message: '{response.text}'")

    if save_to != "":
      with open(save_to, 'w', encoding='utf-8') as f:
        f.write(response.text)
    return response.json()['message']['content']

def get_prompt(prompt_name: str):
  logger.info(f"Imported prompt: {prompt_name}")
  return read_file(path=os.path.join(BASE_PATH, "src", "data", "prompts", f"{prompt_name}.txt"))

def get_texts() -> dict:
    introduction_texts_divided = {}
    TEXTS_PATH = os.path.join(BASE_PATH, "src", "data", "texts", "divided")

    for filename in os.listdir(TEXTS_PATH):
        if not filename.endswith('.json'):
            continue
        introduction_texts_divided[int(filename.split('.')[0])] = json.load(open(TEXTS_PATH + os.path.sep + filename, 'r', encoding='utf-8'))

    logger.info("get_texts executed")
    return introduction_texts_divided

def get_texts_not_divided() -> dict:
    introduction_texts_divided = {}
    TEXTS_PATH = os.path.join(BASE_PATH, "src", "data", "texts", "clean")

    for filename in os.listdir(TEXTS_PATH):
        if not filename.endswith('.txt'):
            continue
        introduction_texts_divided[int(filename.split('.')[0])] = open(TEXTS_PATH + os.path.sep + filename, 'r', encoding='utf-8').read()

    logger.info("get_texts_not_divided executed")
    return introduction_texts_divided

def get_topics() -> dict:
    df = pd.read_csv(os.path.join(BASE_PATH, "src", "data", "texts", "topics.csv"))
    logger.info("get_topics executed")
    return dict(zip(df['ID'], df['Nosaukums']))

def prepare_prompts(introduction_texts_divided: dict, introduction_topics: dict,
                    REFINE_BEFORE_GOAL: str, REFINE_GOAL_WITH_PRECEDING : str, REFINE_GOAL_WITHOUT_PRECEDING: str,
                    REFINE_TASKS_WITH_GOAL: str, REFINE_TASKS_WITHOUT_GOAL: str, REFINE_AFTER_TASKS: str) -> dict:
    absence = ("", None)
    prompts_part_refinement = {}

    for text in introduction_texts_divided:

        before_goal = introduction_texts_divided[text].get("BeforeGoal")
        goal = introduction_texts_divided[text].get("Goal")
        tasks = introduction_texts_divided[text].get("Tasks")
        after_tasks = introduction_texts_divided[text].get("AfterTasks")
        topic = introduction_topics[text]

        prompts_part_refinement[text] = {
            "BeforeGoal" : "",
            "Goal" : "",
            "Tasks" : "",
            "AfterTasks" : ""
        }

        if before_goal not in absence:
            prompts_part_refinement[text]["BeforeGoal"] = REFINE_BEFORE_GOAL.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', before_goal)
        if goal not in absence:
            if before_goal not in absence:
                prompts_part_refinement[text]["Goal"] = REFINE_GOAL_WITH_PRECEDING.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', goal).replace("{PRECEDING_TEXT}", before_goal)
            else:
                prompts_part_refinement[text]["Goal"] = REFINE_GOAL_WITHOUT_PRECEDING.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', goal)
        if tasks not in absence:
            if goal not in absence:
                prompts_part_refinement[text]["Tasks"] = REFINE_TASKS_WITH_GOAL.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', tasks).replace("{THESIS_GOAL}", goal)
            else:
                prompts_part_refinement[text]["Tasks"] = REFINE_TASKS_WITHOUT_GOAL.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', tasks)
        if after_tasks not in absence:
            prompts_part_refinement[text]["AfterTasks"] = REFINE_AFTER_TASKS.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', after_tasks)
    
    logger.info("prepare_prompts executed")
    return prompts_part_refinement

def prepare_prompts_single(introduction_texts: dict, introduction_topics: dict,
                    REFINE_EVERYTHING_AT_ONCE: str):
    prompts_single = {}
    for text in introduction_texts:
       
       topic = introduction_topics[text]
       prompts_single[text] = REFINE_EVERYTHING_AT_ONCE.replace('{THESIS_TOPIC}', topic).replace('{TEXT}', introduction_texts[text])

    logger.info("prepare_prompts_single executed")
    return prompts_single

def save_used_prompts(PROMPTS_PATH: str, prompts_parts: dict):
    for p in prompts_parts:
        if prompts_parts[p]["BeforeGoal"] != "":
            with open(f'{PROMPTS_PATH}{p}_BeforeGoal.txt', 'w', encoding='utf-8') as f:
                f.write(prompts_parts[p]["BeforeGoal"])
        if prompts_parts[p]["Goal"] != "":
            with open(f'{PROMPTS_PATH}{p}_Goal.txt', 'w', encoding='utf-8') as f:
                f.write(prompts_parts[p]["Goal"])
        if prompts_parts[p]["Tasks"] != "":
            with open(f'{PROMPTS_PATH}{p}_Tasks.txt', 'w', encoding='utf-8') as f:
                f.write(prompts_parts[p]["Tasks"])
        if prompts_parts[p]["AfterTasks"] != "":
            with open(f'{PROMPTS_PATH}{p}_AfterTasks.txt', 'w', encoding='utf-8') as f:
                f.write(prompts_parts[p]["AfterTasks"])
    logger.info(f"used prompts saved to {PROMPTS_PATH}")

def save_used_prompts_not_divided(PROMPTS_PATH: str, prompt_not_divided: dict):
    for p in prompt_not_divided:
        with open(f'{PROMPTS_PATH}{p}_full.txt', 'w', encoding='utf-8') as f:
            f.write(prompt_not_divided[p])
    logger.info(f"used prompts saved to {PROMPTS_PATH}")

def make_prompt(text: str, model : str, model_role: str, temperature: float, raw_response_model_path: str, response_model_path: str, fname: str):

  if text == "":
    return

  try:
    os.makedirs(raw_response_model_path, exist_ok=True)
    os.makedirs(response_model_path    , exist_ok=True)

    response = prompt(
      system = f"You are {model_role}",
      user = text,
      model = model,
      save_to=f"{raw_response_model_path}{fname}",
      temperature=temperature
    )
    with open(f'{response_model_path}{fname}', 'w', encoding='utf-8') as f:
      f.write(response)
  except Exception as e:
    logger.error(e)
