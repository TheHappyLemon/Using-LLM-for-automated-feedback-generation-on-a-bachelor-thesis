# One time action after we added quantinizatio information to LLM short names. Let`s rename old generated files

import os

PATH_ANSWER = "C:\\Univer\\work\\grading-with-AI\\data\\dati_new\\results_07"

OLD_TO_NEW = {
    "gemma2-9b" : "gemma2-9b-q8",
    "gemma3-12b" : "gemma3-12b-qat",
    "gemma3-27b" : "gemma3-27b-qat",
    "llama3.1-8b" : "llama3.1-8b-q8",
    "mistral-nemo-12b" : "mistral-nemo-12b-q8",
    "mistral-small-22b" : "mistral-small-22b-q6",
    "mistral-small-24b" : "mistral-small-24b-q4",
    "qwen3-14b" : "qwen3-14b-q8",
    "qwen3-30b" : "qwen3-30b-q4",
    "deepseek-r1-14b" : "deepseek-r1-14b-q8",
    "eurollm-9b" : "eurollm-9b-q8"
}

for file in sorted(os.listdir(PATH_ANSWER)):
    # Take only JSON files
    full_file = os.path.join(PATH_ANSWER, file)
    if not os.path.isfile(full_file):
        continue
    params = file.split(".")
    if params[-1] != "json":
        continue

    params = file.split(".json")[0].split("_")
    text_id = params[0]
    part = params[1]
    model = params[2]
    new_file_name = ""

    if model in OLD_TO_NEW:
        new_file_name = f"{text_id}_{part}_{OLD_TO_NEW[model]}.json"
        new_file_name = os.path.join(PATH_ANSWER, new_file_name)
        print(f"Renaming {full_file} to {new_file_name}")
        os.rename(full_file, new_file_name)
