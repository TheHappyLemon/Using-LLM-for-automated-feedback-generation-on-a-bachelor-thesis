import os
import json

PATH = "C:\\Univer\\work\\grading-with-AI\\data\\dati_new\\original_responses_gpt4o"
PATH_NEW  = "C:\\Univer\\work\\grading-with-AI\\data\\dati_new\\results_05"
i = 0

for f in os.listdir(PATH):
    file_path = os.path.join(PATH, f)
    if not os.path.isfile(file_path):
        continue
    if not file_path.split(".")[-1] == "json":
        continue
    i = i + 1

    print(file_path)
    data = None
    with open(file_path, 'r', encoding='utf-8') as file_open:
        data = json.load(file_open)
    data = data['choices'][0]['message']['content']
    #data = json.loads(data)

    with open(os.path.join(PATH_NEW, f), 'w', encoding='utf-8') as file_open:
        file_open.write(data)