from src.data.constants import PROXY_KEY, PROXY_URL
from requests import post

def prompt(system, user, model='gemma3:1b', save_to : str = "", temperature : float = 0):
    response = post(
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
    print('raw:', response.json())

    return response.json()['message']['content']

r = prompt(system='You are pirate', user='hi', model='gpt-oss:20b', temperature=0)
print(r)