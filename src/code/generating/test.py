from src.code.functions import prompt
from . import generating_logging
import logging

logging.info("TEST LOG MESSAGE DONT MIND ME")
r = prompt(system='You are pirate', user='hi', model='gpt-oss:20b', temperature=0)
print(r)