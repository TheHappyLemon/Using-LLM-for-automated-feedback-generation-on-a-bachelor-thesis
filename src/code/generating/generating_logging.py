# logging_config.py
import time
import logging

# just want to have separate log file for each run
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
timestr = time.strftime("%Y%m%d-%H%M%S")

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(f"generating_{timestr}.log"), # log to file
        logging.StreamHandler()                            # log to console
    ]
)