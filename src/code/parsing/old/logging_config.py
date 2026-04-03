# logging_config.py
import time
import logging
import logging.config

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# just want to have separate log file for each run
timestr = time.strftime("%Y%m%d-%H%M%S")

logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(f"app_{timestr}.log"), # log to file
        logging.StreamHandler()                    # log to console
    ]
)