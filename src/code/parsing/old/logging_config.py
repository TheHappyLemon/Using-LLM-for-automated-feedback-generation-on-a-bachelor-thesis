# logging_config.py
import time
import logging
import logging.config
import os

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# just want to have separate log file for each run
timestr = time.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join("src", "log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"parsing_{timestr}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'), # log to file
        logging.StreamHandler()        # log to console
    ]
)