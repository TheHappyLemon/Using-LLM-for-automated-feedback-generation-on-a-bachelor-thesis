# logging_config.py
import time
import logging
import os

# just want to have separate log file for each run
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
timestr = time.strftime("%Y%m%d-%H%M%S")

log_dir = os.path.join("src", "log")
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"generating_{timestr}.log")

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(log_path), # log to file
        logging.StreamHandler()        # log to console
    ]
)