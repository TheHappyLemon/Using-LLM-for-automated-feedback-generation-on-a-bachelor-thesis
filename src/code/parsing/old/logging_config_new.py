import time
import logging
import os

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def setup_logging(postfix: str = ""):
    timestr = time.strftime("%Y%m%d-%H%M%S")

    log_dir = os.path.join("src", "log")
    os.makedirs(log_dir, exist_ok=True)

    suffix = f"_{postfix}" if postfix else ""
    log_path = os.path.join(log_dir, f"parsing_{timestr}{suffix}.log")

    # avoid duplicate handlers if called multiple times
    root = logging.getLogger()
    if root.handlers:
        return log_path

    logging.basicConfig(
        level=logging.DEBUG,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

    return log_path