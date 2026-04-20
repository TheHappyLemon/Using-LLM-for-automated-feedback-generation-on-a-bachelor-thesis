#!/usr/bin/env python3
import os
import argparse
import logging
import src.code.parsing.old.logging_config as logging_config 
from src.data.constants import BASE_PATH

def find_most_characters(directory):
    max_file = None
    max_chars = -1

    for name in os.listdir(directory):
        if name.endswith(".txt"):
            path = os.path.join(directory, name)
            if os.path.isfile(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        count = len(f.read())
                    if count > max_chars:
                        max_chars = count
                        max_file = path
                except:
                    logging.warning(f"Skipping {path}")

    return max_file, max_chars


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find .txt file with most characters")
    parser.add_argument("directory", help="Path to folder")
    args = parser.parse_args()
    directory = os.path.join(BASE_PATH, args.directory)
    logging.info(f"Looking for longest file in {directory}")
    file, count = find_most_characters(directory)
    logging.info(f"It is {file} with {count} characters")