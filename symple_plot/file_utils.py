# symple_plot/file_utils.py
import os
import glob
import shutil
import re
import numpy as np
import pandas as pd


def del_file(targets):
    if isinstance(targets, str): targets = [targets]
    for target in targets:
        matched_paths = glob.glob(target)
        for path in matched_paths:
            path = path.replace('\\', '/')
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleted file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                print(f"Deleted directory: {path}")

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def straighten_path(folder):
    search_path = os.path.join(folder, "*")
    files = glob.glob(search_path)
    return [f.replace('\\', '/') for f in sorted(files, key=natural_keys)]

