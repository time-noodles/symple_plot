# symple_plot/file_utils.py
from typing import Union, List
import os
import glob
import shutil
import re
import numpy as np
import pandas as pd


def del_file(targets: Union[str, List[str]]) -> None:
    """指定されたパターンに一致するファイルまたはディレクトリを一括で削除します。

    Args:
        targets (Union[str, List[str]]): 削除対象のファイルパス、またはワイルドカード(`*`)を含むパターンの文字列・リスト。
            例: `'*.csv'`, `['./images/*.png', './temp/']`
    """
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

def straighten_path(folder: str) -> List[str]:
    """指定されたフォルダ内のファイルパスを「自然順（Natural Sort）」でソートして取得します。
    （例: '1.txt', '10.txt', '2.txt' ではなく '1.txt', '2.txt', '10.txt' の順になります）

    Args:
        folder (str): 検索対象のフォルダパス。

    Returns:
        List[str]: 自然順でソートされたファイルパスのリスト（スラッシュ区切りに正規化済み）。
    """
    search_path = os.path.join(folder, "*")
    files = glob.glob(search_path)
    return [f.replace('\\', '/') for f in sorted(files, key=natural_keys)]

