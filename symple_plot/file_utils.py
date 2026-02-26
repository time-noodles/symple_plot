import os
import glob
import re

def del_file(rein):
    """
    指定されたパスのリスト (rein) に含まれるファイルをすべて削除します。
    """
    for op in rein:
        if os.path.exists(op):
            os.remove(op)
            print(f"Deleted: {op}")
        else:
            print(f"File not found: {op}")

def atoi(text):
    """自然順ソートのための補助関数"""
    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    ファイルを文字列ではなく数値として自然に降順・昇順に並べ替えるキー。
    使用例: sorted(list, key=natural_keys)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def straighten_path(folder):
    """
    指定したフォルダ内のファイル一覧を取得し、ファイル名に含まれる数字で
    自然順（1, 2, ..., 10, 11）にソートしてリストとして返します。
    """
    # フォルダ内の全ファイルを取得（OSに依存しない安全なパス結合）
    search_path = os.path.join(folder, "*")
    files = glob.glob(search_path)
    
    # 自然順にソートして返す
    sorted_files = sorted(files, key=natural_keys)
    # Windowsパス(\)をスラッシュ(/)に統一して使いやすくする
    return [f.replace('\\', '/') for f in sorted_files]