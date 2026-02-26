import os
import glob
import shutil
import re

def del_file(targets):
    """
    指定されたパスのファイルやディレクトリを削除します。
    ワイルドカード (*.png など) やディレクトリ名、またはそれらのリストを指定可能です。
    
    使用例: 
      del_file('out/fig/*.png')  # pngファイルのみ削除
      del_file('out/fig')        # ディレクトリごと削除
      del_file(['out/A.csv', 'out/B.csv']) # リストで複数指定
    """
    # 文字列が1つだけ渡された場合は、ループを回せるようにリストに変換する
    if isinstance(targets, str):
        targets = [targets]
        
    for target in targets:
        # globを使ってワイルドカードを展開
        matched_paths = glob.glob(target)
        
        for path in matched_paths:
            path = path.replace('\\', '/') # Windowsパスの表記揺れを防止
            if os.path.isfile(path):
                os.remove(path)
                print(f"Deleted file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path) # フォルダの中身ごとまるっと削除
                print(f"Deleted directory: {path}")

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
    search_path = os.path.join(folder, "*")
    files = glob.glob(search_path)
    sorted_files = sorted(files, key=natural_keys)
    return [f.replace('\\', '/') for f in sorted_files]