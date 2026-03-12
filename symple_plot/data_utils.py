# symple_plot/data_utils.py
from typing import List, Tuple, Union, Any, Optional
import numpy as np
import pandas as pd


def ensure_2d(data: Any) -> List[Any]:
    """データが2次元のリスト（または配列）構造になるように保証します。

    Args:
        data (Any): 1次元または2次元のデータ（リスト、NumPy配列、Pandas Seriesなど）。

    Returns:
        List[Any]: 2次元構造に変換されたリストのリスト。
    """
    if len(data) == 0: return [[]]
    if not isinstance(data[0], (list, tuple, np.ndarray, pd.Series)): return [data]
    return data

def pad_list(L: List[Any]) -> List[np.ndarray]:
    """長さが異なる複数のリスト（ジャグ配列）を受け取り、最大の長さに合わせてNaNでパディングします。

    Args:
        L (List[Any]): 長さが不揃いなリストまたは配列のリスト。

    Returns:
        List[np.ndarray]: 全て同じ長さに揃えられたfloat型のNumPy配列のリスト。
    """
    max_len = max([len(i) for i in L]) if len(L) > 0 else 0
    L_padded = [list(i) + [np.nan] * (max_len - len(i)) for i in L]
    res = []
    for i in L_padded:
        try:
            res.append(np.array(i, dtype=float))
        except (ValueError, TypeError):
            res.append(np.asarray(pd.to_numeric(i, errors='coerce'), dtype=float))
    return res

def minmax(val: List[Any], margin: float = 0.05, is_log: bool = False) -> Tuple[float, float]:
    """データのリストから、描画に最適な最小値と最大値（マージン込み）を計算します。

    Args:
        val (List[Any]): データのリストのリスト。
        margin (float, optional): 最小・最大値の外側に設ける余白の割合. Defaults to 0.05.
        is_log (bool, optional): 対数スケールで計算するかどうか. Defaults to False.

    Returns:
        Tuple[float, float]: 計算された (min_value, max_value) のタプル。
    """
    v_flat = np.concatenate([np.ravel(v) for v in val]) if len(val) > 0 else np.array([])
    try:
        v_flat = np.asarray(v_flat, dtype=float)
    except (ValueError, TypeError):
        v_flat = np.asarray(pd.to_numeric(v_flat, errors='coerce'), dtype=float)
        
    v_flat = v_flat[~np.isnan(v_flat)]
    if len(v_flat) == 0: return (0.1, 10) if is_log else (-1, 1)

    if is_log:
        v_flat = v_flat[v_flat > 0]
        if len(v_flat) == 0: return 0.1, 10
        min0, max0 = np.min(v_flat), np.max(v_flat)
        log_min, log_max = np.log10(min0), np.log10(max0)
        dif = log_max - log_min
        if dif == 0: return 10**(log_min - margin), 10**(log_max + margin)
        return 10**(log_min - dif * margin), 10**(log_max + dif * margin)
    else:
        min0, max0 = np.min(v_flat), np.max(v_flat)
        dif = max0 - min0
        if dif == 0: return min0 - abs(min0) * margin, max0 + abs(max0) * margin
        return min0 - dif * margin, max0 + dif * margin

def valid_xy(x: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """欠損値(NaN)や文字列を除外して、プロットに有効なXとYのペアのみを抽出します。

    Args:
        x (Any): X軸データの配列またはリスト。
        y (Any): Y軸データの配列またはリスト。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 有効な値のみを含む (X配列, Y配列) のタプル。
    """
    try: x = np.asarray(x, dtype=float)
    except: x = np.asarray(pd.to_numeric(x, errors='coerce'), dtype=float)
    try: y = np.asarray(y, dtype=float)
    except: y = np.asarray(pd.to_numeric(y, errors='coerce'), dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

def get_yrange(x: Any, y: Any, xmin: float, xmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Xの指定範囲(xmin ~ xmax)に含まれる、X配列とY配列のペアを抽出します。

    Args:
        x (Any): X軸データの配列。
        y (Any): Y軸データの配列。
        xmin (float): 抽出するXの最小値。
        xmax (float): 抽出するXの最大値。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 範囲内の (X配列, Y配列) のタプル。
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_val, y_val = x[mask], y[mask]
    mask_x = (x_val >= xmin) & (x_val <= xmax)
    return x_val[mask_x], y_val[mask_x]

def get_xrange(x: Any, y: Any, ymin: float, ymax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Yの指定範囲(ymin ~ ymax)に含まれる、X配列とY配列のペアを抽出します。

    Args:
        x (Any): X軸データの配列。
        y (Any): Y軸データの配列。
        ymin (float): 抽出するYの最小値。
        ymax (float): 抽出するYの最大値。

    Returns:
        Tuple[np.ndarray, np.ndarray]: 範囲内の (X配列, Y配列) のタプル。
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_val, y_val = x[mask], y[mask]
    mask_y = (y_val >= ymin) & (y_val <= ymax)
    return x_val[mask_y], y_val[mask_y]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def remove_background(
    signal: Union[List[float], np.ndarray], 
    fc: float = 0.1, 
    d: int = 1, 
    r: int = 6, 
    amp: float = 0.8, 
    Nit: int = 15, 
    pen: str = 'L1_v2', 
    xscale_l: float = 10, 
    xscale_r: float = 10, 
    dx: float = 0.5
) -> np.ndarray:
    """pybeadsを用いてシグナルデータからベースライン（バックグラウンド）成分を高精度に除去・補正します。
    端点での発散を防ぐためのシグモイド関数によるパディング処理が組み込まれています。

    ※ 使用には `pip install pybeads` が必要です。

    Args:
        signal (Union[List[float], np.ndarray]): バックグラウンド除去を行う対象の1次元シグナルデータ。
        fc (float, optional): カットオフ周波数。値を小さくするとより滑らかなベースラインになります. Defaults to 0.1.
        d (int, optional): フィルタの次数. Defaults to 1.
        r (int, optional): 非対称性パラメータ. Defaults to 6.
        amp (float, optional): 正則化パラメータの乗数. Defaults to 0.8.
        Nit (int, optional): 反復回数. Defaults to 15.
        pen (str, optional): ペナルティ関数の種類. Defaults to 'L1_v2'.
        xscale_l (float, optional): 左端パディングのスケール. Defaults to 10.
        xscale_r (float, optional): 右端パディングのスケール. Defaults to 10.
        dx (float, optional): パディングのステップ幅. Defaults to 0.5.

    Returns:
        np.ndarray: バックグラウンドが除去されたクリーンなシグナル配列。
    """
    try:
        import pybeads
    except ImportError:
        raise ImportError("pybeads is required for remove_background. Please install it using 'pip install pybeads'.")
        
    d = 1
    r = 6
    amp = 0.8
    lam0, lam1, lam2 = 0.5*amp, 5*amp, 4*amp
    Nit = 15
    pen = 'L1_v2'
    
    xscale_l, xscale_r = 10, 10
    dx = 0.5
    y_difficult_l = signal[0] * sigmoid(1/xscale_l * np.arange(-5*xscale_l, 5*xscale_l, dx))
    y_difficult_r = signal[-1] * sigmoid(-1/xscale_r * np.arange(-5*xscale_r, 5*xscale_r, dx))
    y_difficult_ext = np.hstack([y_difficult_l, signal, y_difficult_r])
    len_l, len_o, len_r = len(y_difficult_l), len(signal), len(y_difficult_r)
    
    signal_est, bg_est, cost = pybeads.beads(y_difficult_ext, d, fc, r, Nit, lam0, lam1, lam2, pen, conv=None)
    
    return signal_est[len_l:len_l+len_o]