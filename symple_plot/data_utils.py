# symple_plot/data_utils.py
import numpy as np
import pandas as pd

def ensure_2d(data):
    if len(data) == 0: return [[]]
    if not isinstance(data[0], (list, tuple, np.ndarray, pd.Series)): return [data]
    return data

def pad_list(L):
    max_len = max([len(i) for i in L]) if len(L) > 0 else 0
    L_padded = [list(i) + [np.nan] * (max_len - len(i)) for i in L]
    res = []
    for i in L_padded:
        try:
            res.append(np.array(i, dtype=float))
        except (ValueError, TypeError):
            res.append(np.asarray(pd.to_numeric(i, errors='coerce'), dtype=float))
    return res

def minmax(val, margin=0.05, is_log=False):
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

def valid_xy(x, y):
    try: x = np.asarray(x, dtype=float)
    except: x = np.asarray(pd.to_numeric(x, errors='coerce'), dtype=float)
    try: y = np.asarray(y, dtype=float)
    except: y = np.asarray(pd.to_numeric(y, errors='coerce'), dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

def get_yrange(x, y, xmin, xmax):
    """Xの範囲(xmin, xmax)に含まれる、X配列とY配列のペアを返す"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_val, y_val = x[mask], y[mask]
    mask_x = (x_val >= xmin) & (x_val <= xmax)
    return x_val[mask_x], y_val[mask_x]

def get_xrange(x, y, ymin, ymax):
    """Yの範囲(ymin, ymax)に含まれる、X配列とY配列のペアを返す"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_val, y_val = x[mask], y[mask]
    mask_y = (y_val >= ymin) & (y_val <= ymax)
    return x_val[mask_y], y_val[mask_y]