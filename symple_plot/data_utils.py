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

def pad_list(*args: Any) -> List[np.ndarray]:
    """長さが異なる複数のリスト（ジャグ配列）を受け取り、最大の長さに合わせてNaNでパディングします。
    
    pad_list([x, y]) のような単一のリスト渡しにも、
    pad_list(x, y, z) のような1〜複数引数渡しにも両方対応します。

    Args:
        *args (Any): パディング対象のリストまたは配列。

    Returns:
        List[np.ndarray]: 全て同じ長さに揃えられたfloat型のNumPy配列のリスト。
    """
    if len(args) == 0:
        return []
    
    # 既存の pad_list([x, y]) の呼び出しに対応するための展開処理
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        if len(args[0]) == 0:
            L = args[0]
        elif hasattr(args[0][0], '__iter__') and not isinstance(args[0][0], str):
            L = args[0]
        else:
            L = args
    else:
        L = args
        
    max_len = max([len(i) for i in L]) if len(L) > 0 else 0
    res = []
    for i in L:
        try:
            arr = np.array(i, dtype=float)
        except (ValueError, TypeError):
            arr = np.asarray(pd.to_numeric(i, errors='coerce'), dtype=float)
        
        if len(arr) < max_len:
            arr = np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan)
        res.append(arr)
    return res


def valid_xy(*args: Any) -> Tuple[np.ndarray, ...]:
    """欠損値(NaN)や文字列を除外して、プロットに有効なデータのみを抽出します。
    X, Y, Z... と1〜複数渡すことができ、すべての配列で有効なインデックスの要素だけを残します。

    Args:
        *args (Any): データ配列またはそのリスト。

    Returns:
        Tuple[np.ndarray, ...]: 有効な値のみを含む配列のタプル。
    """
    if len(args) == 0:
        return ()
    
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        if len(args[0]) == 0:
            target_args = args[0]
        elif hasattr(args[0][0], '__iter__') and not isinstance(args[0][0], str):
            target_args = args[0]
        else:
            target_args = args
    else:
        target_args = args

    arrs = []
    for a in target_args:
        try:
            arr = np.asarray(a, dtype=float)
        except (ValueError, TypeError):
            arr = np.asarray(pd.to_numeric(a, errors='coerce'), dtype=float)
        arrs.append(arr)
    
    min_len = min([len(a) for a in arrs]) if arrs else 0
    arrs = [a[:min_len] for a in arrs]
    
    if min_len == 0:
        return tuple(arrs)
        
    valid_mask = np.ones(min_len, dtype=bool)
    for a in arrs:
        valid_mask &= ~np.isnan(a)
        valid_mask &= ~np.isinf(a)
        
    return tuple(a[valid_mask] for a in arrs)

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
    pad_mode: str = 'symmetric',   # 🌟 追加: 端点の処理モード（'symmetric', 'reflect', 'edge', 'none' 等）
    pad_len: Optional[int] = None, # 🌟 追加: パディングする長さ（Noneで自動計算: 全長の10%）
    xscale_l: float = 10,  # (後方互換用: 使用しません)
    xscale_r: float = 10,  # (後方互換用: 使用しません)
    dx: float = 0.5,       # (後方互換用: 使用しません)
    auto_opt: bool = False,
    max_iter: int = 10,
    fast_opt: bool = True,
    workers: int = 1
) -> np.ndarray:
    """pybeadsを用いてシグナルデータからベースライン（バックグラウンド）成分を高精度に除去・補正します。
    端点での発散を防ぐため、デフォルトで対称折り返しパディング（symmetric）が適用されます。

    ※ 使用には `pip install pybeads` が必要です。

    Args:
        signal (Union[List[float], np.ndarray]): バックグラウンド除去を行う対象の1次元シグナルデータ。
        fc (float, optional): カットオフ周波数. Defaults to 0.1.
        d (int, optional): フィルタの次数. Defaults to 1.
        r (int, optional): 非対称性パラメータ. Defaults to 6.
        amp (float, optional): 正則化パラメータの乗数. Defaults to 0.8.
        Nit (int, optional): 反復回数. Defaults to 15.
        pen (str, optional): ペナルティ関数の種類. Defaults to 'L1_v2'.
        pad_mode (str, optional): 端点のパディング手法。端点の発散を防ぐには 'symmetric' が最適です. Defaults to 'symmetric'.
        pad_len (Optional[int], optional): パディングの長さ。Noneの場合は全長の10%が自動で設定されます. Defaults to None.
        auto_opt (bool, optional): TrueにするとSciPyの差分進化法を用いて最適な fc, r, amp を自動探索します. Defaults to False.
        max_iter (int, optional): auto_opt=True 時の探索イテレーション数. Defaults to 10.
        fast_opt (bool, optional): 最適化探索中の反復計算を間引き、処理を劇的に高速化します. Defaults to True.
        workers (int, optional): 最適化に使用するCPUスレッド数。-1で全コアを使用します. Defaults to 1.

    Returns:
        np.ndarray: バックグラウンドが除去されたクリーンなシグナル配列。
    """
    try:
        import pybeads as be
    except ImportError:
        print("[symple_plot] pybeads is not installed. Please run `pip install pybeads`.")
        return np.asarray(signal)

    signal_arr = np.asarray(signal, dtype=float)

    def _apply_beads(t_fc: float, t_r: int, t_amp: float, current_Nit: int) -> np.ndarray:
        # 🌟 改善部分: データの10%（最低10ポイント）を対称パディングして端点の連続性を確保
        p_len = pad_len if pad_len is not None else max(len(signal_arr) // 10, 10)
        
        if pad_mode == 'none' or pad_mode is None:
            padded_signal = signal_arr
            p_len = 0
        else:
            try:
                padded_signal = np.pad(signal_arr, p_len, mode=pad_mode)
            except ValueError:
                padded_signal = np.pad(signal_arr, p_len, mode='edge')
        
        lam0 = 0.5 * t_amp
        lam1 = 5.0 * t_amp
        lam2 = 4.0 * t_amp
        
        val_map = {
            'd': int(d), 'fc': float(t_fc), 'r': int(t_r),
            'lam0': float(lam0), 'lam1': float(lam1), 'lam2': float(lam2),
            'Nit': int(current_Nit), 'pen': str(pen)
        }
        
        import inspect
        try:
            sig = inspect.signature(be.beads)
            kwargs = {}
            params = list(sig.parameters.keys())
            for param in params[1:]:
                if param in val_map:
                    kwargs[param] = val_map[param]
            clean_signal, _bg, _cost = be.beads(padded_signal, **kwargs)
        except Exception:
            clean_signal, _bg, _cost = be.beads(
                padded_signal, d=int(d), fc=float(t_fc), r=int(t_r), 
                lam0=float(lam0), lam1=float(lam1), lam2=float(lam2), 
                Nit=int(current_Nit), pen=str(pen)
            )

        if p_len == 0:
            return clean_signal
        return clean_signal[p_len:-p_len]

    if auto_opt:
        from scipy.optimize import differential_evolution
        import warnings

        opt_Nit = max(3, int(Nit * 0.3)) if fast_opt else int(Nit)
        opt_popsize = 3 if fast_opt else 5

        def objective(params: List[float]) -> float:
            t_fc, t_r, t_amp = params
            t_r_int = int(np.round(t_r))
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    S_clean = _apply_beads(t_fc, t_r_int, t_amp, opt_Nit)
                    
                    med = np.median(S_clean)
                    mad = np.median(np.abs(S_clean - med))
                    if mad == 0: mad = 1e-9
                    
                    threshold = med + 3 * mad
                    normal_mask = S_clean <= threshold
                    outlier_mask = S_clean > threshold
                    
                    normal_vals = S_clean[normal_mask]
                    outlier_vals = S_clean[outlier_mask]
                    
                    A = np.abs(np.mean(normal_vals)) + np.std(normal_vals) if len(normal_vals) > 0 else 1e9
                    B = np.max(outlier_vals) if len(outlier_vals) > 0 else 0.0
                    
                    neg_penalty = np.abs(np.min(S_clean)) if np.min(S_clean) < 0 else 0.0
                    loss = -B + A + (neg_penalty * 0.5)
                    
                    if np.isnan(loss) or np.isinf(loss):
                        return 1e9
                    return float(loss)
                except Exception:
                    return 1e9

        bounds = [(0.005, 0.5), (1, 10), (0.1, 5.0)]
        print(f"[symple_plot] Optimizing background parameters... (fast_opt={fast_opt}, workers={workers})")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            updating_mode = 'deferred' if workers != 1 else 'immediate'
            res = differential_evolution(
                objective, bounds, maxiter=max_iter, popsize=opt_popsize, 
                tol=0.05, seed=42, workers=workers, updating=updating_mode
            )
        
        fc = res.x[0]
        r = int(np.round(res.x[1]))
        amp = res.x[2]
        print(f"[symple_plot] Optimization finished: fc={fc:.4f}, r={r}, amp={amp:.4f}")

    return _apply_beads(fc, int(r), amp, int(Nit))