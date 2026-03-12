# symple_plot/fit_utils.py
from typing import Callable, Union, Tuple, Optional, Any, List
import numpy as np
import inspect
from scipy.optimize import curve_fit, differential_evolution

def auto_curve_fit(
    f: Union[int, Callable], 
    xdata: Union[List[float], np.ndarray], 
    ydata: Union[List[float], np.ndarray], 
    p0: Optional[Union[List[float], np.ndarray]] = None, 
    bounds: Union[Tuple[float, float], Tuple[List[float], List[float]]] = (-np.inf, np.inf), 
    auto_p0: bool = False, 
    n_trials: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """多項式回帰、または非線形フィッティング（Curve Fit）を実行します。
    
    `f`に整数を指定した場合はその次数の多項式回帰(np.polyfit)を実行します。
    関数を指定し、`auto_p0=True`を指定した場合は、差分進化法で初期値(p0)の大域探索を自動で行い、
    局所解に陥るのを防ぎます。

    Args:
        f (Union[int, Callable]): フィッティングする関数、または多項式回帰の次数(int)。
        xdata (Union[List[float], np.ndarray]): X軸データ。
        ydata (Union[List[float], np.ndarray]): Y軸データ。
        p0 (Optional[Union[List[float], np.ndarray]], optional): 最適化の初期値パラメータ. Defaults to None.
        bounds (Union[Tuple[float, float], Tuple[List[float], List[float]]], optional): パラメータの探索範囲. Defaults to (-np.inf, np.inf).
        auto_p0 (bool, optional): Trueにすると、SciPyの差分進化法を用いて初期値(p0)を自動で大域探索します. Defaults to False.
        n_trials (int, optional): auto_p0=Trueの場合の、差分進化法の最大反復回数. Defaults to 100.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - popt: 最適化されたパラメータの配列（多項式の場合は係数）
            - pcov: パラメータの共分散行列
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    
    # fが整数の場合: 多項式フィッティング
    if isinstance(f, int):
        # cov=Trueで係数(popt)と共分散行列(pcov)を取得
        popt, pcov = np.polyfit(xdata, ydata, f, cov=True)
        return popt, pcov

    # 以下、fが関数の場合 (非線形フィッティング)
    if auto_p0 and p0 is None:
        sig = inspect.signature(f)
        num_params = len(sig.parameters) - 1 # x以外の引数の数
        
        # 差分進化法用の境界(bounds)を準備: [(min0, max0), (min1, max1), ...]
        de_bounds = []
        for j in range(num_params):
            if isinstance(bounds[0], (list, tuple, np.ndarray)) and isinstance(bounds[1], (list, tuple, np.ndarray)):
                low = bounds[0][j]
                high = bounds[1][j]
            else:
                low = bounds[0]
                high = bounds[1]
                
            if np.isinf(low): low = -1000.0
            if np.isinf(high): high = 1000.0
            de_bounds.append((low, high))
            
        def _objective(params):
            try:
                y_pred = f(xdata, *params)
                return np.mean((ydata - y_pred)**2)
            except Exception:
                return np.inf
                
        # 差分進化法による大域探索 (workers=-1で並列化)
        result = differential_evolution(
            _objective,
            bounds=de_bounds,
            strategy='best1bin',
            maxiter=n_trials,
            popsize=15,
            #workers=-1,
            tol=0.01
        )
        
        p0 = result.x
        print(f"Auto-selected p0 (Differential Evolution, maxiter={n_trials}): {np.round(p0, 4)}")
        
    # 最終的な局所最適化
    popt, pcov = curve_fit(f, xdata, ydata, p0=p0, bounds=bounds, maxfev=int(1e4))
    return popt, pcov

def reg_n(reg: Union[List[float], np.ndarray], x: Union[List[float], np.ndarray]) -> np.ndarray:
    """多項式回帰で得られた係数の配列から、指定したX座標に対応する予測値(Y)を計算します。

    Args:
        reg (Union[List[float], np.ndarray]): 多項式の係数配列 (最高次から定数項の順、np.polyfitの戻り値と同じ)。
        x (Union[List[float], np.ndarray]): 予測値を計算したいX座標の配列。

    Returns:
        np.ndarray: 計算された予測値(Y)の配列。
    """
    y = np.zeros_like(x)
    for num, i in enumerate(range(len(reg)-1, -1, -1)):
        y = y + reg[num] * x**i
    return y