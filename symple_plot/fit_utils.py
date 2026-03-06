# symple_plot/fit_utils.py

import numpy as np
import inspect
from scipy.optimize import curve_fit, differential_evolution

def auto_curve_fit(f, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), auto_p0=False, n_trials=100):
    """
    Curve fitを実行。
    - fに整数(int)を指定した場合: その次数の多項式回帰(np.polyfit)を実行。
    - fに関数を指定し、auto_p0=Trueの場合: 差分進化法で初期値(p0)を高速に大域探索する。
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

def reg_n(reg, x):
    y = np.zeros_like(x)
    for num, i in enumerate(range(len(reg)-1, -1, -1)):
        y = y + reg[num] * x**i
    return y