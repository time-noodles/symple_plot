# symple_plot/fit_utils.py
import numpy as np
import optuna
import inspect
from scipy.optimize import curve_fit

def auto_curve_fit(f, xdata, ydata, p0=None, bounds=(-np.inf, np.inf), auto_p0=False, n_trials=100):
    """
    Curve fitを実行。auto_p0=Trueの場合、Optuna(ベイズ最適化)で初期値(p0)を高速に大域探索する。
    """
    xdata = np.asarray(xdata, dtype=float)
    ydata = np.asarray(ydata, dtype=float)
    
    if auto_p0 and p0 is None:
        # Optunaのログ出力を警告のみにしてコンソールを綺麗に保つ
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        sig = inspect.signature(f)
        num_params = len(sig.parameters) - 1 # x以外の引数の数
            
        def objective(trial):
            guess = []
            for j in range(num_params):
                low = bounds[0][j] if isinstance(bounds[0], (list, tuple, np.ndarray)) else -1000.0
                high = bounds[1][j] if isinstance(bounds[1], (list, tuple, np.ndarray)) else 1000.0
                if np.isinf(low): low = -1000.0
                if np.isinf(high): high = 1000.0
                guess.append(trial.suggest_float(f"p{j}", low, high))
                
            try:
                y_pred = f(xdata, *guess)
                return np.mean((ydata - y_pred)**2) # MSEを最小化
            except:
                return float('inf') # 計算エラーになるパラメータは除外
                
        # TPEアルゴリズムによるベイズ最適化
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials)
        p0 = [study.best_params[f"p{j}"] for j in range(num_params)]
        print(f"Auto-selected p0 (Optuna, {n_trials} trials): {np.round(p0, 4)}")
        
    # 最終的な局所最適化
    popt, pcov = curve_fit(f, xdata, ydata, p0=p0, bounds=bounds, maxfev=int(1e4))
    return popt, pcov

def reg_n(reg, x):
    y = np.zeros_like(x)
    for num, i in enumerate(range(len(reg)-1, -1, -1)):
        y = y + reg[num] * x**i
    return y