import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots, remove_background

# 1. ダミーデータの生成 (大きくうねるベースライン + ピーク + ノイズ)
x = np.linspace(0, 100, 500)
baseline = np.sin(x / 15) * 10 + x * 0.2
peaks = 20 * np.exp(-((x - 30)**2) / 2) + 30 * np.exp(-((x - 70)**2) / 5)
y_raw = baseline + peaks + np.random.normal(0, 0.5, 500)

# 2. 全自動でバックグラウンドを除去 (PCの全コアを使って高速化)
y_clean = remove_background(y_raw, auto_opt=True)
y_baseline = y_raw - y_clean

# 3. 結果のプロット
fig, sp = create_symple_plots(1, 2, figsize=(14, 4))

# 左パネル: 元データと推定されたベースライン
sp[0].plot(x, y_raw, col="black", alpha=0.5, lab="Raw Data")
sp[0].plot(x, y_baseline, col="red", linestyle="--", linewidth=2, lab="Estimated Baseline")
sp[0].ax.set_title("Before: Raw & Baseline")

# 右パネル: 除去後のクリーンなデータ
sp[1].plot(x, y_clean, col="blue", lab="Cleaned Signal")
sp[1].ax.set_title("After: Background Removed")

plt.show()