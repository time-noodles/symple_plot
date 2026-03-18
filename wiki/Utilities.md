# Utilities (便利ツール群)

`symple_plot` には、グラフ描画の前処理や分光データの解析を劇的に効率化するための強力なユーティリティ関数が含まれています。

## 目次

- [📥 インポート方法](#-インポート方法)
- [1. データの整形ツール (`data_utils.py`)](#1-データの整形ツール)
  - [`valid_xy(*args)`](#valid_xyargs)
  - [`pad_list(*args)`](#pad_listargs)
  - [`list_1d(l_2d)`](#list_1dl_2d)
- [2. 解析・抽出ツール (`data_utils.py` / `fit_utils.py`)](#2-解析抽出ツール)
  - [`remove_background` (ベースライン自動除去)](#remove_background)
  - [`f_peak` (ピーク自動抽出)](#f_peaky)
  - [`auto_curve_fit` (大域探索フィッティング)](#auto_curve_fit)
- [3. ファイル操作ツール (`file_utils.py`)](#3-ファイル操作ツール)

---

## 📥 インポート方法

ユーティリティはすべてトップレベルから直接インポートできます。

```python
from symple_plot import (
    valid_xy, pad_list, list_1d, 
    remove_background, f_peak, auto_curve_fit,
    straighten_path, del_file
)
```

---

## 1. データの整形ツール

### `valid_xy(*args)`
欠損値（`NaN`）や無限大（`Inf`）、文字列などを除外して、有効なデータのみを抽出します。可変長引数に対応しており、複数の配列を渡した場合、**すべての配列で共通して有効なインデックスの要素のみ**を残します。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `*args` | `Any` | - | フィルタリングしたい1つ以上のデータ配列（またはリスト）。 |

**【実行できるサンプルコード】**
```python
import numpy as np
from symple_plot import valid_xy

# ノイズや欠損値を含むダミーデータ
x = [1, 2, np.nan, 4, 5]
y = [10, 20, 30, np.inf, 50]
z = [100, 200, 300, 400, "error"]

# すべての配列において正常な数値を持つ行 (インデックス0と1) だけを抽出
x_clean, y_clean, z_clean = valid_xy(x, y, z)

print("X:", x_clean) # X: [1. 2.]
print("Y:", y_clean) # Y: [10. 20.]
print("Z:", z_clean) # Z: [100. 200.]
```

### `pad_list(*args)`
長さが異なる複数のリスト（ジャグ配列）を受け取り、最大の長さに合わせて末尾を `NaN` でパディングし、同じ長さの NumPy 配列に揃えます。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `*args` | `Any` | - | パディングして長さを揃えたい複数のリスト。 |

**【実行できるサンプルコード】**
```python
from symple_plot import pad_list

L1 = [1, 2]
L2 = [1, 2, 3, 4]
L3 = [1, 2, 3]

# 最大長さ(4)に合わせて NaN で埋められる
padded = pad_list(L1, L2, L3)

for arr in padded:
    print(arr)
# [ 1.  2. nan nan]
# [ 1.  2.  3.  4.]
# [ 1.  2.  3. nan]
```

### `list_1d(l_2d)`
2次元のリストや配列のリストを、高速に1次元に平坦化（Flatten）します。`symple_plot` で複数パネルにまたがる軸の共通化や、一括の色指定 (`vcol` 等) を行う際に便利です。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `l_2d` | `Any` | - | 平坦化したい2次元のリストやタプル。 |

**【実行できるサンプルコード】**
```python
from symple_plot import list_1d

data = [[1, 2, 3], [4, 5], [6]]
flat_data = list_1d(data)

print(flat_data)
# [1, 2, 3, 4, 5, 6]
```

---

## 2. 解析・抽出ツール

### `remove_background`
XRD、ラマン分光、XPSなどのスペクトルデータから、大きくうねるバックグラウンド（ベースライン）を高精度に除去します。`auto_opt=True` を指定することで、差分進化法を用いて最適なパラメータを**全自動探索**します。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `signal` | `Array` | - | バックグラウンド除去を行う対象の1次元配列。 |
| `auto_opt` | `bool` | `False` | `True`で差分進化法による最適な `fc`, `r`, `amp` の自動探索を実行します。 |
| `pad_mode` | `str` | `'symmetric'` | 端点での発散を防ぐためのパディング手法。 |
| `fast_opt` | `bool` | `True` | 自動探索中の計算を間引き、数倍高速化します。 |
| `workers` | `int` | `1` | 自動探索に使うCPUコア数。`-1` で全コアを使用し爆速化します。 |
| `fc` | `float` | `0.1` | (手動用) カットオフ周波数。小さいほど滑らかなベースラインになります。 |

**【実行できるサンプルコード】**
※実行には `pip install pybeads` が必要です。

```python
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
```

### `f_peak(Y, ...)`
データ配列からピーク（山・谷）のインデックス（X座標の位置）を抽出します。
内部で自動的にデータを `0.0〜1.0` に正規化するため、「最大振幅の何%以上の高さ/深さか」という相対的な指定が直感的に行えます。
`mode` 引数により、上に凸のピーク（山）、下に凸のピーク（谷）、あるいはその両方を抽出可能です。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `Y` | `Array` | - | ピークを探す対象の1次元データ配列。 |
| `mode` | `str` | `'+-'` | 抽出するピークの方向。`'+'`(山のみ), `'-'`(谷のみ), `'+-'`(山と谷の両方)。 |
| `distance` | `int` | `None` | 隣接するピーク間の最小距離（データポイント数）。 |
| `rel_height` | `float`| `None` | 全体の振幅に対する割合（`0.0〜1.0`）で指定するピークの最小高さ/深さ。 |
| `height` | `float`| `None` | 絶対値で指定するピークの最小高さ/深さ（`rel_height`が優先されます）。 |
| `**kwargs` | `Any`  | - | `scipy.signal.find_peaks` に渡される追加引数 (`prominence`等)。 |

**【実行できるサンプルコード】**
```python
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots, f_peak

# 1. ダミーデータの生成 (山と谷が混在する干渉・振動シグナル)
x = np.linspace(0, 100, 500)
y = np.sin(x / 5) * 10 + np.cos(x / 2) * 5 + np.random.normal(0, 0.5, 500)

# 2. ピーク抽出 (山と谷の両方を抽出、最小距離: 20ポイント)
# mode='+-' がデフォルトなので指定しなくても両方抽出されます
peak_indices = f_peak(y, mode='+-', distance=20, rel_height=0.1)

# 3. プロットして確認
fig, sp = create_symple_plots(figsize=(10, 4))
sp.plot(x, y, col="black", alpha=0.6, lab="Signal")

# 抽出したインデックスを使って散布図でマーキング
sp.scatter(x[peak_indices], y[peak_indices], col="red", marker="o", size=80, lab="Detected Peaks (+ and -)")

plt.show()
```

### `auto_curve_fit`
多項式回帰、または非線形フィッティング（Curve Fit）を実行します。ユーザー定義の関数を渡す際、`auto_p0=True` を指定するとSciPyの差分進化法を用いて初期値(`p0`)の大域探索を自動で行い、**「初期値が悪くてフィッティングが失敗する（局所解に陥る）」というMatplotlib/SciPyあるあるを回避**します。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `f` | `int` / `Callable` | - | フィッティングする関数、または多項式回帰の次数(int)。 |
| `xdata`, `ydata` | `Array` | - | X軸およびY軸のデータ配列。 |
| `p0` | `Array` | `None` | (手動用) 最適化の初期値パラメータ。 |
| `bounds` | `tuple` | `(-inf, inf)` | パラメータの探索範囲。 |
| `auto_p0` | `bool` | `False` | `True`で差分進化法を用いて初期値(p0)を自動大域探索します。 |

**【実行できるサンプルコード】**
```python
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots
from symple_plot.fit_utils import auto_curve_fit

# 1. ダミーデータの生成とユーザー定義関数
def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

x = np.linspace(0, 4, 50)
y_true = exp_decay(x, 2.5, 1.3, 0.5)
y_noise = y_true + 0.2 * np.random.normal(size=len(x))

# 2. 初期値(p0)を完全自動探索してフィッティング
popt, pcov = auto_curve_fit(exp_decay, x, y_noise, auto_p0=True)
print(f"Optimized Parameters: a={popt[0]:.2f}, b={popt[1]:.2f}, c={popt[2]:.2f}")

# 3. プロット
fig, sp = create_symple_plots()
sp.scatter(x, y_noise, col="black", alpha=0.6, lab="Data")
sp.plot(x, exp_decay(x, *popt), col="red", linewidth=2, lab="Auto Curve Fit")
plt.show()
```

---

## 3. ファイル操作ツール

### `straighten_path(folder)`
指定されたフォルダ内のファイルパスを取得し、**自然順（Natural Sort）**でソートします。
標準の `glob` 等では `['1.txt', '10.txt', '2.txt']` のように文字列順になってしまうものを、人間が期待する `['1.txt', '2.txt', '10.txt']` の順序で取得できます。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `folder` | `str` | - | 検索対象のフォルダパス（例: `'./data_folder'`）。 |

```python
from symple_plot import del_file

# tempフォルダ内のファイルをソート
straighten_path("./temp/")
```

### `del_file(targets)`
指定されたパターンに一致するファイルまたはディレクトリを一括で削除します。ワイルドカード(`*`)に対応しており、一時ファイルのリセットなどに便利です。

| 引数 | 型 | 初期値 | 説明 |
| :--- | :--- | :--- | :--- |
| `targets` | `str` / `list` | - | 削除対象のパス、またはワイルドカードを含む文字列。 |

```python
from symple_plot import del_file

# tempフォルダ内のすべてのPNG画像と、特定のCSVを削除
del_file(["./temp/*.png", "./results/regression_results.csv"])
```