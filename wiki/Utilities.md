# データ前処理・ファイル操作・解析ユーティリティ

`symple_plot` は、グラフ描画の裏側を支えるための高度なユーティリティ群をモジュール化して提供しています。
機能ごとにファイルが分割されていますが（`data_utils`, `file_utils`, `fit_utils`）、これらは `__init__.py` で統合されているため、すべて最上位階層から直接インポートして単独で使用することが可能です。

## 📥 インポート方法

```python
from symple_plot import valid_xy, pad_list, straighten_path, del_file, auto_curve_fit, reg_n
```

---

## 1. データの整形ツール (`data_utils.py`)

### `valid_xy(x, y)`
欠損値 (`NaN`) を含むデータから、プロットや計算に有効なペアだけを抽出します。リストや PandasのSeries、文字列が混ざったデータなど、どんなデータ型が渡されても安全に NumPy配列に変換し、両方に数値が存在するインデックスのみを抽出して返します。

**使用例:**
```python
import numpy as np
from symple_plot import valid_xy

x = [1, 2, np.nan, 4, 'invalid']
y = [10, np.nan, 30, 40, 50]

clean_x, clean_y = valid_xy(x, y)
print(clean_x) # [1. 4.]
print(clean_y) # [10. 40.]
```

### `pad_list(L)`
長さの異なる複数のリストや配列（ジャグ配列）を受け取り、最大の長さに合わせて `NaN` でパディングし、計算に使える綺麗な配列として返します。
また、CSVから読み込んだ文字列データ（例: `'5.0'`）を自動で数値（`float`）に変換し、`'[Header]'` のような変換できない文字列は安全に `NaN` として処理します。

**使用例:**
```python
from symple_plot import pad_list

data1 = [1, 2, 3]
data2 = [4, 5, 6, 7, 8]

padded_data = pad_list([data1, data2])
print(padded_data)
# [array([ 1.,  2.,  3., nan, nan]), 
#  array([ 4.,  5.,  6.,  7.,  8.])]
```

---

## 2. ファイル操作ツール (`file_utils.py`)

### `straighten_path(folder)`
指定したフォルダ内のファイル一覧を取得し、「自然順（Natural Sort）」でソートされたファイルパスのリストを返します。
通常の文字コード順（1, 10, 2...）ではなく、人間の感覚に近い順番（1, 2, ..., 10）で複数の測定データを一括で読み込む際に非常に便利です。

**使用例:**
```python
from symple_plot import straighten_path

# 'data_folder' 内のファイルを自然順で取得
files = straighten_path('./data_folder')
for file in files:
    print(file)
```

### `del_file(targets)`
指定されたパスのファイルやディレクトリをスマートに削除します。ワイルドカード（`*.png` など）やディレクトリ名、またはそれらのリストを直接指定可能です。
毎回のプロットテスト実行前に、前回の出力結果をサクッと一掃したい場合などに役立ちます。

**使用例:**
```python
from symple_plot import del_file

del_file('out/fig/*.png') # pngのみ削除
del_file('out/fig')       # ディレクトリごと削除
```

---

## 3. 解析・最適化ツール (`fit_utils.py`)

### `auto_curve_fit(f, xdata, ydata)`
SciPyの `curve_fit` をラップした高度な非線形フィッティング関数です。
`auto_p0=True` を指定すると、**Optuna（ベイズ最適化）**を用いて高速にパラメータの大域探索を行い、最適な初期値（`p0`）を自動で決定してから局所最適化を実行します。初期値依存の強い物理モデルや、局所解に陥りやすい複雑な関数に最適です。

**使用例:**
```python
import numpy as np
from symple_plot import auto_curve_fit

# 局所解に陥りやすい複雑な関数
def complex_func(x, a, b, c):
    return a * np.sin(b * x) * np.exp(-c * x)

x = np.linspace(0.1, 5, 50)
y = complex_func(x, 2.5, 1.2, 0.5) + np.random.normal(0, 0.1, 50)

# Optunaで初期値を自動探索してフィッティング
popt, pcov = auto_curve_fit(
    complex_func, x, y, 
    auto_p0=True, 
    n_trials=200,                  # 探索回数
    bounds=([0, 0, 0], [10, 5, 2]) # 探索範囲を絞るとさらに高速・高精度
)
print("Optimized Parameters:", popt)
```

### `reg_n(fit, x)`
多項式回帰などで得られた係数の配列（`fit`）と X の配列（`x`）を受け取り、その多項式に基づいて予測された Y の配列を計算して返します。`np.polyfit` 等で得た結果から近似曲線をプロットする際に便利です。

**使用例:**
```python
import numpy as np
from symple_plot import reg_n

x = np.linspace(0, 5, 20)
y = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 1, 20)

# 2次関数で係数を取得
fit_coef = np.polyfit(x, y, 2)

# 係数から予測値 (y_pred) を計算
y_pred = reg_n(fit_coef, x)
```