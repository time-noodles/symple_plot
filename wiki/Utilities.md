# データ前処理・ファイル操作・解析ユーティリティ

`symple_plot` は、グラフ描画の裏側を支えるための高度なユーティリティ群をモジュール化して提供しています。
機能ごとにファイルが分割されていますが（`data_utils`, `file_utils`, `fit_utils`）、これらは `__init__.py` で統合されているため、すべて最上位階層から直接インポートして単独で使用することが可能です。

## 📥 インポート方法

```python
from symple_plot import valid_xy, get_yrange, get_xrange, pad_list, straighten_path, del_file, auto_curve_fit, reg_n
```

---

## 1. データの整形ツール (`data_utils.py`)

### `get_yrange(x, y, xmin, xmax)` / `get_xrange(x, y, ymin, ymax)`
指定された範囲内に含まれる **X の配列** と、それに対応する **Y の配列** をフィルタリングして返します。欠損値も安全に無視されます。プロットの特定領域だけを切り出して、積分や平均値の計算に回したい場合に非常に便利です。

**▶ 実行して試せるコード:**
```python
import numpy as np
from symple_plot import get_yrange, get_xrange

x = np.array([1, 2, 3, 4, 5])
y = np.array([10, np.nan, 20, 90, 30]) # 欠損値を含むテストデータ

# Xが1.5から3.5の範囲にあるデータの、X配列とY配列を抽出
x_fil, y_fil = get_yrange(x, y, 1.5, 3.5)
print(f"Filtered X (range 1.5~3.5): {x_fil}") # 出力: [3.]
print(f"Filtered Y: {y_fil}")                # 出力: [20.]

# Yが25から100の範囲にあるデータの、X配列とY配列を抽出
x_fil2, y_fil2 = get_xrange(x, y, 25, 100)
print(f"Filtered X (Y range 25~100): {x_fil2}") # 出力: [4. 5.]
print(f"Filtered Y: {y_fil2}")                  # 出力: [90. 30.]
```

### `valid_xy(x, y)`
欠損値 (`NaN`) や文字列が混ざったデータから、プロットや計算に有効なペアだけを抽出します。

**▶ 実行して試せるコード:**
```python
import numpy as np
from symple_plot import valid_xy

# 無効な値 (NaNや文字列) が混入したデータ
x_raw = [1, 2, np.nan, 4, 'invalid']
y_raw = [10, np.nan, 30, 40, 50]

clean_x, clean_y = valid_xy(x_raw, y_raw)
print("Clean X:", clean_x) # 出力: [1. 4.]
print("Clean Y:", clean_y) # 出力: [10. 40.]
```

### `pad_list(L)`
長さの異なる複数のリストや配列（ジャグ配列）を受け取り、最大の長さに合わせて `NaN` でパディングし、計算に使える綺麗な配列として返します。

**▶ 実行して試せるコード:**
```python
from symple_plot import pad_list

data1 = [1, 2, 3]
data2 = [4, 5, 6, 7, 8]

padded_data = pad_list([data1, data2])
print("Padded Data 1:\n", padded_data[0]) # 出力: [ 1.  2.  3. nan nan]
print("Padded Data 2:\n", padded_data[1]) # 出力: [ 4.  5.  6.  7.  8.]
```

---

## 2. ファイル操作ツール (`file_utils.py`)

### `straighten_path(folder)` & `del_file(targets)`
ファイルの自然順（Natural Sort: 1, 2, ..., 10）ソート取得と、ファイル・ディレクトリのスマートな一括削除を行います。

**▶ 実行して試せるコード:**
```python
import os
from symple_plot import straighten_path, del_file

# テスト用にダミーフォルダとファイルを作成
test_dir = 'dummy_test_folder'
os.makedirs(test_dir, exist_ok=True)
for i in [1, 10, 2]:
    with open(os.path.join(test_dir, f'data_{i}.txt'), 'w') as f:
        f.write('test')

# 自然順でファイルを取得 (文字コード順の 1, 10, 2 ではなく、1, 2, 10 で取得される)
sorted_files = straighten_path(test_dir)
print("Sorted Files:")
for file in sorted_files:
    print(f" - {file}")

# テストが終わったらフォルダごとまるっと削除
del_file(test_dir)
print(f"Deleted directory: {test_dir}")
```

---

## 3. 解析・最適化ツール (`fit_utils.py`)

### `auto_curve_fit(f, xdata, ydata)`
`auto_p0=True` を指定すると、**Optuna（ベイズ最適化）**を用いて高速にパラメータの大域探索を行い、最適な初期値（`p0`）を自動で決定してから局所最適化を実行します。

通常の `curve_fit` では、周波数や位相が含まれる「振動関数」などは、初期値を与えないと局所解（間違った周期）に陥って100%失敗します。Optuna連携はこのような初期値依存性の強いモデルを解決します。

**▶ 実行して試せるコード:**
```python
import numpy as np
from symple_plot import auto_curve_fit

# 初期値がないと確実に失敗する減衰正弦波 (Damped Sine Wave)
def damped_sine(x, amp, decay, freq, phase):
    return amp * np.exp(-decay * x) * np.sin(2 * np.pi * freq * x + phase)

# テスト用データの生成 (正解: amp=5.0, decay=0.5, freq=2.0, phase=0.0)
x_data = np.linspace(0, 5, 200)
true_params = [5.0, 0.5, 2.0, 0.0]
y_data = damped_sine(x_data, *true_params) + np.random.normal(0, 0.2, 200)

# Optunaで初期値を自動探索してフィッティング
popt, pcov = auto_curve_fit(
    damped_sine, x_data, y_data, 
    auto_p0=True, 
    n_trials=150,  # 探索回数
    # パラメータの探索範囲: amp, decay, freq, phase
    bounds=([0, 0, 0.1, -np.pi], [10, 2, 5.0, np.pi]) 
)

print("\n=== Fitting Results ===")
print(f"True Params: {true_params}")
print(f"Opt Params : {np.round(popt, 2)}")
```

### `reg_n(fit, x)`
多項式回帰などで得られた係数の配列（`fit`）と X の配列（`x`）を受け取り、その多項式に基づいて予測された Y の配列を計算して返します。

**▶ 実行して試せるコード:**
```python
import numpy as np
from symple_plot import reg_n

# テストデータ (2次関数ベース)
x = np.linspace(0, 5, 20)
y = 2 * x**2 - 3 * x + 1 + np.random.normal(0, 1, 20)

# np.polyfit で2次関数の係数を取得 [a, b, c]
fit_coef = np.polyfit(x, y, 2)
print("Fitted Coefficients:", np.round(fit_coef, 2))

# 係数から予測値 (y_pred) を計算
y_pred = reg_n(fit_coef, x)
print("First 5 Predicted Y values:", np.round(y_pred[:5], 2))
```