# symple_plot

`symple_plot` は、Matplotlibをベースにした研究・データ解析用の強力なグラフ描画ラッパーライブラリです。
論文やプレゼンテーションでそのまま使える美しい図を、最小限のコードで生成することを目的に設計されています。

## ✨ 主な特徴 (Features)

* **Auto Smart Formatter**: 軸のスケールを自動解析し、`5.0 × 10^4` のような美しい科学的記数法に自動フォーマットします。複数のデータ間で指数も統一されます。
* **Inset Zoom (自動拡大図)**: 範囲 (`xlim` または `ylim`) を指定するだけで、データの該当部分を自動探索し、小窓（Inset）として拡大描画します。
* **GrADS & Perceptually Uniform Colormaps**: 気象学で人気のGrADSカラーマップを標準搭載。他にも `turbo`, `plasma` などの知覚的均等カラーマップを視認性の高い範囲に絞って適用します。
* **高度な回帰分析 (Optuna搭載)**: 任意の次数の多項式回帰や、ユーザー定義関数による非線形フィッティングを自動実行します。`optuna` を用いたベイズ最適化による初期値(`p0`)の自動大域探索にも対応しています。
* **ワンライナー設定**: 軸ラベル、凡例、対数軸、範囲、垂直/水平の補助線(`vx`, `hy`)、アスペクト比などを1行の引数で完結させます。

---

## 📦 インストール (Installation)

GitHubから直接インストールできます。

```bash
pip install git+https://github.com/time-noodles/symple_plot.git
```

## 🚀 基本的な使い方 (Basic Usage)

`create_symple_plots` を使ってグラフ枠を生成し、`plot` や `scatter` メソッドを呼び出します。

```python
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots

# 1行1列のグラフを生成
fig, sp = create_symple_plots(nrows=1, ncols=1)

x = np.linspace(0, 10, 50)
y1 = np.sin(x)
y2 = np.cos(x)

# データをリストで渡し、ラベルや引数を指定するだけ
sp.plot(
    [x, x], [y1, y2],
    alab=["Time (s)", "Amplitude (a.u.)"],
    lab=["Sample A", "Sample B"],
    linestyle=['-', '--'],
    linewidth=2
)

plt.show()
```

**▼ 出力例:**
![基本プロット](images/example0_basic.png)

---

## 🛠 機能リファレンスと実例 (Examples)

### 1. 指数の自動統一と科学的記数法

大きな桁数のデータをプロットすると、軸全体で指数が統一され、`$0.5 \times 10^4$` のように美しくフォーマットされます。

```python
fig, sp = create_symple_plots(1, 1)

x = np.linspace(1, 5, 5)
y = np.array([5000, 10000, 15000, 20000, 25000])

sp.scatter(x, y, alab=["X", "Large Value"])
```

**▼ 出力例:**
![指数統一](images/example1_exponent.png)

### 2. Inset Zoom（自動探索・拡大小窓）

特定の部分を強調したい場合、`add_inset_zoom` メソッドを使います。範囲を指定するだけでY方向のスケールは自動計算されます。

```python
fig, sp = create_symple_plots(1, 1)

x = np.linspace(0, 10, 500)
y = np.sin(x) + 5 * np.exp(-((x - 7.5)**2) / 0.01)

sp.plot(x, y, alab=["X", "Intensity"])

# x=7.2〜7.8の範囲を指定すると、Yの範囲を自動探索して左上に拡大図を生成
sp.add_inset_zoom(xlim=[7.2, 7.8], bounds='upper left')
```

**▼ 出力例:**
![Inset Zoom](images/example2_zoom.png)

### 3. 回帰分析と補助線 (Regression & Guide Lines)

`Regression` メソッドは、引数に「整数」を渡せば多項式回帰を、「関数オブジェクト」を渡せば非線形フィッティングを実行します。`auto_p0=True` を指定すると、Optunaによるベイズ最適化で最適な初期値を自動探索します。

```python
fig, sp_arr = create_symple_plots(1, 2, figsize=(12, 5))

# --- パネル1: 多項式回帰 (3次関数) ---
sp1 = sp_arr[0]
x1 = np.linspace(-5, 5, 30)
y1 = 0.5 * x1**3 - 2 * x1 + np.random.normal(0, 5, 30)
sp1.scatter(x1, y1, alab=["X", "Y"], lab="Data")
sp1.Regression(regr=3) # 3次関数でフィッティング
sp1.ax.set_title("Polynomial Regression (regr=3)")

# --- パネル2: 任意関数フィッティングと補助線 ---
sp2 = sp_arr[1]
x2 = np.linspace(0.1, 5, 50)
y2 = 2.5 * np.exp(-1.2 * x2) + np.random.normal(0, 0.05, 50)

# vx=[1, 3] で垂直な補助線を、hy=0 で水平線を引く
sp2.scatter(x2, y2, alab=["Time (s)", "Intensity"], lab="Data",
            vx=[1, 3], vcol='red', vstyle='--', vwidth=1.5,
            hy=0, hcol='blue', hstyle=':', hwidth=1.0)

# 任意の関数を定義してフィッティング
def exp_decay(x, a, b):
    return a * np.exp(-b * x)

# Optunaを使用して初期値を自動探索
sp2.Regression(regr=exp_decay, auto_p0=True, n_trials=50, bounds=([0, 0], [10, 5]))
sp2.ax.set_title("Optuna Auto Fit & Guide Lines")
```

**▼ 出力例:**
![回帰と補助線](images/example3_regression.png)

### 4. 画像プロット (Imshow) と 3D プロット

2Dマッピング画像や3D空間のプロットもサポートしています。

```python
fig, sp_arr = create_symple_plots(1, 2, figsize=(12, 5))

# --- パネル1: Imshow ---
sp1 = sp_arr[0]
z_im = np.random.rand(50, 50) * 1e-4
sp1.imshow(
    [np.linspace(0, 5, 50)], [np.linspace(0, 5, 50)], z_im,
    vmax=1e-4, alab=["X (um)", "Y (um)", "Intensity"]
)

# --- パネル2: 3D Scatter ---
sp_arr[1].ax.remove()
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
sp2 = symple_plot(ax_3d)

z_3d = np.linspace(0, 10, 100)
sp2.tdscatter(
    np.sin(z_3d), np.cos(z_3d), z_3d,
    alab=["X", "Y", "Z"]
)
```

**▼ 出力例:**
![Imshowと3D](images/example4_3d.png)

---

### 5. 論文・プレゼン用ユーティリティ (Auto Style & Labels)

論文やスライド作成を加速するため、描画スタイルの一括適用（`style`）と、各パネルへの `(a)`, `(b)` ラベルの自動付与（`auto_label`）をサポートしています。

```python
# style='slide' で太字・大きな文字に設定。auto_label=True で (a), (b) を自動付与
fig, sp_arr = create_symple_plots(1, 2, figsize=(10, 4), style='slide', auto_label=True)

x = np.linspace(0, 5, 20)
sp_arr[0].plot(x, np.exp(x), alab=["Time", "Growth"], lab="Exponential")
sp_arr[1].scatter(x, x**3, alab=["Time", "Value"], size=80, marker='s', lab="Quadratic")
```

**▼ 出力例:**
![Auto Style & Labels](images/example5_utils.png)

---

### 6. 個別カラー指定と強制ズーム (Custom Color & Auto Zoom)

`col` 引数で特定のプロットだけ色を変更したり、`zoom` 引数を使って「後から追加したデータ」の範囲にグラフ全体をピタッとフォーカスさせることができます。

```python
fig6, sp_arr6 = create_symple_plots(2, 2)

x_bg = np.linspace(0, 20, 100)
y_bg = np.sin(x_bg)

# --- 左パネル: `zoom='x'` のテスト（Y軸は維持し、X軸だけ上書きズーム） ---
sp6_1 = sp_arr6[0]
sp6_2 = sp_arr6[1]
sp6_3 = sp_arr6[2]
sp6_1.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
sp6_2.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
sp6_3.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])

x_target = np.linspace(5, 10, 50)
y_target = np.sin(x_target)
# zoom='x' を指定すると、Y軸の高さ(±1)は保ったまま、X軸だけが 5〜10 にズームされる
sp6_1.plot(x_target, y_target, col='red', lab="Target (zoom='x')", zoom='x', linewidth=3)
sp6_2.plot(x_target, y_target, col='red', lab="Target (zoom='y')", zoom='y', linewidth=3)
sp6_3.plot(x_target, y_target, col='red', lab="Target (zoom='both')", zoom='xy', linewidth=3)
sp6_1.ax.set_title("zoom='x'", fontsize=14)
sp6_2.ax.set_title("zoom='y'", fontsize=14)
sp6_3.ax.set_title("zoom='xy'", fontsize=14)

# --- 右パネル: `zoomx` のテスト（プロットと同時に拡大小窓を自動生成） ---
sp6_4 = sp_arr6[3]
sp6_4.plot(x_bg, y_bg, col='gray', lab="Full Data", alab=["X", "Y"])

# zoomx=[7.2, 7.8] を引数に入れるだけで、勝手に add_inset_zoom が発動する！
x_peak = np.linspace(7.2, 7.8, 50)
y_peak = np.sin(x_peak) + 3 * np.exp(-((x_peak - 7.5)**2) / 0.01)
sp6_4.plot(x_peak, y_peak, col='green', lab="Sharp Peak", zoomx=[7.2, 7.8])
sp6_4.ax.set_title("Auto Inset Zoom (zoomx)", fontsize=14)
```

**▼ 出力例:**
![Custom Color and Zoom](images/example6_zoom_col.png)

---

## ⚙️ パラメータ一覧 (Kwargs Reference)

引数の適用先に応じて、2つの表に分けています。

### 1. `create_symple_plots` の引数 (グラフ枠の生成・初期設定)

| 引数名 | 型 | 説明 |
| --- | --- | --- |
| `style` | str | `'paper'` または `'slide'` で描画スタイルを一括適用 |
| `auto_label`| bool | `True`で各パネルの左上に (a), (b)... と自動でラベルを付与 |

### 2. 各種描画メソッドの引数 (plot, scatter, imshow, tdscatter 等)

| 引数名 | 型 | 説明 |
| --- | --- | --- |
| `aspect` | float | グラフのアスペクト比を指定（例: 1.0で正方形、0.5で横長） |
| `alab` | list | 軸ラベルを指定 `[xlabel, ylabel, (zlabel)]` |
| `lab` | list/str | 凡例のテキスト |
| `cx` / `cy` | list | 軸の描画範囲を固定 `[min, max]` |
| `logx` / `logy` | bool | 軸を対数スケールにする |
| `nox` / `noy` | bool | 軸の目盛りラベルのみを非表示にする |
| `zoom` | str | 今回渡したデータ範囲に枠を強制拡大 ('x', 'y', 'xy') |
| `zoomx` / `zoomy` | list | 指定範囲 `[min, max]` の拡大図（Inset Zoom小窓）を生成 |
| `col` | str/list | プロットの色を指定 ('red', 'plasma', 'mode1' 等) |
| `vx` / `hy` | list/float | 垂直線(x) / 水平線(y) を引く座標 |
| `vcol` / `hcol` | str | 垂直/水平線の色 (デフォルト: 'gray') |
| `vstyle` / `hstyle` | str | 垂直/水平線のスタイル (デフォルト: '--') |
| `marker` / `size` | - | [scatter等] マーカー形状とサイズ |
| `linestyle` / `linewidth` | - | [plot等] 線の種類と太さ |
| `auto_p0` | bool | [Regression] Trueにすると、Optunaで非線形フィッティングの初期値を自動探索する |
| `n_trials`| int  | [Regression] `auto_p0=True` の際のOptuna探索回数（デフォルト: 100） |
| `p0` / `bounds`| - | [Regression] 非線形フィッティングの初期推測値 / 探索範囲 |

---

## 🧰 その他の便利ツール (Utilities)

`symple_plot` には、グラフ描画以外にも実験データの整理やモデリングに役立つ便利な関数群が含まれています。データの前処理、ファイルの自然順ソート、Optunaを単独で呼び出すフィッティング機能などが揃っています。

詳細な使い方や実例については、以下のWikiページをご覧ください。
👉 **[データ前処理・ファイル操作・解析ユーティリティ (Wiki)](wiki/Utilities.md)**

---

## 謝辞 (Acknowledgments)
The core functionalities and documentation of this library were developed with the assistance of an AI language model (Google Gemini).

---

Copyright (c) 2026 Your Name. All rights reserved.