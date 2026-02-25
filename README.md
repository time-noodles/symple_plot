以下のコードブロック右上の「コピー」ボタンを押して、そのまま `README.md` に貼り付けてください。

```markdown
# symple_plot

`symple_plot` は、Matplotlibをベースにした研究・データ解析用の強力なグラフ描画ラッパーライブラリです。
論文やプレゼンテーションでそのまま使える美しい図を、最小限のコードで生成することを目的に設計されています。

## ✨ 主な特徴 (Features)

* **Auto Smart Formatter**: 軸のスケールを自動解析し、`5.0 × 10^4` のような美しい科学的記数法に自動フォーマットします。複数のデータ間で指数も統一されます。
* **Inset Zoom (自動拡大図)**: 範囲 (`xlim` または `ylim`) を指定するだけで、データの該当部分を自動探索し、小窓（Inset）として拡大描画します。
* **GrADS & Perceptually Uniform Colormaps**: 気象学で人気のGrADSカラーマップを標準搭載。他にも `turbo`, `plasma` などの知覚的均等カラーマップを視認性の高い範囲に絞って適用します。
* **多項式回帰 (Regression)**: 任意の次数の回帰曲線を自動で引き、係数やR2スコアを1つのCSVファイル (`regression_results.csv`) に追記保存します。
* **ワンライナー設定**: 軸ラベル、凡例、対数軸、範囲、目盛りの非表示などを1行の引数で完結させます。

---

## 📦 インストール (Installation)

GitHubから直接インストールできます。（※Privateリポジトリの場合はアクセストークンが必要です）

```bash
pip install git+[https://github.com/あなたのユーザー名/symple_plot.git](https://github.com/あなたのユーザー名/symple_plot.git)

```

※ `あなたのユーザー名` の部分はご自身のGitHub IDに書き換えてください。

---

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

---

## 🛠 機能リファレンスと実例 (Examples)

### 1. 指数の自動統一と科学的記数法

大きな桁数のデータをプロットすると、Y軸などが自動的に `$1.5 \times 10^4$` のようにフォーマットされます。

```python
fig, sp = create_symple_plots(1, 1)

x = np.linspace(1, 5, 5)
y = np.array([5000, 10000, 15000, 20000, 25000])

sp.scatter(x, y, alab=["X", "Large Value"])

```

### 2. Inset Zoom（自動探索・拡大小窓）

特定の部分を強調したい場合、`add_inset_zoom` メソッドを使います。

```python
fig, sp = create_symple_plots(1, 1)

x = np.linspace(0, 10, 500)
y = np.sin(x) + 5 * np.exp(-((x - 7.5)**2) / 0.01)

sp.plot(x, y, alab=["X", "Intensity"])

# x=7.2〜7.8の範囲を指定すると、Yの範囲を自動探索して左上に拡大図を生成
sp.add_inset_zoom(xlim=[7.2, 7.8], bounds='upper left')

```

### 3. 多項式回帰 (Regression)

散布図を描画し、そのまま `Regression` を呼ぶことで近似曲線を引けます。

```python
fig, sp = create_symple_plots(1, 1)

x = np.linspace(-5, 5, 30)
y = 0.5 * x**3 - 2 * x + np.random.normal(0, 5, 30)

sp.scatter(x, y, alab=["X", "Y"])

# 3次関数でフィッティングし、結果をカレントディレクトリのCSVに保存
sp.Regression(regr=3, directory='./')

```

### 4. 対数軸と範囲の強制指定

`logx` や `logy` を `True` にするだけで対数スケールになります。また、`cx`, `cy` で軸の範囲を固定できます。

```python
fig, sp = create_symple_plots(1, 1)

x = np.linspace(0.1, 100, 100)
y = 1 / x**2

sp.plot(
    x, y, 
    alab=["Time", "Decay"], 
    logy=True,          # Y軸を対数スケールに
    cx=[0, 50],         # X軸の範囲を 0~50 に固定
    cy=[1e-4, 1e2]      # Y軸の範囲を 10^-4~10^2 に固定
)

```

### 5. 画像プロット (Imshow) と 3D プロット

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

---

## ⚙️ パラメータ一覧 (Kwargs Reference)

`plot`, `scatter`, `pre_set`, `imshow`, `tdscatter`, `tdplot` メソッドで共通して使用できる主なキーワード引数(`**kwargs`)です。

| 引数名 | 型 | 説明 | 例 |
| --- | --- | --- | --- |
| `alab` | list | 軸ラベルを指定 `[xlabel, ylabel, (zlabel)]` | `alab=["Time (s)", "Voltage (V)"]` |
| `lab` | list/str | 凡例のテキスト | `lab=["Sample A", "Sample B"]` |
| `loc` | str | 凡例の位置（デフォルトはグラフの右外側） | `loc='upper left'` |
| `cx` | list | X軸の描画範囲を固定 `[xmin, xmax]` | `cx=[0, 100]` |
| `cy` | list | Y軸の描画範囲を固定 `[ymin, ymax]` | `cy=[-1.5, 1.5]` |
| `logx` | bool | X軸を対数スケールにする | `logx=True` |
| `logy` | bool | Y軸を対数スケールにする | `logy=True` |
| `nox` | bool | X軸の**数字（目盛りラベル）のみ**を非表示にする | `nox=True` |
| `noy` | bool | Y軸の**数字（目盛りラベル）のみ**を非表示にする | `noy=True` |
| `zoom` | str | プロットしたデータに合わせて枠を自動拡大する | `zoom='xy'` |
| `margin` | float | データ端から軸枠までの余白割合（デフォルト: 0.05） | `margin=0.1` |
| `marker` | list/str | [scatter専用] マーカーの形状 | `marker=['o', '^', 's']` |
| `size` | int | [scatter専用] マーカーのサイズ | `size=50` |
| `linestyle` | list/str | [plot専用] 線の種類 | `linestyle=['-', '--']` |
| `linewidth` | float | [plot専用] 線の太さ | `linewidth=2.5` |

### 🎨 カラーマップの設定 (`sp.col`)

インスタンスの `col` プロパティを変更することで、配色を一括で変更できます。データ数に応じて自動で色が分割されます。

```python
sp.col = 'grads'  # 気象分野特有のレインボーカラー（逆順）
sp.col = 'turbo'  # 視認性の高いGoogle開発のレインボーカラー（デフォルト）
sp.col = 'plasma' # 明るい暖色系のカラーマップ
sp.col = 'cool'   # シアン〜マゼンタの寒色系
sp.col = 'mode1'  # Matplotlibのデフォルトカラーサイクル
sp.col = ['red', 'blue'] # 手動でリスト指定も可能

```
