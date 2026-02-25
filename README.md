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
pip install git+[https://github.com/Chaim-Weizmann/symple_plot.git](https://github.com/Chaim-Weizmann/symple_plot.git)
