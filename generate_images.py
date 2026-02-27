import os
import numpy as np
import matplotlib.pyplot as plt

# ご自身のライブラリをインポート
from symple_plot import create_symple_plots, symple_plot, set_style

def main():
    # 画像保存用のフォルダを作成
    os.makedirs("images", exist_ok=True)
    print("画像生成を開始します...")

    # ==========================================
    # 1. 基本プロット (example1_basic.png)
    # ==========================================
    fig1, sp1 = create_symple_plots(1, 1, figsize=(6, 5))
    x1 = np.linspace(0, 10, 50)
    y1_1 = np.sin(x1)
    y1_2 = np.cos(x1)

    sp1.plot(
        [x1, x1], [y1_1, y1_2],
        alab=["Time (s)", "Amplitude (a.u.)"],
        lab=["Sin Curve", "Cos Curve"],
        linestyle=['-', '--'],
        linewidth=2
    )
    sp1.ax.set_title("Basic Usage", fontsize=14)
    fig1.savefig("images/example1_basic.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(" - example1_basic.png を作成しました")

    # ==========================================
    # 2. 指数統一プロット (example2_exponent.png)
    # ==========================================
    fig2, sp2 = create_symple_plots(1, 1, figsize=(6, 5))
    x2 = np.linspace(1, 5, 5)
    y2 = np.array([5000, 10000, 15000, 20000, 25000])

    sp2.scatter(x2, y2, alab=["X", "Large Value"], size=80, marker='D')
    sp2.ax.set_title("Auto Smart Formatter", fontsize=14)
    fig2.savefig("images/example2_exponent.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(" - example2_exponent.png を作成しました")

    # ==========================================
    # 3. Inset Zoomプロット (example3_zoom.png)
    # ==========================================
    fig3, sp3 = create_symple_plots(1, 1, figsize=(6, 5))
    x3 = np.linspace(0, 10, 500)
    y3 = np.sin(x3) + 5 * np.exp(-((x3 - 7.5)**2) / 0.01)

    sp3.plot(x3, y3, alab=["X", "Intensity"], lab="Signal with sharp peak")
    
    # 🌟 新機能の反映: boundsを指定せず、'auto' で最適な位置に自動配置させます
    sp3.add_inset_zoom(xlim=[7.2, 7.8])
    sp3.ax.set_title("Inset Zoom (Auto Bounds)", fontsize=14)
    
    fig3.savefig("images/example3_zoom.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(" - example3_zoom.png を作成しました")

    # ==========================================
    # 4. 回帰分析プロット (example4_regression.png)
    # ==========================================
    fig4, sp4 = create_symple_plots(1, 1, figsize=(6, 5))
    np.random.seed(42) # 画像を毎回同じにするためシード固定
    x4 = np.linspace(-5, 5, 30)
    y4 = 0.5 * x4**3 - 2 * x4 + np.random.normal(0, 5, 30)

    sp4.scatter(x4, y4, alab=["X", "Y"], lab="Measured Data", size=50)
    sp4.Regression(regr=3, directory='./') # 回帰線の追加
    sp4.ax.set_title("Polynomial Regression (3rd degree)", fontsize=14)
    
    fig4.savefig("images/example4_regression.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    print(" - example4_regression.png を作成しました")

    # ==========================================
    # 5. Imshow と 3D プロット (example5_3d.png)
    # ==========================================
    fig5, sp_arr = create_symple_plots(1, 2, figsize=(12, 5))

    # 左パネル: Imshow
    sp5_1 = sp_arr[0]
    z_im = (np.sin(np.linspace(0, 5, 50)[:, None] * 2) * np.cos(np.linspace(0, 5, 50)[None, :] * 2) + 1) * 1e-4
    sp5_1.imshow(
        [np.linspace(0, 5, 50)], [np.linspace(0, 5, 50)], z_im,
        vmax=2e-4, alab=["X ($\\mu$m)", "Y ($\\mu$m)", "Intensity"]
    )
    sp5_1.ax.set_title("2D Mapping (imshow)", fontsize=14)

    # 右パネル: 3D Scatter
    sp_arr[1].ax.remove()
    ax_3d = fig5.add_subplot(1, 2, 2, projection='3d')
    sp5_2 = symple_plot(ax_3d)
    sp5_2.col = 'plasma'

    z_3d = np.linspace(0, 10, 100)
    sp5_2.tdscatter(
        np.sin(z_3d)*z_3d, np.cos(z_3d)*z_3d, z_3d,
        alab=["X", "Y", "Z"], size=30, lab="3D Spiral"
    )
    ax_3d.set_title("3D Scatter", fontsize=14)

    fig5.savefig("images/example5_3d.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig5)
    print(" - example5_3d.png を作成しました")

    # ==========================================
    # 🌟 6. 論文・プレゼン用ユーティリティ (example6_utils.png) 🌟
    # ==========================================
    # style='slide' と auto_label=True を引数で渡すだけ！
    fig6, sp_arr6 = create_symple_plots(1, 2, figsize=(10, 4), style='slide', auto_label=True)

    x6 = np.linspace(0, 5, 20)
    sp_arr6[0].plot(x6, np.exp(x6), alab=["Time", "Growth"], lab="Exponential")
    sp_arr6[1].scatter(x6, x6**3, alab=["Time", "Value"], size=80, marker='s', lab="Quadratic")

    fig6.savefig("images/example6_utils.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig6)
    print(" - example6_utils.png を作成しました")
    
    # 次の描画に影響が出ないよう、スタイルをデフォルトに戻す
    set_style('default')

    # ==========================================
    # 🌟 7. 個別カラー指定と強制ズーム (example7_zoom_col.png) 🌟
    # ==========================================
    # 1行2列のパネルで2種類のズーム機能をデモ
    fig7, sp_arr7 = create_symple_plots(2, 2)
    
    x_bg = np.linspace(0, 20, 100)
    y_bg = np.sin(x_bg)
    
    # --- 左パネル: `zoom='x'` のテスト（Y軸は維持し、X軸だけ上書きズーム） ---
    sp7_1 = sp_arr7[0]
    sp7_2 = sp_arr7[1]
    sp7_3 = sp_arr7[2]
    sp7_1.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
    sp7_2.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
    sp7_3.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
    
    x_target = np.linspace(5, 10, 50)
    y_target = np.sin(x_target)
    # zoom='x' を指定すると、Y軸の高さ(±1)は保ったまま、X軸だけが 5〜10 にズームされる
    sp7_1.plot(x_target, y_target, col='red', lab="Target (zoom='x')", zoom='x', linewidth=3)
    sp7_2.plot(x_target, y_target, col='red', lab="Target (zoom='y')", zoom='y', linewidth=3)
    sp7_3.plot(x_target, y_target, col='red', lab="Target (zoom='both')", zoom='xy', linewidth=3)
    sp7_1.ax.set_title("zoom='x' (Override X-axis)", fontsize=14)
    sp7_2.ax.set_title("zoom='y' (Override Y-axis)", fontsize=14)
    sp7_3.ax.set_title("zoom='xy' (Override Both)", fontsize=14)

    # --- 右パネル: `zoomx` のテスト（プロットと同時に拡大小窓を自動生成） ---
    sp7_4 = sp_arr7[3]
    sp7_4.plot(x_bg, y_bg, col='gray', lab="Full Data", alab=["X", "Y"])
    
    # zoomx=[7.2, 7.8] を引数に入れるだけで、勝手に add_inset_zoom が発動する！
    x_peak = np.linspace(7.2, 7.8, 50)
    y_peak = np.sin(x_peak) + 3 * np.exp(-((x_peak - 7.5)**2) / 0.01)
    sp7_4.plot(x_peak, y_peak, col='green', lab="Sharp Peak", zoomx=[7.2, 7.8])
    sp7_4.ax.set_title("Auto Inset Zoom (zoomx)", fontsize=14)
    
    fig7.savefig("images/example7_zoom_col.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig7)
    print(" - example7_zoom_col.png を作成しました")

    print("すべての画像生成が完了しました！ 'images' フォルダを確認してください。")

if __name__ == "__main__":
    main()