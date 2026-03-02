### 1. `generate_images.py` の完全版（比較パネル化・ズーム復活）

import os
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots, set_style, symple_plot

def main():
    os.makedirs('images', exist_ok=True)
    print("Generating example images for README...")

    # 0. 基本プロット
    fig0, sp0 = create_symple_plots(1, 1)
    x = np.linspace(0, 10, 50)
    sp0.plot([x, x], [np.sin(x), np.cos(x)], alab=["Time (s)", "Amplitude"], lab=["A", "B"], linestyle=['-', '--'])
    fig0.savefig("images/example0_basic.png", dpi=300, bbox_inches='tight')
    plt.close(fig0)

    # 1. 指数統一
    fig1, sp1 = create_symple_plots(1, 1)
    sp1.scatter(np.linspace(1, 5, 5), [5000, 10000, 15000, 20000, 25000], alab=["X", "Large Value"])
    fig1.savefig("images/example1_exponent.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. 軸の描画範囲の固定 (cx, cy) - 左:デフォルト, 右:適用
    fig2, sp_arr2 = create_symple_plots(1, 2, figsize=(12, 4))
    x2 = np.linspace(0, 10, 100)
    sp_arr2[0].plot(x2, np.sin(x2), alab=["X", "Y"])
    sp_arr2[0].ax.set_title("Default")
    sp_arr2[1].plot(x2, np.sin(x2), alab=["X (Limited)", "Y (Limited)"], cx=[2, 8], cy=[-0.8, 0.8])
    sp_arr2[1].ax.set_title("cx=[2, 8], cy=[-0.8, 0.8]")
    fig2.savefig("images/example2_range.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 3. 対数スケール (logx, logy) - 左:デフォルト, 右:適用
    fig3, sp_arr3 = create_symple_plots(1, 2, figsize=(12, 4))
    x3 = np.linspace(0.1, 10, 100)
    sp_arr3[0].plot(x3, 10**x3, alab=["X", "Y"])
    sp_arr3[0].ax.set_title("Default")
    sp_arr3[1].plot(x3, 10**x3, alab=["X", "Y (Log)"], logy=True)
    sp_arr3[1].ax.set_title("logy=True")
    fig3.savefig("images/example3_log.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 4. 目盛りの非表示 (nox, noy) - 左:デフォルト, 右:適用
    fig4, sp_arr4 = create_symple_plots(1, 2, figsize=(12, 4))
    sp_arr4[0].plot(x, np.sin(x), alab=["X", "Y"])
    sp_arr4[0].ax.set_title("Default")
    sp_arr4[1].plot(x, np.sin(x), alab=["X", "Y (Hidden Ticks)"], noy=True)
    sp_arr4[1].ax.set_title("noy=True")
    fig4.savefig("images/example4_noticks.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # 5. Inset Zoom
    fig5, sp5 = create_symple_plots(1, 1)
    y_zoom = np.sin(x) + 5 * np.exp(-((x - 7.5)**2) / 0.01)
    sp5.plot(x, y_zoom, alab=["X", "Intensity"])
    sp5.add_inset_zoom(xlim=[7.2, 7.8], bounds='upper left')
    fig5.savefig("images/example5_zoom.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # 6. 個別カラー指定と強制ズーム (復活！)
    fig6, sp_arr6 = create_symple_plots(2, 2, figsize=(10, 8))
    x_bg, y_bg = np.linspace(0, 20, 100), np.sin(np.linspace(0, 20, 100))
    x_target, y_target = np.linspace(5, 10, 50), np.sin(np.linspace(5, 10, 50))
    
    for i, title, zoom in zip(range(3), ["zoom='x'", "zoom='y'", "zoom='xy'"], ['x', 'y', 'xy']):
        sp = sp_arr6[i]
        sp.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
        sp.plot(x_target, y_target, col='red', lab=f"Target ({title})", zoom=zoom, linewidth=3)
        sp.ax.set_title(title, fontsize=14)

    sp6_4 = sp_arr6[3]
    sp6_4.plot(x_bg, y_bg, col='gray', lab="Full Data", alab=["X", "Y"])
    x_peak = np.linspace(7.2, 7.8, 50)
    y_peak = np.sin(x_peak) + 3 * np.exp(-((x_peak - 7.5)**2) / 0.01)
    sp6_4.plot(x_peak, y_peak, col='green', lab="Sharp Peak", zoomx=[7.2, 7.8])
    sp6_4.ax.set_title("Auto Inset Zoom (zoomx)", fontsize=14)
    fig6.savefig("images/example6_zoom_col.png", dpi=300, bbox_inches='tight')
    plt.close(fig6)

    # 7. 回帰と補助線
    fig7, sp7 = create_symple_plots(1, 2)
    x_reg = np.linspace(-5, 5, 30)
    sp7[0].scatter(x_reg, 0.5 * x_reg**3 - 2 * x_reg + np.random.normal(0, 5, 30), alab=["X", "Y"])
    sp7[0].Regression(regr=3)
    sp7[0].ax.set_title("Polynomial Regression (regr=3)")
    sp7[1].scatter(np.linspace(0.1, 5, 50), 2.5 * np.exp(-1.2 * np.linspace(0.1, 5, 50)) + np.random.normal(0, 0.05, 50), 
                   alab=["Time", "Intensity"], vx=[1, 3], vcol='red', vstyle='--', hy=0, hcol='blue', hstyle=':')
    sp7[1].Regression(regr=lambda x, a, b: a * np.exp(-b * x), auto_p0=True, bounds=([0, 0], [10, 5]))
    sp7[1].ax.set_title("Optuna Auto Fit & Guide Lines")
    fig7.savefig("images/example7_regression.png", dpi=300, bbox_inches='tight')
    plt.close(fig7)

    # 8. 3D & Imshow
    fig8, sp_arr8 = create_symple_plots(1, 2, figsize=(12, 5))
    sp_arr8[0].imshow([np.linspace(0, 5, 50)], [np.linspace(0, 5, 50)], np.random.rand(50, 50) * 1e-4, vmax=1e-4, alab=["X", "Y", "Int"])
    sp_arr8[1].ax.remove()
    sp8_2 = symple_plot(fig8.add_subplot(1, 2, 2, projection='3d'))
    sp8_2.tdscatter(np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100)), np.linspace(0, 10, 100), alab=["X", "Y", "Z"])
    fig8.savefig("images/example8_3d.png", dpi=300, bbox_inches='tight')
    plt.close(fig8)

    # 9. Style
    fig9, sp_arr9 = create_symple_plots(1, 2, figsize=(10, 4), style='slide', auto_label=True)
    sp_arr9[0].plot(np.linspace(0, 5, 20), np.exp(np.linspace(0, 5, 20)), alab=["Time", "Growth"])
    sp_arr9[1].scatter(np.linspace(0, 5, 20), np.linspace(0, 5, 20)**3, alab=["Time", "Value"], size=80, marker='s')
    fig9.savefig("images/example9_utils.png", dpi=300, bbox_inches='tight')
    plt.close(fig9)
    set_style('default')

    print("\n✅ All example images have been successfully generated!")

if __name__ == "__main__":
    main()