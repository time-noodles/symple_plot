import os
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots, set_style, symple_plot, del_file

del_file('*.csv')
del_file('./images/*.png')

def main():
    os.makedirs('images', exist_ok=True)
    print("Generating example images...")

    # ==========================
    # 🌟 README掲載用: 基本メソッド 🌟
    # ==========================
    
    # 01. クイックスタート (Quickstart)
    fig0 = plt.figure(figsize=(12, 5))
    x = np.linspace(0, 10, 50)
    y1, y2 = np.sin(x), np.cos(x)
    ax0_mpl = fig0.add_subplot(121)
    ax0_mpl.plot(x, y1, label="Sample A")
    ax0_mpl.plot(x, y2, label="Sample B")
    ax0_mpl.set_xlabel("Time (s)")
    ax0_mpl.set_ylabel("Amplitude (a.u.)")
    ax0_mpl.legend()
    ax0_mpl.set_title("Matplotlib Default")
    ax0_sp = fig0.add_subplot(122)
    sp0 = symple_plot(ax0_sp)
    sp0.plot([x, x], [y1, y2], alab=["Time (s)", "Amplitude (a.u.)"], lab=["Sample A", "Sample B"])
    ax0_sp.set_title("symple_plot")
    fig0.savefig("images/01_quickstart.png", dpi=300, bbox_inches='tight')
    plt.close(fig0)

    # 02. create_symple_plots の例
    fig02, sp_arr02 = create_symple_plots(nrows=1, ncols=2, figsize=(10, 4))
    sp_arr02[0].plot(x, y1, alab=["X", "Y1"])
    sp_arr02[1].plot(x, y2, alab=["X", "Y2"], col='red')
    fig02.savefig("images/02_create_symple_plots.png", dpi=300, bbox_inches='tight')
    plt.close(fig02)

    # 03. コアメソッド (plot, scatter, hollow)
    fig_core, sp_arr_core = create_symple_plots(1, 2, figsize=(12, 5))
    x_core = np.linspace(0, 10, 20)
    sp_arr_core[0].plot([x_core, x_core], [np.sin(x_core), np.cos(x_core)], alab=["X", "Y"], lab=["Line 1", "Line 2"], linestyle=['-', '--'])
    sp_arr_core[0].ax.set_title("sp.plot()")
    sp_arr_core[1].scatter(x_core, np.sin(x_core), alab=["X", "Y"], lab="Filled", marker='o', size=80, col='blue')
    sp_arr_core[1].scatter(x_core, np.cos(x_core), lab="Hollow", marker='s', size=80, hollow=True, linewidth=2.0, col='red')
    sp_arr_core[1].ax.set_title("sp.scatter()")
    fig_core.savefig("images/03_core_methods.png", dpi=300, bbox_inches='tight')
    plt.close(fig_core)

    # 04. コアメソッド (imshow, tdscatter)
    fig8, sp_arr8 = create_symple_plots(1, 2, figsize=(12, 5))
    sp_arr8[0].imshow([np.linspace(0, 5, 50)], [np.linspace(0, 5, 50)], np.random.rand(50, 50) * 1e-4, vmax=1e-4, alab=["X", "Y", "Intensity"])
    sp_arr8[0].ax.set_title("sp.imshow()")
    sp_arr8[1].ax.remove()
    sp8_2 = symple_plot(fig8.add_subplot(1, 2, 2, projection='3d'))
    sp8_2.tdscatter(np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100)), np.linspace(0, 10, 100), alab=["X", "Y", "Z"])
    sp8_2.ax.set_title("sp.tdscatter()")
    fig8.savefig("images/04_core_3d_imshow.png", dpi=300, bbox_inches='tight')
    plt.close(fig8)

    # ==========================
    # 🌟 Wiki掲載用: 応用ギャラリー (10〜) 🌟
    # ==========================

    # 10. 指数統一 (1-1) - 🌟 1x2 の巨大な値と微小な値の比較に変更
    fig1, sp_arr1 = create_symple_plots(1, 2, figsize=(12, 5))
    x_exp = np.linspace(1, 5, 5)
    y_large = np.array([5000, 10000, 15000, 20000, 25000])
    y_small = np.array([0.0005, 0.0010, 0.0015, 0.0020, 0.0025])
    sp_arr1[0].scatter(x_exp, y_large, alab=["X", "Large Value"], size=60)
    sp_arr1[0].ax.set_title("Large Values")
    sp_arr1[1].scatter(x_exp, y_small, alab=["X", "Small Value"], col='red', size=60)
    sp_arr1[1].ax.set_title("Small Values")
    fig1.savefig("images/10_exponent.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 11. 描画範囲固定 (1-2)
    fig2, sp_arr2 = create_symple_plots(1, 2, figsize=(12, 4))
    x2 = np.linspace(0, 10, 100)
    sp_arr2[0].plot(x2, np.sin(x2), alab=["X", "Y"])
    sp_arr2[0].ax.set_title("Default")
    sp_arr2[1].plot(x2, np.sin(x2), alab=["X (Limited)", "Y (Limited)"], cx=[2, 8], cy=[-0.8, 0.8])
    sp_arr2[1].ax.set_title("cx=[2, 8], cy=[-0.8, 0.8]")
    fig2.savefig("images/11_range.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 12. 目盛り非表示 (1-3)
    fig4, sp_arr4 = create_symple_plots(1, 2, figsize=(12, 4))
    x4 = np.linspace(0, 10, 100)
    sp_arr4[0].plot(x4, np.sin(x4), alab=["X", "Y"])
    sp_arr4[0].ax.set_title("Default")
    sp_arr4[1].plot(x4, np.sin(x4), alab=["X", "Y (Hidden Ticks)"], noy=True)
    sp_arr4[1].ax.set_title("noy=True")

    fig4.savefig("images/12_noticks.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # 🌟 追加: 12b. 枠の事前生成とネイティブ関数との連携 (pre_set) (1-4) 🌟
    fig4b, sp_arr4b = create_symple_plots(1, 2, figsize=(12, 4))
    x4b = np.linspace(0, 10, 100)
    y_mean = np.sin(x4b)
    y_err = 0.2 * np.ones_like(x4b)
    
    # 左パネル: pre_setを使って枠を設定し、fill_betweenで描画
    ax4b = sp_arr4b[0].pre_set(x4b, y_mean, alab=["X", "Y (with Error)"], cx=[2, 8])
    ax4b.plot(x4b, y_mean, color='blue')
    ax4b.fill_between(x4b, y_mean - y_err, y_mean + y_err, color='blue', alpha=0.3)
    sp_arr4b[0].ax.set_title("pre_set + ax.fill_between")
    
    # 右パネル: 比較用の通常プロット
    sp_arr4b[1].plot(x4b, y_mean, alab=["X", "Y (Normal)"], cx=[2, 8])
    sp_arr4b[1].ax.set_title("Standard plot")
    
    fig4b.savefig("images/12b_preset.png", dpi=300, bbox_inches='tight')
    plt.close(fig4b)

    # 13. グリッド共有軸 (2-1)
    fig11, sp_arr11 = create_symple_plots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    x11 = np.linspace(0, 10, 100)
    for i in range(4):
        freq = i + 1
        sp_arr11[i].plot(x11, np.sin(freq * x11), col='blue')
        sp_arr11[i].ax.set_title(f"Frequency {freq}Hz")
    fig11.savefig("images/13_shared_axes.png", dpi=300, bbox_inches='tight')
    plt.close(fig11)

    # 14. 隙間なしグリッド (Flush Grid) (2-2)
    fig13, sp_arr13 = create_symple_plots(3, 3, figsize=(6, 6), flush=True)
    x13 = np.linspace(-6, 6, 20)
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            y13 = np.sin(x13) * (3 - i) + 5
            alab_x = "x-label" if i == 2 else ""
            alab_y = "y-label" if j == 0 else ""
            sp = sp_arr13[idx]
            sp.plot(x13, y13, marker='.', alab=[alab_x, alab_y])
            sp.ax.text(0.05, 0.95, f"({i},{j})", transform=sp.ax.transAxes, color='red', fontsize=12, va='top', ha='left')
    fig13.savefig("images/14_flush_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig13)

    # 15. 第二軸と変換 (2-3)
    fig12, sp_arr12 = create_symple_plots(1, 2, figsize=(14, 5))
    sp12_left = sp_arr12[0]
    x12_a = np.linspace(0.1, 10, 50)
    sp12_left.plot(x12_a, x12_a**2, col='blue', alab=["Time (s)", "Linear Scale"])
    sp12_right = sp12_left.twinx(col='red', alab="Log Scale")
    sp12_right.plot(x12_a, np.exp(x12_a), logy=True)
    sp12_left.ax.set_title("Twin Axes (Secondary Y)")
    
    sp12_bottom = sp_arr12[1]
    T_celsius = np.linspace(0, 100, 50)
    sp12_bottom.plot(T_celsius, np.sqrt(T_celsius), alab=["Temperature (°C)", "Value"])
    sp12_bottom.secondary_xaxis(lambda c: c * 1.8 + 32, location='top', alab="Temperature (°F)")
    sp12_bottom.ax.set_title("Secondary Xaxis (Scale Conv.)")
    fig12.savefig("images/15_twin_axes.png", dpi=300, bbox_inches='tight')
    plt.close(fig12)

    # 16. Inset Zoom (cx引数を使用) (3-1)
    fig5, sp_arr5 = create_symple_plots(1, 3, figsize=(15, 5))
    x_bg = np.linspace(0, 20, 200)
    y_bg = np.sin(x_bg)
    x_peak = np.linspace(7.2, 7.8, 50)
    y_peak = np.sin(x_peak) + 3 * np.exp(-((x_peak - 7.5)**2) / 0.01)
    x_all = np.concatenate([x_bg, x_peak])
    y_all = np.concatenate([y_bg, y_peak])
    sort_idx = np.argsort(x_all)
    x_all, y_all = x_all[sort_idx], y_all[sort_idx]
    
    sp_arr5[0].plot(x_all, y_all, alab=["X", "Intensity"], col='gray')
    sp_arr5[0].ax.set_title("1. Original")
    
    sp_arr5[1].plot(x_bg, y_bg, col='gray', alab=["X", "Intensity"])
    sp_arr5[1].plot(x_peak, y_peak, col='green', zoomx=[7.2, 7.8])
    sp_arr5[1].ax.set_title("2. zoomx=[7.2, 7.8]")
    
    sp_arr5[2].plot(x_bg, y_bg, col='gray', alab=["X", "Intensity"])
    sp_arr5[2].plot(x_peak, y_peak, col='green')
    sp_arr5[2].add_inset_zoom(cx=[7.2, 7.8], bounds='upper left', noy=True)
    sp_arr5[2].ax.set_title("3. add_inset_zoom(cx=...)")
    
    fig5.savefig("images/16_zoom.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # 17. 個別カラー指定と強制ズーム (3-2)
    fig6, sp_arr6 = create_symple_plots(1, 3, figsize=(15, 4))
    x_target, y_target = np.linspace(5, 10, 50), np.sin(np.linspace(5, 10, 50))
    for i, title, zoom in zip(range(3), ["zoom='x'", "zoom='y'", "zoom='xy'"], ['x', 'y', 'xy']):
        sp = sp_arr6[i]
        sp.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
        sp.plot(x_target, y_target, col='red', lab=f"Target", zoom=zoom, linewidth=3)
        sp.ax.set_title(title, fontsize=14)
    fig6.savefig("images/17_zoom_col.png", dpi=300, bbox_inches='tight')
    plt.close(fig6)

    # 18. インラインラベル (3-3)
    def logistic(x, L, k, x0): return L / (1 + np.exp(-k * (x - x0)))    
    fig10, sp_arr10 = create_symple_plots(1, 2, figsize=(12, 5))
    x10 = np.linspace(0, 20, 100)
    y10_1, y10_2 = logistic(x10, 10, 0.8, 10), logistic(x10, 8, 0.5, 12) + 1.5
    sp_arr10[0].plot([x10, x10], [y10_1, y10_2], alab=["Time (days)", "Growth Yield"], lab=["Sample A", "Sample B"], col=["magenta", "darkviolet"])
    sp_arr10[0].ax.set_title("symple_plot Default (Legend)")
    sp_arr10[1].plot([x10, x10], [y10_1, y10_2], alab=["Time (days)", "Growth Yield"], lab=["Sample A", "Sample B"], loc='inline', lab_fs=13, inline_dy=[0.4, -0.4], inline_pad=0.08, col=["magenta", "darkviolet"])
    sp_arr10[1].ax.set_title("symple_plot (Inline Labels)")
    fig10.savefig("images/18_inline.png", dpi=300, bbox_inches='tight')
    plt.close(fig10)

    # 19. Styleと自動ラベル (3-4)
    fig9, sp_arr9 = create_symple_plots(1, 2, figsize=(10, 4), style='slide', auto_label=True)
    sp_arr9[0].plot(np.linspace(0, 5, 20), np.exp(np.linspace(0, 5, 20)), alab=["Time", "Growth"], lab="Exponential")
    sp_arr9[1].scatter(np.linspace(0, 5, 20), np.linspace(0, 5, 20)**3, alab=["Time", "Value"], size=80, marker='s', lab="Quadratic")
    fig9.savefig("images/19_style_labels.png", dpi=300, bbox_inches='tight')
    plt.close(fig9)

    # 20. 回帰と補助線 (4-1)
    fig7, sp7 = create_symple_plots(1, 2, figsize=(12, 5))
    x_reg = np.linspace(-5, 5, 30)
    sp7[0].scatter(x_reg, 0.5 * x_reg**3 - 2 * x_reg + np.random.normal(0, 5, 30), alab=["X", "Y"])
    sp7[0].Regression(regr=3)
    sp7[0].ax.set_title("Polynomial Regression (regr=3)")
    sp7[1].scatter(np.linspace(0.1, 5, 50), 2.5 * np.exp(-1.2 * np.linspace(0.1, 5, 50)) + np.random.normal(0, 0.05, 50), 
                   alab=["Time (s)", "Intensity"], vx=[1, 3], vcol='red', vstyle='--', hy=0, hcol='blue', hstyle=':')
    sp7[1].Regression(regr=lambda x, a, b: a * np.exp(-b * x), auto_p0=True, bounds=([0, 0], [10, 5]))
    sp7[1].ax.set_title("Global Auto Fit & Guide Lines")
    fig7.savefig("images/20_regression.png", dpi=300, bbox_inches='tight')
    plt.close(fig7)

    set_style('default')
    print("\n✅ All example images generated successfully.")

if __name__ == "__main__":
    main()