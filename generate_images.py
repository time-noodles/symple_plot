import os
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots, set_style, symple_plot, del_file

del_file('*.csv')
del_file('./images/*.png')

def main():
    os.makedirs('images', exist_ok=True)
    print("Generating example images for README...")

    # 0. 基本プロット (Matplotlib vs symple_plot)
    fig0 = plt.figure(figsize=(12, 5))
    x = np.linspace(0, 10, 50)
    y1, y2 = np.sin(x), np.cos(x)
    
    # 左: Matplotlib Default
    ax0_mpl = fig0.add_subplot(121)
    ax0_mpl.plot(x, y1, label="Sample A")
    ax0_mpl.plot(x, y2, label="Sample B")
    ax0_mpl.set_xlabel("Time (s)")
    ax0_mpl.set_ylabel("Amplitude (a.u.)")
    ax0_mpl.legend()
    ax0_mpl.set_title("Matplotlib Default")

    # 右: symple_plot
    ax0_sp = fig0.add_subplot(122)
    sp0 = symple_plot(ax0_sp)
    sp0.plot([x, x], [y1, y2], alab=["Time (s)", "Amplitude (a.u.)"], lab=["Sample A", "Sample B"])
    ax0_sp.set_title("symple_plot")
    
    fig0.savefig("images/example0_basic.png", dpi=300, bbox_inches='tight')
    plt.close(fig0)

    # 1. 指数統一 (Matplotlib vs symple_plot)
    fig1 = plt.figure(figsize=(12, 5))
    x_exp = np.linspace(1, 5, 5)
    y_exp = np.array([5000, 10000, 15000, 20000, 25000])

    # 左: Matplotlib Default
    ax1_mpl = fig1.add_subplot(121)
    ax1_mpl.scatter(x_exp, y_exp)
    ax1_mpl.set_xlabel("X")
    ax1_mpl.set_ylabel("Large Value")
    ax1_mpl.set_title("Matplotlib Default")

    # 右: symple_plot
    ax1_sp = fig1.add_subplot(122)
    sp1 = symple_plot(ax1_sp)
    sp1.scatter(x_exp, y_exp, alab=["X", "Large Value"])
    ax1_sp.set_title("symple_plot")
    
    fig1.savefig("images/example1_exponent.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 2. 軸の描画範囲の固定 (cx, cy)
    fig2, sp_arr2 = create_symple_plots(1, 2, figsize=(12, 4))
    x2 = np.linspace(0, 10, 100)
    sp_arr2[0].plot(x2, np.sin(x2), alab=["X", "Y"])
    sp_arr2[0].ax.set_title("Default")
    sp_arr2[1].plot(x2, np.sin(x2), alab=["X (Limited)", "Y (Limited)"], cx=[2, 8], cy=[-0.8, 0.8])
    sp_arr2[1].ax.set_title("cx=[2, 8], cy=[-0.8, 0.8]")
    fig2.savefig("images/example2_range.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 3. 対数スケール (logx, logy)
    fig3, sp_arr3 = create_symple_plots(1, 2, figsize=(12, 4))
    x3 = np.linspace(0.1, 10, 100)
    sp_arr3[0].plot(x3, 10**x3, alab=["X", "Y"])
    sp_arr3[0].ax.set_title("Default")
    sp_arr3[1].plot(x3, 10**x3, alab=["X", "Y (Log)"], logy=True)
    sp_arr3[1].ax.set_title("logy=True")
    fig3.savefig("images/example3_log.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 4. 目盛りの非表示 (nox, noy)
    fig4, sp_arr4 = create_symple_plots(1, 2, figsize=(12, 4))
    x4 = np.linspace(0, 10, 100)
    sp_arr4[0].plot(x4, np.sin(x4), alab=["X", "Y"])
    sp_arr4[0].ax.set_title("Default")
    sp_arr4[1].plot(x4, np.sin(x4), alab=["X", "Y (Hidden Ticks)"], noy=True)
    sp_arr4[1].ax.set_title("noy=True")
    fig4.savefig("images/example4_noticks.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # 5. Inset Zoom (1:Original, 2:zoomx, 3:add_inset_zoom)
    fig5, sp_arr5 = create_symple_plots(1, 3, figsize=(15, 5))
    x_bg = np.linspace(0, 20, 200)
    y_bg = np.sin(x_bg)
    x_peak = np.linspace(7.2, 7.8, 50)
    y_peak = np.sin(x_peak) + 3 * np.exp(-((x_peak - 7.5)**2) / 0.01)

    # 結合してソート (左のプロット用)
    x_all = np.concatenate([x_bg, x_peak])
    y_all = np.concatenate([y_bg, y_peak])
    sort_idx = np.argsort(x_all)
    x_all, y_all = x_all[sort_idx], y_all[sort_idx]

    # 1. Original
    sp_arr5[0].plot(x_all, y_all, alab=["X", "Intensity"], col='gray')
    sp_arr5[0].ax.set_title("1. Original")

    # 2. zoomx=[] で指定
    sp_arr5[1].plot(x_bg, y_bg, col='gray', alab=["X", "Intensity"])
    sp_arr5[1].plot(x_peak, y_peak, col='green', zoomx=[7.2, 7.8])
    sp_arr5[1].ax.set_title("2. zoomx=[7.2, 7.8]")

    # 3. add_inset_zoom で指定
    sp_arr5[2].plot(x_bg, y_bg, col='gray', alab=["X", "Intensity"])
    sp_arr5[2].plot(x_peak, y_peak, col='green')
    sp_arr5[2].add_inset_zoom(xlim=[7.2, 7.8], bounds='upper left')
    sp_arr5[2].ax.set_title("3. add_inset_zoom()")
    
    fig5.savefig("images/example5_zoom.png", dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # 6. 個別カラー指定と強制ズーム
    fig6, sp_arr6 = create_symple_plots(1, 3, figsize=(15, 4))
    x_bg, y_bg = np.linspace(0, 20, 100), np.sin(np.linspace(0, 20, 100))
    x_target, y_target = np.linspace(5, 10, 50), np.sin(np.linspace(5, 10, 50))
    
    for i, title, zoom in zip(range(3), ["zoom='x'", "zoom='y'", "zoom='xy'"], ['x', 'y', 'xy']):
        sp = sp_arr6[i]
        sp.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
        sp.plot(x_target, y_target, col='red', lab=f"Target", zoom=zoom, linewidth=3)
        sp.ax.set_title(title, fontsize=14)

    fig6.savefig("images/example6_zoom_col.png", dpi=300, bbox_inches='tight')
    plt.close(fig6)

    # 7. 回帰と補助線
    fig7, sp7 = create_symple_plots(1, 2, figsize=(12, 5))
    x_reg = np.linspace(-5, 5, 30)
    sp7[0].scatter(x_reg, 0.5 * x_reg**3 - 2 * x_reg + np.random.normal(0, 5, 30), alab=["X", "Y"])
    sp7[0].Regression(regr=3)
    sp7[0].ax.set_title("Polynomial Regression (regr=3)")
    sp7[1].scatter(np.linspace(0.1, 5, 50), 2.5 * np.exp(-1.2 * np.linspace(0.1, 5, 50)) + np.random.normal(0, 0.05, 50), 
                   alab=["Time (s)", "Intensity"], vx=[1, 3], vcol='red', vstyle='--', hy=0, hcol='blue', hstyle=':')
    sp7[1].Regression(regr=lambda x, a, b: a * np.exp(-b * x), auto_p0=True, bounds=([0, 0], [10, 5]))
    sp7[1].ax.set_title("Global Auto Fit & Guide Lines")
    fig7.savefig("images/example7_regression.png", dpi=300, bbox_inches='tight')
    plt.close(fig7)

    # 8. 3D & Imshow
    fig8, sp_arr8 = create_symple_plots(1, 2, figsize=(12, 5))
    sp_arr8[0].imshow([np.linspace(0, 5, 50)], [np.linspace(0, 5, 50)], np.random.rand(50, 50) * 1e-4, vmax=1e-4, alab=["X (um)", "Y (um)", "Intensity"])
    sp_arr8[1].ax.remove()
    sp8_2 = symple_plot(fig8.add_subplot(1, 2, 2, projection='3d'))
    sp8_2.tdscatter(np.sin(np.linspace(0, 10, 100)), np.cos(np.linspace(0, 10, 100)), np.linspace(0, 10, 100), alab=["X", "Y", "Z"])
    fig8.savefig("images/example8_3d.png", dpi=300, bbox_inches='tight')
    plt.close(fig8)

    # 9. Style
    fig9, sp_arr9 = create_symple_plots(1, 2, figsize=(10, 4), style='slide', auto_label=True)
    sp_arr9[0].plot(np.linspace(0, 5, 20), np.exp(np.linspace(0, 5, 20)), alab=["Time", "Growth"], lab="Exponential")
    sp_arr9[1].scatter(np.linspace(0, 5, 20), np.linspace(0, 5, 20)**3, alab=["Time", "Value"], size=80, marker='s', lab="Quadratic")
    fig9.savefig("images/example9_utils.png", dpi=300, bbox_inches='tight')
    plt.close(fig9)
    
    # 10. インラインラベル (Inline Labels) - 比較パネル版
    def logistic(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))    
    
    fig10, sp_arr10 = create_symple_plots(1, 2, figsize=(12, 5))
    x10 = np.linspace(0, 20, 100)
    y1 = logistic(x10, 10, 0.8, 10)
    y2 = logistic(x10, 8, 0.5, 12) + 1.5
    
    sp_arr10[0].plot([x10, x10], [y1, y2], 
                     alab=["Time (days)", "Growth Yield"], 
                     lab=["Sample A", "Sample B"],
                     col=["magenta", "darkviolet"])
    sp_arr10[0].ax.set_title("symple_plot Default (Legend)")

    sp_arr10[1].plot([x10, x10], [y1, y2], 
                     alab=["Time (days)", "Growth Yield"], 
                     lab=["Sample A", "Sample B"], 
                     loc='inline',
                     lab_fs=13,               
                     inline_dy=[0.4, -0.4],   
                     inline_pad=0.08,         
                     col=["magenta", "darkviolet"])
    sp_arr10[1].ax.set_title("symple_plot (Inline Labels)")
    fig10.savefig("images/example10_inline.png", dpi=300, bbox_inches='tight')
    plt.close(fig10)

    # 11. グリッド共有軸
    fig11, sp_arr11 = create_symple_plots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    x11 = np.linspace(0, 10, 100)
    for i in range(4):
        freq = i + 1
        alab_11 = ["Time (s)", "Signal"] if i >= 2 else ["", "Signal"]
        sp_arr11[i].plot(x11, np.sin(freq * x11), col='blue', alab=alab_11)
        sp_arr11[i].ax.set_title(f"Frequency {freq}Hz")
    fig11.savefig("images/example11_shared_axes.png", dpi=300, bbox_inches='tight')
    plt.close(fig11)

    # 12. 第二軸と変換
    fig12, sp_arr12 = create_symple_plots(1, 2, figsize=(14, 5))

    # パネル1: Twin Axes (第二Y軸)
    sp12_left = sp_arr12[0]
    x12_a = np.linspace(0.1, 10, 50)
    sp12_left.plot(x12_a, x12_a**2, col='blue', alab=["Time (s)", "Linear Scale"])
    sp12_right = sp12_left.twinx(col='red', alab="Log Scale")
    sp12_right.plot(x12_a, np.exp(x12_a), logy=True)
    sp12_left.ax.set_title("Twin Axes (Secondary Y)")

    # パネル2: Secondary Xaxis (スケール変換)
    sp12_bottom = sp_arr12[1]
    T_celsius = np.linspace(0, 100, 50)
    sp12_bottom.plot(T_celsius, np.sqrt(T_celsius), alab=["Temperature (°C)", "Value"])
    
    def C_to_F(c): return c * 1.8 + 32
    def F_to_C(f): return (f - 32) / 1.8
    sp12_bottom.secondary_xaxis(location='top', functions=(C_to_F, F_to_C), alab="Temperature (°F)")
    sp12_bottom.ax.set_title("Secondary Xaxis (Scale Conv.)")
    
    fig12.savefig("images/example12_twin_axes.png", dpi=300, bbox_inches='tight')
    plt.close(fig12)

    # ==========================
    # 🌟 NEW 13. 隙間なしグリッド (Flush Grid) 🌟
    # ==========================
    # flush=True を指定するだけで wspace=0, hspace=0 と sharex/y が適用されます
    fig13, sp_arr13 = create_symple_plots(3, 3, figsize=(6, 6), flush=True)
    x13 = np.linspace(-6, 6, 20)
    
    for i in range(3):      # 行
        for j in range(3):  # 列
            idx = i * 3 + j
            y13 = np.sin(x13) * (3 - i) + 5
            
            # 左端と下端のパネルのみに軸ラベルを設定するロジック
            is_left = (j == 0)
            is_bottom = (i == 2)
            alab_x = "x-label" if is_bottom else ""
            alab_y = "y-label" if is_left else ""
            
            sp = sp_arr13[idx]
            sp.plot(x13, y13, marker='.', markersize=8, alab=[alab_x, alab_y])
            
            # (行, 列) のテキストを左上に赤文字で追加
            sp.ax.text(0.05, 0.95, f"({i},{j})", transform=sp.ax.transAxes, 
                       color='red', fontsize=12, va='top', ha='left')

    fig13.savefig("images/example13_flush_grid.png", dpi=300, bbox_inches='tight')
    plt.close(fig13)

    set_style('default')
    print("\n✅ All example images generated (including Twin Axes & Shared Grid).")

if __name__ == "__main__":
    main()