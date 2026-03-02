import os
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots, set_style, symple_plot

def main():
    os.makedirs('images', exist_ok=True)
    print("Generating example images for README...")

    # ==========================================
    # 🌟 Basic Usage (example0_basic.png)
    # ==========================================
    fig0, sp0 = create_symple_plots(nrows=1, ncols=1)
    x0 = np.linspace(0, 10, 50)
    sp0.plot([x0, x0], [np.sin(x0), np.cos(x0)], 
             alab=["Time (s)", "Amplitude (a.u.)"], 
             lab=["Sample A", "Sample B"], 
             linestyle=['-', '--'], linewidth=2)
    fig0.savefig("images/example0_basic.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig0)
    print(" - example0_basic.png を作成しました")

    # ==========================================
    # 🌟 1. 指数の自動統一 (example1_exponent.png)
    # ==========================================
    fig1, sp1 = create_symple_plots(1, 1)
    x1 = np.linspace(1, 5, 5)
    y1 = np.array([5000, 10000, 15000, 20000, 25000])
    sp1.scatter(x1, y1, alab=["X", "Large Value"])
    fig1.savefig("images/example1_exponent.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    print(" - example1_exponent.png を作成しました")

    # ==========================================
    # 🌟 2. Inset Zoom (example2_zoom.png)
    # ==========================================
    fig2, sp2 = create_symple_plots(1, 1)
    x2 = np.linspace(0, 10, 500)
    y2 = np.sin(x2) + 5 * np.exp(-((x2 - 7.5)**2) / 0.01)
    sp2.plot(x2, y2, alab=["X", "Intensity"])
    sp2.add_inset_zoom(xlim=[7.2, 7.8], bounds='upper left')
    fig2.savefig("images/example2_zoom.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(" - example2_zoom.png を作成しました")

    # ==========================================
    # 🌟 3. 回帰分析と補助線 (example3_regression.png)
    # ==========================================
    fig3, sp_arr3 = create_symple_plots(1, 2, figsize=(12, 5))
    
    # パネル1: 多項式回帰
    sp3_1 = sp_arr3[0]
    np.random.seed(42)
    x3_1 = np.linspace(-5, 5, 30)
    y3_1 = 0.5 * x3_1**3 - 2 * x3_1 + np.random.normal(0, 5, 30)
    sp3_1.scatter(x3_1, y3_1, alab=["X", "Y"], lab="Data")
    sp3_1.Regression(regr=3)
    sp3_1.ax.set_title("Polynomial Regression (regr=3)")

    # パネル2: 任意関数フィットと補助線 (Optuna使用)
    sp3_2 = sp_arr3[1]
    x3_2 = np.linspace(0.1, 5, 50)
    y3_2 = 2.5 * np.exp(-1.2 * x3_2) + np.random.normal(0, 0.05, 50)
    sp3_2.scatter(x3_2, y3_2, alab=["Time (s)", "Intensity"], lab="Data",
                  vx=[1, 3], vcol='red', vstyle='--', vwidth=1.5,
                  hy=0, hcol='blue', hstyle=':', hwidth=1.0)
    
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
        
    # auto_p0=True にすることで、Optunaが p0 を自動探索します
    sp3_2.Regression(regr=exp_decay, auto_p0=True, n_trials=50, bounds=([0, 0], [10, 5]))
    sp3_2.ax.set_title("Optuna Auto Fit & Guide Lines")
    
    fig3.savefig("images/example3_regression.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    print(" - example3_regression.png を作成しました")

    # ==========================================
    # 🌟 4. Imshow と 3D (example4_3d.png)
    # ==========================================
    fig4, sp_arr4 = create_symple_plots(1, 2, figsize=(12, 5))
    sp4_1 = sp_arr4[0]
    z_im = np.random.rand(50, 50) * 1e-4
    sp4_1.imshow([np.linspace(0, 5, 50)], [np.linspace(0, 5, 50)], z_im,
                 vmax=1e-4, alab=["X (um)", "Y (um)", "Intensity"])

    sp_arr4[1].ax.remove()
    ax_3d = fig4.add_subplot(1, 2, 2, projection='3d')
    sp4_2 = symple_plot(ax_3d)
    z_3d = np.linspace(0, 10, 100)
    sp4_2.tdscatter(np.sin(z_3d), np.cos(z_3d), z_3d, alab=["X", "Y", "Z"])
    fig4.savefig("images/example4_3d.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig4)
    print(" - example4_3d.png を作成しました")

    # ==========================================
    # 🌟 5. ユーティリティ・スタイル適用 (example5_utils.png)
    # ==========================================
    fig5, sp_arr5 = create_symple_plots(1, 2, figsize=(10, 4), style='slide', auto_label=True)
    x5 = np.linspace(0, 5, 20)
    sp_arr5[0].plot(x5, np.exp(x5), alab=["Time", "Growth"], lab="Exponential")
    sp_arr5[1].scatter(x5, x5**3, alab=["Time", "Value"], size=80, marker='s', lab="Quadratic")
    fig5.savefig("images/example5_utils.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig5)
    print(" - example5_utils.png を作成しました")

    set_style('default')

    # ==========================================
    # 🌟 6. 個別カラー指定と強制ズーム (example6_zoom_col.png)
    # ==========================================
    fig6, sp_arr6 = create_symple_plots(2, 2, figsize=(15, 15))
    
    x_bg = np.linspace(0, 20, 100)
    y_bg = np.sin(x_bg)
    x_target = np.linspace(5, 10, 50)
    y_target = np.sin(x_target)
    
    sp6_1 = sp_arr6[0]
    sp6_1.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
    sp6_1.plot(x_target, y_target, col='red', lab="Target (zoom='x')", zoom='x', linewidth=3)
    sp6_1.ax.set_title("zoom='x'", fontsize=14)

    sp6_2 = sp_arr6[1]
    sp6_2.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
    sp6_2.plot(x_target, y_target, col='red', lab="Target (zoom='y')", zoom='y', linewidth=3)
    sp6_2.ax.set_title("zoom='y'", fontsize=14)

    sp6_3 = sp_arr6[2]
    sp6_3.plot(x_bg, y_bg, col='gray', lab="Background", linestyle=['--'], alab=["X", "Y"])
    sp6_3.plot(x_target, y_target, col='red', lab="Target (zoom='both')", zoom='xy', linewidth=3)
    sp6_3.ax.set_title("zoom='xy'", fontsize=14)

    sp6_4 = sp_arr6[3]
    sp6_4.plot(x_bg, y_bg, col='gray', lab="Full Data", alab=["X", "Y"])
    x_peak = np.linspace(7.2, 7.8, 50)
    y_peak = np.sin(x_peak) + 3 * np.exp(-((x_peak - 7.5)**2) / 0.01)
    sp6_4.plot(x_peak, y_peak, col='green', lab="Sharp Peak", zoomx=[7.2, 7.8])
    sp6_4.ax.set_title("Auto Inset Zoom (zoomx)", fontsize=14)
    
    fig6.savefig("images/example6_zoom_col.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig6)
    print(" - example6_zoom_col.png を作成しました")

    print("\n✅ All example images have been successfully generated!")

if __name__ == "__main__":
    main()