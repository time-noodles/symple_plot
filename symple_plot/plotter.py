import numpy as np
import os
import string
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
import mpl_toolkits.axes_grid1

# ==========================================
# 🌟 論文・スライド用スタイル一括設定機能 🌟
# ==========================================
def set_style(mode='default'):
    """
    描画スタイルを一括設定します。
    mode='paper': 論文用 (serifフォント, 細めの線)
    mode='slide': プレゼン用 (sans-serifフォント, 太めの線, 大きな文字)
    mode='default': Matplotlibの初期状態に戻す
    """
    if mode == 'paper':
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'mathtext.fontset': 'stix',
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.5,
            'font.size': 14,
        })
    elif mode == 'slide':
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'axes.linewidth': 2.0,
            'lines.linewidth': 2.5,
            'font.size': 18,
        })
    elif mode == 'default':
        plt.rcdefaults()

# ==========================================
# 0. GrADSカラーマップ生成
# ==========================================
def get_grads_cmap():
    colors = np.array([
        [160,   0, 200], [130,   0, 220], [ 30,  60, 255], 
        [  0, 160, 255], [  0, 200, 200], [  0, 210, 140], 
        [  0, 220,   0], [160, 230,  50], [230, 220,  50], 
        [230, 175,  45], [240, 130,  40], [250,  60,  60], 
        [240,   0, 130]
    ], dtype=float)
    colors /= 255.0
    return LinearSegmentedColormap.from_list("grads_cmap", colors)

# ==========================================
# 1. 軸フォーマッタ (指数統一・科学的記数法)
# ==========================================
class AutoSmartFormatter(Formatter):
    def __call__(self, x, pos=None):
        locs = self.axis.get_ticklocs()
        step = np.median(np.diff(locs)) if len(locs) > 1 else 1.0
        
        max_val = np.max(np.abs(locs)) if len(locs) > 0 else 1.0
        if max_val == 0: max_val = 1.0
            
        if step > 0:
            log_step = np.log10(step)
            base_dec = -int(np.floor(log_step))
            if base_dec < 0:
                decimals = 0
            else:
                norm_step = round(step * 10**base_dec, 5)
                decimals = base_dec + 1 if not float(norm_step).is_integer() else base_dec
        else:
            decimals = 0
            
        if np.isclose(x, 0, atol=step*1e-5 if step > 0 else 1e-8):
            return "0"
            
        if max_val >= 1e4 or (max_val <= 1e-4 and max_val > 0):
            global_exp = int(np.floor(np.log10(max_val)))
            mantissa = x / (10**global_exp)
            step_mag = int(np.floor(np.log10(step))) if step > 0 else 0
            m_dec = max(0, global_exp - step_mag)
            return rf"${mantissa:.{m_dec}f} \times 10^{{{global_exp}}}$"
                
        return f"{x:.{decimals}f}"

# ==========================================
# 2. 補助関数群 (対数スケールマージン対応)
# ==========================================
def ensure_2d(data):
    if len(data) == 0: return [[]]
    # 🌟 pd.Series も配列として正しく認識させる！
    if not isinstance(data[0], (list, tuple, np.ndarray, pd.Series)): 
        return [data]
    return data

def pad_list(L):
    max_len = max([len(i) for i in L])
    L_padded = [list(i) + [np.nan] * (max_len - len(i)) for i in L]
    # 🌟 CSVから読み込んだ文字列('5.0'など)も確実に数値としてグラフ化できるよう dtype=float を明記
    return [np.array(i, dtype=float) for i in L_padded]

def minmax(val, margin=0.05, is_log=False):
    v_flat = np.concatenate([np.ravel(v) for v in val]) if len(val) > 0 else np.array([])
    v_flat = v_flat[~np.isnan(v_flat)]
    
    if len(v_flat) == 0:
        return (0.1, 10) if is_log else (-1, 1)

    if is_log:
        v_flat = v_flat[v_flat > 0]
        if len(v_flat) == 0: return 0.1, 10
        min0, max0 = np.min(v_flat), np.max(v_flat)
        log_min, log_max = np.log10(min0), np.log10(max0)
        dif = log_max - log_min
        if dif == 0: return 10**(log_min - margin), 10**(log_max + margin)
        return 10**(log_min - dif * margin), 10**(log_max + dif * margin)
    else:
        min0, max0 = np.min(v_flat), np.max(v_flat)
        dif = max0 - min0
        if dif == 0: return min0 - abs(min0) * margin, max0 + abs(max0) * margin
        return min0 - dif * margin, max0 + dif * margin

def valid_xy(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]

def reg_n(reg, x):
    y = np.zeros_like(x)
    for num, i in enumerate(range(len(reg)-1, -1, -1)):
        y = y + reg[num] * x**i
    return y

def alpha_calc(N, num):
    N -= 1
    return 1 if N == 0 else (num / N * 0.75 + 0.25)

# 🌟 自動化対応版 create_symple_plots 🌟
def create_symple_plots(nrows=1, ncols=1, figsize=None, style=None, auto_label=False, **kwargs):
    """
    グラフ枠を生成します。
    style: 'paper', 'slide', 'default' を指定するとスタイルが一括適用されます。
    auto_label: Trueにすると、各パネルの左上に (a), (b)... と自動でラベルを振ります。
    ※複数パネルの場合、戻り値のオブジェクト群は常に1次元配列として返されます。
    """
    if style:
        set_style(style)

    if figsize is None: figsize = (6 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    # グラフオブジェクトの生成
    if nrows == 1 and ncols == 1:
        # 1x1の場合はそのまま返す（後方互換性）
        ret_arr = symple_plot(axes)
        flat_sps = [ret_arr]
    elif axes.ndim == 1:
        # 1行複数列、または複数行1列の場合はそのまま1次元配列にする
        ret_arr = np.array([symple_plot(ax) for ax in axes])
        flat_sps = ret_arr
    else:
        # 複数行・複数列の場合も、flatten()を使って1次元配列（[0], [1], [2]...）として返す
        flat_sps = np.array([symple_plot(ax) for ax in axes.flatten()])
        ret_arr = flat_sps

    # 🌟 パネルラベルの全自動付与 🌟
    if auto_label:
        import string
        alphabet = string.ascii_lowercase # a, b, c, d...
        for i, sp in enumerate(flat_sps):
            if i < len(alphabet):
                sp.add_panel_label(f"({alphabet[i]})")

    return fig, ret_arr

# ==========================================
# 3. メインクラス: symple_plot
# ==========================================
class symple_plot:
    def __init__(self, ax):
        self.ax = ax
        self.axilab = 20
        self.axinum = 17
        self.tlength = 5
        self.col = 'grads'
        self.aspect = 1
        
        self.X, self.Y, self.Z = [], [], []
        self.COL = []
        self.sca = []
        
        self.current_xmin, self.current_xmax = None, None
        self.current_ymin, self.current_ymax = None, None
        self.current_zmin, self.current_zmax = None, None
        
        self.all_handles = []
        self.all_labels = []

    def setxy(self, X, Y):
        X, Y = ensure_2d(X), ensure_2d(Y)
        self.X, self.Y = pad_list(X), pad_list(Y)

    def setxyz(self, X, Y, Z):
        X, Y, Z = ensure_2d(X), ensure_2d(Y), ensure_2d(Z)
        self.X, self.Y, self.Z = pad_list(X), pad_list(Y), pad_list(Z)

    def col_c(self, **kwargs):
        # 🌟 ここで kwargs から col を受け取るように一元化！
        if 'col' in kwargs:
            self.col = kwargs['col']
            
        self.COL = []
        num_data = len(self.X)
        if self.col in ['default', 'turbo', 'plasma', 'viridis', 'cool']:
            cmap = plt.get_cmap(self.col if self.col != 'default' else 'turbo')
            self.COL = [cmap(0.5)] if num_data == 1 else [cmap(val) for val in np.linspace(0.90, 0.05, num_data)]
        elif self.col == 'grads':
            cmap = get_grads_cmap()
            self.COL = [cmap(0.5)] if num_data == 1 else [cmap(val) for val in np.linspace(1, 0, num_data)]
        elif self.col == 'mode1':
            cl = plt.rcParams['axes.prop_cycle'].by_key()['color']
            self.COL = [cl[i % len(cl)] for i in range(num_data)]
        elif isinstance(self.col, list):
            self.COL = self.col
        else:
            self.COL = [self.col for _ in range(num_data)]

    def _apply_common_settings(self, **kwargs):
        margin = kwargs.get('margin', 0.05)
        is_logx = kwargs.get('logx', False)
        is_logy = kwargs.get('logy', False)
        is_logz = kwargs.get('logz', False)
        
        # 🌟 zoom引数を確実に文字列として評価し、バグを回避
        zoom_str = kwargs.get('zoom', '')
        if zoom_str is None: zoom_str = ''
        zoom_str = str(zoom_str).lower()

        new_xmin, new_xmax = minmax(self.X, margin, is_log=is_logx)
        new_ymin, new_ymax = minmax(self.Y, margin, is_log=is_logy)

        cx = kwargs.get('cx')
        cy = kwargs.get('cy')

        if cx and not cy:
            valid_y = []
            for x_arr, y_arr in zip(self.X, self.Y):
                vx, vy = valid_xy(x_arr, y_arr)
                mask = (vx >= cx[0]) & (vx <= cx[1])
                valid_y.append(vy[mask])
            new_ymin, new_ymax = minmax(valid_y, margin, is_log=is_logy)
            
        if cy and not cx:
            valid_x = []
            for x_arr, y_arr in zip(self.X, self.Y):
                vx, vy = valid_xy(x_arr, y_arr)
                mask = (vy >= cy[0]) & (vy <= cy[1])
                valid_x.append(vx[mask])
            new_xmin, new_xmax = minmax(valid_x, margin, is_log=is_logx)

        # 🌟 zoom機能: 'x' や 'y' が明確に含まれていれば強制上書き
        if self.current_xmin is None or 'x' in zoom_str:
            self.current_xmin, self.current_xmax = new_xmin, new_xmax
        else:
            self.current_xmin = min(self.current_xmin, new_xmin)
            self.current_xmax = max(self.current_xmax, new_xmax)

        if self.current_ymin is None or 'y' in zoom_str:
            self.current_ymin, self.current_ymax = new_ymin, new_ymax
        else:
            self.current_ymin = min(self.current_ymin, new_ymin)
            self.current_ymax = max(self.current_ymax, new_ymax)

        if cx: self.current_xmin, self.current_xmax = cx[0], cx[1]
        if cy: self.current_ymin, self.current_ymax = cy[0], cy[1]

        if is_logx: self.ax.set_xscale('log')
        if is_logy: self.ax.set_yscale('log')

        self.ax.set_xlim(self.current_xmin, self.current_xmax)
        self.ax.set_ylim(self.current_ymin, self.current_ymax)

        is_3d = hasattr(self.ax, 'set_zlim')
        if is_3d and len(self.Z) > 0:
            new_zmin, new_zmax = minmax(self.Z, margin, is_log=is_logz)
            if self.current_zmin is None or 'z' in zoom_str:
                self.current_zmin, self.current_zmax = new_zmin, new_zmax
            else:
                self.current_zmin = min(self.current_zmin, new_zmin)
                self.current_zmax = max(self.current_zmax, new_zmax)
                
            if cz := kwargs.get('cz'): self.current_zmin, self.current_zmax = cz[0], cz[1]
            if is_logz: self.ax.set_zscale('log')
            self.ax.set_zlim(self.current_zmin, self.current_zmax)
            if not is_logz: self.ax.zaxis.set_major_formatter(AutoSmartFormatter())

        if not is_logx: self.ax.xaxis.set_major_formatter(AutoSmartFormatter())
        if not is_logy: self.ax.yaxis.set_major_formatter(AutoSmartFormatter())
            
        self.ax.tick_params(axis='both', labelsize=self.axinum, length=self.tlength)
        if kwargs.get('nox', False): self.ax.tick_params(labelbottom=False)
        if kwargs.get('noy', False): self.ax.tick_params(labelleft=False)

        if alab := kwargs.get('alab'):
            self.ax.set_xlabel(alab[0], fontsize=self.axilab)
            self.ax.set_ylabel(alab[1], fontsize=self.axilab)
            if is_3d and len(alab) > 2: self.ax.set_zlabel(alab[2], fontsize=self.axilab)

        if lab := kwargs.get('lab'):
            if not isinstance(lab, list): lab = [lab]
            for handle, label_text in zip(self.sca, lab):
                self.all_handles.append(handle)
                self.all_labels.append(label_text)
            loc = kwargs.get('loc', 'upper left')
            if len(self.all_handles) > 0:
                self.ax.legend(self.all_handles, self.all_labels, bbox_to_anchor=(1.01, 1), loc=loc, frameon=False, fontsize=self.axinum)

        if not is_3d: self.ax.set_aspect(self.aspect / self.ax.get_data_ratio(), adjustable="box")
        self.ax.figure.tight_layout()

        # 🌟 zoomx, zoomy で自動的に add_inset_zoom を呼び出す機能
        zoomx = kwargs.get('zoomx')
        zoomy = kwargs.get('zoomy')
        if zoomx is not None or zoomy is not None:
            self.add_inset_zoom(xlim=zoomx, ylim=zoomy)

    # ---------------------------------------------------------
    # 各種描画メソッド群
    # ---------------------------------------------------------
    def pre_set(self, X, Y, **kwargs):
        self.setxy(X, Y)
        self.sca = []
        self._apply_common_settings(**kwargs)
        return self.ax

    def scatter(self, X, Y, **kwargs):
        self.setxy(X, Y)
        self.col_c(**kwargs)
        marker_size = kwargs.get('size', 40)
        markers = kwargs.get('marker', ['o'])
        if not isinstance(markers, list): markers = [markers]
        self.sca = []
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            m = markers[i % len(markers)]
            scat = self.ax.scatter(x, y, color=self.COL[i], s=marker_size, marker=m)
            self.sca.append(scat)
        self._apply_common_settings(**kwargs)
        return self.ax

    def plot(self, X, Y, **kwargs):
        self.setxy(X, Y)
        self.col_c(**kwargs)
        linestyles = kwargs.get('linestyle', ['-'])
        if not isinstance(linestyles, list): linestyles = [linestyles]
        linewidth = kwargs.get('linewidth', 2)
        self.sca = []
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            ls = linestyles[i % len(linestyles)]
            p, = self.ax.plot(x, y, color=self.COL[i], linestyle=ls, linewidth=linewidth)
            self.sca.append(p)
        self._apply_common_settings(**kwargs)
        return self.ax

    def Regression(self, regr, directory='./', **kwargs):
        self.col_c(**kwargs)
        x_l = np.linspace(self.current_xmin, self.current_xmax, 1000)
        df_rows = []
        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            vx, vy = valid_xy(x, y)
            if len(vx) <= regr: continue
            fit, cov = np.polyfit(vx, vy, regr, cov=True)
            err = [cov[j][j]**0.5 * 2 for j in range(regr+1)]
            y_pred = reg_n(fit, vx)
            r2 = r2_score(vy, y_pred)
            df_rows.append([f"Data_{i}_Coef"] + fit.tolist())
            df_rows.append([f"Data_{i}_Error"] + err)
            df_rows.append([f"Data_{i}_R2"] + [r2] + [np.nan] * regr)
            self.ax.plot(x_l, reg_n(fit, x_l), color=self.COL[i], linestyle='--')
            
        if df_rows:
            df = pd.DataFrame(df_rows)
            save_path = os.path.join(directory, 'regression_results.csv')
            df.to_csv(save_path, mode='a' if os.path.exists(save_path) else 'w', header=False, index=False)
            print(f"Regression data appended to {save_path}")
        self.ax.figure.tight_layout()
        return self.ax

    def tdscatter(self, X, Y, Z, **kwargs):
        self.setxyz(X, Y, Z)
        self.col_c(**kwargs)
        marker_size = kwargs.get('size', 40)
        self.sca = []
        for i, (x, y, z) in enumerate(zip(self.X, self.Y, self.Z)):
            scat = self.ax.scatter(x, y, z, color=self.COL[i], s=marker_size)
            self.sca.append(scat)
        self._apply_common_settings(**kwargs)
        return self.ax, self.sca

    def tdplot(self, X, Y, Z, **kwargs):
        self.setxyz(X, Y, Z)
        self.col_c(**kwargs)
        self.sca = []
        for i, (x, y, z) in enumerate(zip(self.X, self.Y, self.Z)):
            p = self.ax.plot_wireframe(x, y, z, color=self.COL[i])
            self.sca.append(p)
        self._apply_common_settings(**kwargs)
        return self.ax, self.sca

    def imshow(self, X, Y, Z, vmax, **kwargs):
        Z = np.array(Z)
        if Z.ndim == 3: Z = Z[0]
        zx, zy = Z.shape
        if 'col' in kwargs: self.col = kwargs['col']
        if self.col == 'grads':
            cmap_obj = get_grads_cmap()
        elif self.col == 'default':
            cmap_obj = 'jet'
        elif isinstance(self.col, str):
            cmap_obj = self.col
        else:
            cmap_obj = 'jet'
            
        self.im = self.ax.imshow(Z, vmin=0, vmax=vmax, aspect=zy/zx, cmap=cmap_obj)
        self.ax.invert_yaxis()
        
        if len(X) > 0 and len(Y) > 0:
            X, Y = np.array(X)[0], np.array(Y)[0]
            self.ax.set_xlim(-0.5, zy - 0.5)
            self.ax.set_ylim(-0.5, zx - 0.5)
            
            if kwargs.get('logx', False): self.ax.set_xscale('log')
            else: self.ax.xaxis.set_major_formatter(AutoSmartFormatter())
            
            if kwargs.get('logy', False): self.ax.set_yscale('log')
            else: self.ax.yaxis.set_major_formatter(AutoSmartFormatter())
        
        if kwargs.get('nox', False): self.ax.tick_params(labelbottom=False)
        if kwargs.get('noy', False): self.ax.tick_params(labelleft=False)

        divider = mpl_toolkits.axes_grid1.make_axes_locatable(self.ax)
        cax = divider.append_axes('right', size='5%', pad='3%')
        cbar = self.ax.figure.colorbar(self.im, cax=cax)
        cbar.ax.tick_params(labelsize=self.axinum)
        
        if kwargs.get('logz', False): 
            pass 
        else:
            cbar.ax.yaxis.set_major_formatter(AutoSmartFormatter())
        
        if alab := kwargs.get('alab'):
            self.ax.set_xlabel(alab[0], fontsize=self.axilab)
            self.ax.set_ylabel(alab[1], fontsize=self.axilab)
            if len(alab) > 2:
                cbar.set_label(alab[2], fontsize=self.axilab)
                
        self.ax.figure.tight_layout()
        return self.ax, self.im

    # ==========================================
    # 🌟 パネルラベル自動付与 (a), (b) 🌟
    # ==========================================
    def add_panel_label(self, text, x=-0.15, y=1.05, fontsize=None, weight='bold'):
        """
        論文用のパネルラベル (a), (b) などを自動配置します。
        """
        if fontsize is None:
            fontsize = self.axilab + 2
            
        self.ax.text(x, y, text, transform=self.ax.transAxes, 
                     fontsize=fontsize, fontweight=weight, 
                     va='bottom', ha='right')
        return self.ax

# ==========================================
    # 🌟 INSET ZOOM (自動探索拡大図 - 絶妙バランス・最大化・全自動対応版) 🌟
    # ==========================================
    def add_inset_zoom(self, xlim=None, ylim=None, bounds='auto', margin=0.02, draw_lines=True):
        """
        xlimまたはylimを与えると、プロット済みの全データから該当範囲を自動探索し、
        inset_axes（拡大図）を作成して元のグラフと枠線で結びます。
        xlimとylimの両方が与えられた場合（zoomx, zoomyなど）は、そのままその範囲を描画します。
        """
        all_x, all_y = [], []
        for line in self.ax.get_lines():
            all_x.extend(line.get_xdata())
            all_y.extend(line.get_ydata())
        for coll in self.ax.collections:
            import matplotlib.collections as mcoll
            if isinstance(coll, mcoll.PathCollection):
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    all_x.extend(offsets[:, 0])
                    all_y.extend(offsets[:, 1])
                    
        all_x, all_y = np.array(all_x, dtype=float), np.array(all_y, dtype=float)
        mask = ~np.isnan(all_x) & ~np.isnan(all_y)
        all_x, all_y = all_x[mask], all_y[mask]
        
        is_logx = self.ax.get_xscale() == 'log'
        is_logy = self.ax.get_yscale() == 'log'

        if xlim is not None and ylim is None:
            in_range = (all_x >= xlim[0]) & (all_x <= xlim[1])
            valid_y = all_y[in_range]
            if is_logy: valid_y = valid_y[valid_y > 0]
            if len(valid_y) > 0:
                y_min, y_max = np.min(valid_y), np.max(valid_y)
                if is_logy:
                    log_min, log_max = np.log10(y_min), np.log10(y_max)
                    dif = log_max - log_min if log_max != log_min else 1
                    ylim = [10**(log_min - dif*margin), 10**(log_max + dif*margin)]
                else:
                    dif = y_max - y_min if y_max != y_min else abs(y_min)*0.1
                    ylim = [y_min - dif*margin, y_max + dif*margin]
            else:
                ylim = self.ax.get_ylim()

        elif ylim is not None and xlim is None:
            in_range = (all_y >= ylim[0]) & (all_y <= ylim[1])
            valid_x = all_x[in_range]
            if is_logx: valid_x = valid_x[valid_x > 0]
            if len(valid_x) > 0:
                x_min, x_max = np.min(valid_x), np.max(valid_x)
                if is_logx:
                    log_min, log_max = np.log10(x_min), np.log10(x_max)
                    dif = log_max - log_min if log_max != log_min else 1
                    xlim = [10**(log_min - dif*margin), 10**(log_max + dif*margin)]
                else:
                    dif = x_max - x_min if x_max != x_min else abs(x_min)*0.1
                    xlim = [x_min - dif*margin, x_max + dif*margin]
            else:
                xlim = self.ax.get_xlim()
        
        elif xlim is not None and ylim is not None:
            pass # 両方指定された場合は、そのまま限界値として採用する
            
        elif xlim is None and ylim is None:
            return None 

        if isinstance(bounds, str):
            if bounds == 'auto':
                xmin_main, xmax_main = self.ax.get_xlim()
                ymin_main, ymax_main = self.ax.get_ylim()
                
                ax_x = np.zeros_like(all_x)
                ax_y = np.zeros_like(all_y)
                
                if is_logx and xmin_main > 0 and xmax_main > xmin_main:
                    valid_x = all_x > 0
                    ax_x[valid_x] = (np.log10(all_x[valid_x]) - np.log10(xmin_main)) / (np.log10(xmax_main) - np.log10(xmin_main))
                    ax_x[~valid_x] = -1
                else:
                    ax_x = (all_x - xmin_main) / (xmax_main - xmin_main) if xmax_main != xmin_main else all_x * 0
                    
                if is_logy and ymin_main > 0 and ymax_main > ymin_main:
                    valid_y = all_y > 0
                    ax_y[valid_y] = (np.log10(all_y[valid_y]) - np.log10(ymin_main)) / (np.log10(ymax_main) - np.log10(ymin_main))
                    ax_y[~valid_y] = -1
                else:
                    ax_y = (all_y - ymin_main) / (ymax_main - ymin_main) if ymax_main != ymin_main else all_y * 0
                    
                in_plot = (ax_x >= 0) & (ax_x <= 1) & (ax_y >= 0) & (ax_y <= 1)
                ax_x, ax_y = ax_x[in_plot], ax_y[in_plot]
                
                # サイズを45%に保ちつつ、衝突回避のため左と下だけ余白を0.12取る
                sizes_to_try = [0.45, 0.40, 0.35, 0.30]
                best_bound = None
                fallback_bound = None
                min_overlap = float('inf')
                
                pad_left = 0.12
                pad_bottom = 0.12
                pad_right = 0.05
                pad_top = 0.05
                
                for size in sizes_to_try:
                    loc_map_dynamic = {
                        'upper left':  [pad_left, 1 - pad_top - size, size, size],
                        'upper right': [1 - pad_right - size, 1 - pad_top - size, size, size],
                        'lower left':  [pad_left, pad_bottom, size, size],
                        'lower right': [1 - pad_right - size, pad_bottom, size, size]
                    }
                    
                    for name, box in loc_map_dynamic.items():
                        x0, y0, w, h = box
                        pad_data = 0.03
                        overlap = (ax_x >= x0 - pad_data) & (ax_x <= x0 + w + pad_data) & \
                                  (ax_y >= y0 - pad_data) & (ax_y <= y0 + h + pad_data)
                        num_overlap = np.sum(overlap)
                        
                        if num_overlap == 0:
                            best_bound = box
                            break
                            
                        if num_overlap < min_overlap:
                            min_overlap = num_overlap
                            fallback_bound = box
                            
                    if best_bound is not None:
                        break
                
                if best_bound is not None:
                    bounds = best_bound
                else:
                    if min_overlap > len(ax_x) * 0.15 and len(ax_x) > 0:
                        bounds = [1.05, 0.3, 0.45, 0.45] 
                    else:
                        bounds = fallback_bound
            else:
                size = 0.40
                pad_left, pad_bottom, pad_right, pad_top = 0.12, 0.12, 0.05, 0.05
                loc_map = {
                    'upper left':  [pad_left, 1 - pad_top - size, size, size],
                    'upper right': [1 - pad_right - size, 1 - pad_top - size, size, size],
                    'lower left':  [pad_left, pad_bottom, size, size],
                    'lower right': [1 - pad_right - size, pad_bottom, size, size]
                }
                bounds = loc_map.get(bounds, loc_map['upper right'])

        axins = self.ax.inset_axes(bounds)

        for line in self.ax.get_lines():
            axins.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(), 
                       linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                       marker=line.get_marker(), markersize=line.get_markersize(), alpha=line.get_alpha())
        for coll in self.ax.collections:
            import matplotlib.collections as mcoll
            if isinstance(coll, mcoll.PathCollection):
                offsets = coll.get_offsets()
                if len(offsets) > 0:
                    axins.scatter(offsets[:,0], offsets[:,1], color=coll.get_facecolors(), 
                                  s=coll.get_sizes(), alpha=coll.get_alpha())

        if is_logx: axins.set_xscale('log')
        if is_logy: axins.set_yscale('log')
        
        axins.set_xlim(xlim[0], xlim[1])
        axins.set_ylim(ylim[0], ylim[1])
        
        if not is_logx: axins.xaxis.set_major_formatter(AutoSmartFormatter())
        if not is_logy: axins.yaxis.set_major_formatter(AutoSmartFormatter())
        
        axins.tick_params(labelsize=self.axinum - 7)

        if draw_lines:
            self.ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)
        
        return axins