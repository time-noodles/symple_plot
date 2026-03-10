import numpy as np
import os
import string
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import Formatter
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score
import mpl_toolkits.axes_grid1

# 上部のインポート
from .data_utils import valid_xy, pad_list, minmax, ensure_2d, get_yrange, get_xrange
from .fit_utils import auto_curve_fit, reg_n

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
# 1. GrADSカラーマップ生成
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
# 2. 軸フォーマッタ (指数統一・科学的記数法)
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

def alpha_calc(N, num):
    N -= 1
    return 1 if N == 0 else (num / N * 0.75 + 0.25)

def create_symple_plots(nrows=1, ncols=1, figsize=None, style=None, auto_label=False, flush=False, **kwargs):
    """
    グラフ枠を生成します。
    flush=True にすると、隙間のない結合グリッド（共有軸）を作成します。
    """
    if style:
        set_style(style)

    if figsize is None: figsize = (7*ncols+(ncols-1),5*nrows+(ncols-1)*2)
    
    # 🌟 隙間なしグリッド機能 (flush=True) 🌟
    if flush:
        if 'gridspec_kw' not in kwargs:
            kwargs['gridspec_kw'] = {}
        kwargs['gridspec_kw'].setdefault('wspace', 0)
        kwargs['gridspec_kw'].setdefault('hspace', 0)
        kwargs.setdefault('sharex', True)
        kwargs.setdefault('sharey', True)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    
    if nrows == 1 and ncols == 1:
        ret_arr = symple_plot(axes)
        flat_sps = [ret_arr]
    elif axes.ndim == 1:
        ret_arr = np.array([symple_plot(ax) for ax in axes])
        flat_sps = ret_arr
    else:
        flat_sps = np.array([symple_plot(ax) for ax in axes.flatten()])
        ret_arr = flat_sps

    # 🌟 flush=True の場合は、隙間を埋めるために強制的にアスペクト比固定を解除する
    if flush:
        for sp in flat_sps:
            sp.aspect = 'auto'

    if auto_label:
        import string
        alphabet = string.ascii_lowercase
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

    def setxy(self, X, Y):
        X, Y = ensure_2d(X), ensure_2d(Y)
        self.X, self.Y = pad_list(X), pad_list(Y)

    def setxyz(self, X, Y, Z):
        X, Y, Z = ensure_2d(X), ensure_2d(Y), ensure_2d(Z)
        self.X, self.Y, self.Z = pad_list(X), pad_list(Y), pad_list(Z)

    def col_c(self, **kwargs):
        if 'col' in kwargs:
            self.col = kwargs['col']
            
        self.COL = []
        num_data = len(self.X)
        if self.col in ['default', 'turbo', 'plasma', 'viridis', 'cool','gist_gray']:
            cmap = plt.get_cmap(self.col if self.col != 'default' else 'turbo')
            self.COL = [cmap(0.5)] if num_data == 1 else [cmap(val) for val in np.linspace(0.90, 0.05, num_data)]
        elif self.col == 'grads':
            cmap = get_grads_cmap()
            self.COL = [cmap(0.5)] if num_data == 1 else [cmap(val) for val in np.linspace(1, 0, num_data)]
        elif self.col == 'model1':
            cl = plt.rcParams['axes.prop_cycle'].by_key()['color']
            self.COL = [cl[i % len(cl)] for i in range(num_data)]
        elif isinstance(self.col, list):
            self.COL = self.col
        else:
            self.COL = [self.col for _ in range(num_data)]

    def _apply_common_settings(self, **kwargs):
        self.axilab = kwargs.get('axilab', self.axilab)
        self.axinum = kwargs.get('axinum', self.axinum)        
        margin = kwargs.get('margin', 0.05)
        is_logx = kwargs.get('logx', False)
        is_logy = kwargs.get('logy', False)
        is_logz = kwargs.get('logz', False)
        
        zoom_str = kwargs.get('zoom', '')
        if zoom_str is None: zoom_str = ''
        zoom_str = str(zoom_str).lower()

        # 🌟 全描画要素からデータ範囲を取得し、重ね描き時にすべてが収まるようにする
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
                    
        all_x = np.array(all_x, dtype=float)
        all_y = np.array(all_y, dtype=float)
        mask = ~np.isnan(all_x) & ~np.isnan(all_y)
        all_x, all_y = all_x[mask], all_y[mask]
        
        if len(all_x) == 0:
            all_x = np.concatenate([np.ravel(v) for v in self.X]) if len(self.X) > 0 else np.array([])
            all_y = np.concatenate([np.ravel(v) for v in self.Y]) if len(self.Y) > 0 else np.array([])

        new_xmin, new_xmax = minmax([all_x], margin, is_log=is_logx)
        new_ymin, new_ymax = minmax([all_y], margin, is_log=is_logy)

        cx = kwargs.get('cx')
        cy = kwargs.get('cy')

        if cx and not cy:
            _, y_fil = get_yrange(all_x, all_y, cx[0], cx[1])
            if len(y_fil) > 0: 
                new_ymin, new_ymax = minmax([y_fil], margin, is_log=is_logy)
            
        if cy and not cx:
            x_fil, _ = get_xrange(all_x, all_y, cy[0], cy[1])
            if len(x_fil) > 0: 
                new_xmin, new_xmax = minmax([x_fil], margin, is_log=is_logx)

        # zoom引数で 'x' や 'y' が指定された場合のみ、今回渡されたデータ範囲にフォーカス
        if 'x' in zoom_str:
            new_xmin, new_xmax = minmax(self.X, margin, is_log=is_logx)
        if 'y' in zoom_str:
            new_ymin, new_ymax = minmax(self.Y, margin, is_log=is_logy)

        self.current_xmin, self.current_xmax = new_xmin, new_xmax
        self.current_ymin, self.current_ymax = new_ymin, new_ymax

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
        
        self.ax.tick_params(which='major', labelsize=self.axinum)
        
        if not is_3d:
            self.ax.minorticks_on()
            self.ax.tick_params(which='major', direction='in', length=self.tlength, 
                                top=True, bottom=True, left=True, right=True, labelsize=self.axinum)
            self.ax.tick_params(which='minor', direction='in', length=self.tlength * 0.5, 
                                top=True, bottom=True, left=True, right=True)
        else:
            self.ax.tick_params(axis='both', labelsize=self.axinum, length=self.tlength)

        if kwargs.get('nox', False) or kwargs.get('nonx', False): self.ax.tick_params(labelbottom=False)
        if kwargs.get('noy', False) or kwargs.get('nony', False): self.ax.tick_params(labelleft=False)

        if alab := kwargs.get('alab'):
            self.ax.set_xlabel(alab[0], fontsize=self.axilab)
            self.ax.set_ylabel(alab[1], fontsize=self.axilab)
            if is_3d and len(alab) > 2: self.ax.set_zlabel(alab[2], fontsize=self.axilab)

        if lab := kwargs.get('lab'):
            if not isinstance(lab, list): lab = [lab]
            loc = kwargs.get('loc', 'upper left')
            lab_fs = kwargs.get('lab_fs', self.axinum)
            
            if isinstance(loc, str) and loc.startswith('inline'):
                align = loc.split('_')[1] if '_' in loc else 'auto'
                
                left_ys, right_ys = [], []
                for x_arr, y_arr in zip(self.X, self.Y):
                    vx, vy = valid_xy(x_arr, y_arr)
                    if len(vx) > 0:
                        left_ys.append(vy[0]); right_ys.append(vy[-1])
                    else:
                        left_ys.append(np.nan); right_ys.append(np.nan)
                
                if align == 'auto':
                    l_val = np.array(left_ys)[~np.isnan(left_ys)]
                    r_val = np.array(right_ys)[~np.isnan(right_ys)]
                    def min_dist(arr):
                        if len(arr) < 2: return 1.0
                        return np.min(np.diff(np.sort(arr)))
                    align = 'right' if min_dist(r_val) >= min_dist(l_val) else 'left'

                inline_dy = kwargs.get('inline_dy', 0)
                if not isinstance(inline_dy, (list, tuple, np.ndarray)):
                    inline_dy = [inline_dy] * len(self.X)
                
                x_range = self.current_xmax - self.current_xmin
                x_in_offset = x_range * 0.005 

                for i, (x_arr, y_arr) in enumerate(zip(self.X, self.Y)):
                    if i >= len(lab): break
                    vx, vy = valid_xy(x_arr, y_arr)
                    if len(vx) == 0: continue
                    
                    dy = inline_dy[i % len(inline_dy)]
                    
                    if align == 'right':
                        x_pos = vx[-1] - x_in_offset
                        ha = 'right'
                    else:
                        x_pos = vx[0] + x_in_offset
                        ha = 'left'
                        
                    color = self.COL[i] if i < len(self.COL) else 'black'
                    
                    self.ax.text(x_pos, vy[-1 if align=='right' else 0] + dy, 
                                 lab[i], color=color, ha=ha, va='center', 
                                 fontsize=lab_fs, fontweight='bold')
                
                inline_pad = kwargs.get('inline_pad', 0.05)
                if align == 'right':
                    self.ax.set_xlim(self.current_xmin, self.current_xmax + x_range * inline_pad)
                elif align == 'left':
                    self.ax.set_xlim(self.current_xmin - x_range * inline_pad, self.current_xmax)
            
            else:
                if len(self.sca) > 0:
                    self.ax.legend(self.sca, lab, bbox_to_anchor=(1.01, 1), 
                                   loc=loc, frameon=False, fontsize=lab_fs)

        if 'aspect' in kwargs:
            self.aspect = kwargs['aspect']
            
        if not is_3d:
            if isinstance(self.aspect, str): 
                self.ax.set_aspect(self.aspect)
            else:
                self.ax.set_aspect(self.aspect / self.ax.get_data_ratio(), adjustable="box")
                
        if 'vx' in kwargs:
            vx_list = kwargs['vx'] if isinstance(kwargs['vx'], (list, tuple, np.ndarray)) else [kwargs['vx']]
            vcol = kwargs.get('vcol', 'gray')
            vstyle = kwargs.get('vstyle', '--')
            vwidth = kwargs.get('vwidth', 1.0)
            for v in vx_list:
                self.ax.axvline(x=v, color=vcol, linestyle=vstyle, linewidth=vwidth, zorder=0)

        if 'hy' in kwargs:
            hy_list = kwargs['hy'] if isinstance(kwargs['hy'], (list, tuple, np.ndarray)) else [kwargs['hy']]
            hcol = kwargs.get('hcol', 'gray')
            hstyle = kwargs.get('hstyle', '--')
            hwidth = kwargs.get('hwidth', 1.0)
            for h in hy_list:
                self.ax.axhline(y=h, color=hcol, linestyle=hstyle, linewidth=hwidth, zorder=0)

        try:
            self.ax.figure.tight_layout()
        except RuntimeError as e:
            if "Adjustable 'box'" in str(e) or "twinned Axes" in str(e):
                for a in self.ax.figure.axes:
                    a.set_aspect('auto')
                self.ax.figure.tight_layout()
            else:
                raise e

        zoomx = kwargs.get('zoomx')
        zoomy = kwargs.get('zoomy')
        if zoomx is not None or zoomy is not None:
            self.add_inset_zoom(xlim=zoomx, ylim=zoomy, draw_lines=False)

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
        
        p0 = kwargs.get('p0', None)
        bounds = kwargs.get('bounds', (-np.inf, np.inf))
        auto_p0 = kwargs.get('auto_p0', False)
        n_trials = kwargs.get('n_trials', 100)

        for i, (x, y) in enumerate(zip(self.X, self.Y)):
            vx_, vy_ = valid_xy(x, y)
            if len(vx_) < 2: continue
            
            if callable(regr):
                try:
                    popt, pcov = auto_curve_fit(regr, vx_, vy_, p0=p0, bounds=bounds, 
                                                auto_p0=auto_p0, n_trials=n_trials)
                    err = np.sqrt(np.diag(pcov)) * 2 if not np.isinf(pcov).all() else [np.nan] * len(popt)
                    y_pred = regr(vx_, *popt)
                    r2 = r2_score(vy_, y_pred)
                    
                    df_rows.append([f"Data_{i}_Params"] + popt.tolist())
                    df_rows.append([f"Data_{i}_Error"] + err.tolist())
                    df_rows.append([f"Data_{i}_R2"] + [r2])
                    self.ax.plot(x_l, regr(x_l, *popt), color=self.COL[i], linestyle='--')
                except Exception as e:
                    print(f"Curve fit failed for Data_{i}: {e}")
                    
            elif isinstance(regr, int):
                if len(vx_) <= regr: continue
                fit, cov = np.polyfit(vx_, vy_, regr, cov=True)
                err = [cov[j][j]**0.5 * 2 for j in range(regr+1)]
                y_pred = reg_n(fit, vx_)
                r2 = r2_score(vy_, y_pred)
                
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
        
        if kwargs.get('nox', False) or kwargs.get('nonx', False): self.ax.tick_params(labelbottom=False)
        if kwargs.get('noy', False) or kwargs.get('nony', False): self.ax.tick_params(labelleft=False)

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

    def add_panel_label(self, text, x=-0.15, y=1.05, fontsize=None, weight='bold'):
        if fontsize is None:
            fontsize = self.axilab + 2
            
        self.ax.text(x, y, text, transform=self.ax.transAxes, 
                     fontsize=fontsize, fontweight=weight, 
                     va='bottom', ha='right')
        return self.ax

    def add_inset_zoom(self, xlim=None, ylim=None, bounds='auto', margin=0.02, draw_lines=False):
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
            _, y_fil = get_yrange(all_x, all_y, xlim[0], xlim[1])
            if len(y_fil) > 0:
                y_min, y_max = np.min(y_fil), np.max(y_fil)
                if is_logy:
                    log_min, log_max = np.log10(max(y_min, 1e-10)), np.log10(max(y_max, 1e-10))
                    dif = log_max - log_min if log_max != log_min else 1
                    ylim = [10**(log_min - dif*margin), 10**(log_max + dif*margin)]
                else:
                    dif = y_max - y_min if y_max != y_min else abs(y_min)*0.1
                    ylim = [y_min - dif*margin, y_max + dif*margin]
            else:
                ylim = self.ax.get_ylim()

        elif ylim is not None and xlim is None:
            x_fil, _ = get_xrange(all_x, all_y, ylim[0], ylim[1])
            if len(x_fil) > 0:
                x_min, x_max = np.min(x_fil), np.max(x_fil)
                if is_logx:
                    log_min, log_max = np.log10(max(x_min, 1e-10)), np.log10(max(x_max, 1e-10))
                    dif = log_max - log_min if log_max != log_min else 1
                    xlim = [10**(log_min - dif*margin), 10**(log_max + dif*margin)]
                else:
                    dif = x_max - x_min if x_max != x_min else abs(x_min)*0.1
                    xlim = [x_min - dif*margin, x_max + dif*margin]
            else:
                xlim = self.ax.get_xlim()
                
        elif xlim is not None and ylim is not None:
            pass 
            
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
        
        axins.minorticks_on()
        axins.tick_params(which='major', direction='in', length=self.tlength * 0.7, 
                          top=True, bottom=True, left=True, right=True, labelsize=self.axinum - 7)
        axins.tick_params(which='minor', direction='in', length=self.tlength * 0.35, 
                          top=True, bottom=True, left=True, right=True)

        if draw_lines:
            self.ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)
        
        return axins

    # ==========================================
    # 🌟 第二軸 (Twin Axes & Secondary Axes) 🌟
    # ==========================================
    def twinx(self, **kwargs):
        """第二Y軸 (右側) を作成して symple_plot インスタンスを返す。"""
        self.aspect = 'auto'            # 元の軸のアスペクト比を自動にする
        self.ax.set_aspect('auto')
        ax2 = self.ax.twinx()
        sp2 = symple_plot(ax2)
        sp2.aspect = 'auto'             # 追加する軸のアスペクト比も自動にする
        
        if alab := kwargs.get('alab'):
            ax2.set_ylabel(alab, fontsize=self.axilab)
        if col := kwargs.get('col'):
            ax2.spines['right'].set_color(col)
            ax2.tick_params(axis='y', colors=col)
            ax2.yaxis.label.set_color(col)
            sp2.col = col
        return sp2

    def twiny(self, **kwargs):
        """第二X軸 (上側) を作成して symple_plot インスタンスを返す。"""
        self.aspect = 'auto'
        self.ax.set_aspect('auto')
        ax2 = self.ax.twiny()
        sp2 = symple_plot(ax2)
        sp2.aspect = 'auto'
        
        if alab := kwargs.get('alab'):
            ax2.set_xlabel(alab, fontsize=self.axilab)
        if col := kwargs.get('col'):
            ax2.spines['top'].set_color(col)
            ax2.tick_params(axis='x', colors=col)
            ax2.xaxis.label.set_color(col)
            sp2.col = col
        return sp2

    def secondary_xaxis(self, functions, location='top', **kwargs):
        """スケール変換用の第二X軸を作成する。functions=(forward_func, inverse_func) を渡す。"""
        sec_ax = self.ax.secondary_xaxis(location, functions=functions)
        if alab := kwargs.get('alab'):
            sec_ax.set_xlabel(alab, fontsize=self.axilab)
        sec_ax.tick_params(labelsize=self.axinum)
        return sec_ax

    def secondary_yaxis(self, functions, location='right', **kwargs):
        """スケール変換用の第二Y軸を作成する。functions=(forward_func, inverse_func) を渡す。"""
        sec_ax = self.ax.secondary_yaxis(location, functions=functions)
        if alab := kwargs.get('alab'):
            sec_ax.set_ylabel(alab, fontsize=self.axilab)
        sec_ax.tick_params(labelsize=self.axinum)
        return sec_ax