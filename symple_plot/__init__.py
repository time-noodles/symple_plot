# symple_plot/__init__.py

from .plotter import symple_plot, create_symple_plots, set_style, get_grads_cmap
from .data_utils import valid_xy, get_yrange, get_xrange, pad_list, remove_background, list_1d, f_peak
from .file_utils import straighten_path, del_file
from .fit_utils import auto_curve_fit, reg_n

__all__ = [
    'symple_plot', 'create_symple_plots', 'set_style', 'get_grads_cmap',
    'valid_xy', 'get_yrange', 'get_xrange', 'pad_list',
    'list_1d', 'f_peak',
    'remove_background',
    'straighten_path', 'del_file',
    'auto_curve_fit', 'reg_n'
]