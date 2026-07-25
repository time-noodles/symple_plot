from typing import Any, Callable, Iterator, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from matplotlib.collections import PathCollection

class symple_plot:
    ax: Axes
    alab_fs: int
    tick_fs: int
    tlength: int
    col: Any
    aspect: Any

    def __init__(self, ax: Axes) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def __getitem__(self, key: Any) -> 'symple_plot': ...
    def __iter__(self) -> Iterator['symple_plot']: ...
    def flatten(self) -> List['symple_plot']: ...
    
    def plot(
        self, X: Any, Y: Any, 
        alab: Optional[Union[List[str], str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        lab: Optional[Union[List[str], str]] = None,
        label: Optional[Union[List[str], str]] = None,
        cx: Optional[List[float]] = None,
        xlim: Optional[List[float]] = None,
        cy: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        col: Optional[Union[str, List[str]]] = None,
        color: Optional[Union[str, List[str]]] = None,
        linestyle: Optional[Union[str, List[str]]] = None,
        linewidth: Optional[float] = None,
        loc: Optional[str] = None,
        logx: bool = False,
        logy: bool = False,
        nox: bool = False,
        noy: bool = False,
        hide_xticks: bool = False,
        hide_yticks: bool = False,
        **kwargs: Any
    ) -> Axes: ...

    def scatter(
        self, X: Any, Y: Any, 
        alab: Optional[Union[List[str], str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        lab: Optional[Union[List[str], str]] = None,
        label: Optional[Union[List[str], str]] = None,
        cx: Optional[List[float]] = None,
        xlim: Optional[List[float]] = None,
        cy: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        col: Optional[Union[str, List[str]]] = None,
        color: Optional[Union[str, List[str]]] = None,
        marker: Optional[Union[str, List[str]]] = None,
        size: float = 40,
        hollow: bool = False,
        loc: Optional[str] = None,
        logx: bool = False,
        logy: bool = False,
        nox: bool = False,
        noy: bool = False,
        **kwargs: Any
    ) -> Axes: ...

    def pre_set(
        self, X: Any, Y: Any, 
        alab: Optional[Union[List[str], str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cx: Optional[List[float]] = None,
        xlim: Optional[List[float]] = None,
        cy: Optional[List[float]] = None,
        ylim: Optional[List[float]] = None,
        logx: bool = False,
        logy: bool = False,
        nox: bool = False,
        noy: bool = False,
        **kwargs: Any
    ) -> Axes: ...

    def tdscatter(
        self, X: Any, Y: Any, Z: Any, 
        alab: Optional[List[str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        zlabel: Optional[str] = None,
        cz: Optional[List[float]] = None,
        zlim: Optional[List[float]] = None,
        col: Optional[Union[str, List[str]]] = None,
        color: Optional[Union[str, List[str]]] = None,
        size: float = 40,
        logz: bool = False,
        **kwargs: Any
    ) -> Tuple[Axes, List[PathCollection]]: ...

    def tdplot(self, X: Any, Y: Any, Z: Any, **kwargs: Any) -> Tuple[Axes, List[Any]]: ...

    def imshow(
        self, X: Any, Y: Any, Z: Any, vmax: float, 
        alab: Optional[List[str]] = None,
        col: str = 'grads',
        logz: bool = False,
        **kwargs: Any
    ) -> Tuple[Axes, AxesImage]: ...

    def add_panel_label(self, text: str, x: float = ..., y: float = ..., fontsize: Optional[int] = ..., weight: str = ...) -> Axes: ...
    def twiny(self, **kwargs: Any) -> 'symple_plot': ...
    def twinx(self, **kwargs: Any) -> 'symple_plot': ...
    def secondary_yaxis(self, functions: Union[Callable, Tuple[Callable, Callable]], location: str = ..., **kwargs: Any) -> Axes: ...
    def secondary_xaxis(self, functions: Union[Callable, Tuple[Callable, Callable]], location: str = ..., **kwargs: Any) -> Axes: ...
    def Regression(self, regr: Union[int, Callable], directory: str = ..., **kwargs: Any) -> Axes: ...
    def add_inset_zoom(self, xlim: Optional[List[float]] = ..., ylim: Optional[List[float]] = ..., bounds: Union[str, List[float]] = ..., margin: float = ..., draw_lines: bool = ..., **kwargs: Any) -> Optional[Axes]: ...

def create_symple_plots(
    nrows: int = 1, ncols: int = 1, figsize: Optional[Tuple[float, float]] = None,
    style: Optional[str] = None, auto_label: bool = False, flush: bool = False, **kwargs: Any
) -> Tuple[plt.Figure, symple_plot]: ...

def set_style(mode: str = ...) -> None: ...