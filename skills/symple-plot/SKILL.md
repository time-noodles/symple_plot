---
name: symple-plot
description: Guidelines and API specifications for using `symple_plot`, a Matplotlib wrapper library designed for publication and slide figures. Use when the user requests generating plots, charts, graphs, or data visualizations with `symple_plot` or asks for publication/presentation-ready scientific figures.
---

# `symple_plot` Usage Guide for AI Agents

`symple_plot` is a high-level Matplotlib wrapper that generates publication- and presentation-grade figures with minimal code. **It accepts both standard Matplotlib parameter names (`xlabel`, `ylabel`, `label`, `xlim`, `ylim`, `color`) and compact short-names (`alab`, `lab`, `cx`, `cy`, `col`)**.

## Key Concepts & Design Philosophy
1. **Matplotlib Parameter Compatibility**: Use standard names (`xlabel="X"`, `ylabel="Y"`, `color="blue"`, `label="Data"`) or short names (`alab=["X", "Y"]`, `col="blue"`).
2. **Direct Axes Passthrough**: You can directly call Matplotlib Axes methods on `sp` (e.g. `sp.set_title()`, `sp.grid()`).
3. **List Batching**: Pass lists of arrays `[x1, x2]`, `[y1, y2]` to plot multiple datasets in one shot.
4. **Smart Formatting**: Automatic inward ticks, scientific notation, and paper/slide themes.

## Standard Usage Pattern

```python
import numpy as np
import matplotlib.pyplot as plt
from symple_plot import create_symple_plots

# Data preparation
x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)

# 1. Create figure and panel wrappers
# Use style='paper' for papers, style='slide' for presentation slides
fig, sp = create_symple_plots(nrows=1, ncols=1, figsize=(6, 4), style='paper')

# 2. Plotting (Standard Matplotlib parameters work out of the box!)
sp.plot(
    [x, x], [y1, y2], 
    xlabel="Time (s)", 
    ylabel="Amplitude (a.u.)", 
    label=["Signal A", "Signal B"], 
    color=["#1f77b4", "#ff7f0e"],
    linestyle=["-", "--"]
)

# 3. Direct Axes methods passthrough
sp.set_title("Experimental Results")
sp.grid(True)

plt.show()
```

## Parameter Translation

| Goal | Standard Matplotlib Parameter | Compact Alias | Example |
| :--- | :--- | :--- | :--- |
| Set Axis Labels | `xlabel="X"`, `ylabel="Y"` | `alab=["X", "Y"]` | `xlabel="Time (s)"` |
| Set Legend Text | `label="Name"` | `lab="Name"` | `label="Dataset 1"` |
| Set Crop Limits | `xlim=[min, max]`, `ylim=[min, max]` | `cx=[min, max]`, `cy=[min, max]` | `xlim=[0, 10]` |
| Set Color | `color="red"` / `c="red"` | `col="red"` | `color="blue"` |
| Set Line Style / Width | `linestyle="-"` / `ls="-"`, `linewidth=2` / `lw=2` | same | `linestyle="--"` |
| Set Legend Position | `loc` | `loc` | `loc='upper right'` or `loc='inline'` |
| Hide Tick Labels | `hide_xticks=True`, `hide_yticks=True` | `nox=True`, `noy=True` | `hide_xticks=True` |
| Hollow Scatter Markers | `hollow=True` (scatter) | same | `sp.scatter(x, y, hollow=True)` |

## Direct Axes Method Support
Calling Axes methods directly on `sp` is fully supported:
- `sp.set_title(...)`
- `sp.grid(...)`
- `sp.set_yscale('log')`
- `sp.annotate(...)`

## Constraints & Best Practices
- ✅ Prefer using `create_symple_plots(style='paper')` or `style='slide'` for consistent formatting.
- ✅ Pass lists `[x1, x2]`, `[y1, y2]` to `sp.plot()` instead of writing verbose loops.
