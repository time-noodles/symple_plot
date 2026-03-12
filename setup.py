from setuptools import setup, find_packages

setup(
    name="symple_plot",
    version="0.1.3.17",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "scipy",
    ]
)

from symple_plot import create_symple_plots, set_style, symple_plot, del_file
fig6, sp_arr6 = create_symple_plots(1, 3, figsize=(15, 4))
sp_arr6[0].plot([1, 2, 3], [1, 4, 9])