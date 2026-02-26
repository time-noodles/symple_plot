from setuptools import setup, find_packages

setup(
    name="symple_plot",
    version="0.1.3.4",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "scikit-learn"
    ]
)