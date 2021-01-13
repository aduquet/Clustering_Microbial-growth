# Clustering_Microbial-growth
 yeasts are important cell factories for the transition towards more sustainable industrial processes. Their physiological characterization in bioreactors generates large amounts of data that is manually analyzed by the scientists to find patterns and cluster the growth into different phases.  The development of improved tools for efficient and accurate data analysis is essential for bioprocesses.  We want to present a tool that can organize, cluster, and derive meaningful patterns from the physiological characterization of microbial cell factories.

#Description

For organizing, clustering and driving meaningful patters from the physiological characterization of microbial cell factories.

Script `all_main.py` is designed to take 4 arguments in the following order:

* Data in csv format of the direct biomass
* Data in csv format of the biomass values based in absorbance from the online probe (ODa)
* A parameter 'd' who indicates how many files the user wants to process. Write to loading multiple files (dir) or (f) for loading just one file 
* A parameter 'o' who indicates the name in which the data output will be stored
 
# What do you need to run this code?

## Basic needs

`1. Anaconda`

Anaconda is a distribution of the Python and R programming languages for scientific computing (data science, machine learning applications, large-scale data processing, predictive analytics, etc.), that aims to simplify package management and deployment. The distribution includes data-science packages suitable for Windows, Linux, and macOS.

To install please go to: https://www.anaconda.com/products/individual#Downloads

`2. Python`
 
 :D (We are using Python 3.8.3)
 
`3. python pip`

pip is the package installer for Python. You can use pip to install packages from the Python Package Index and other indexes.

To install please go to: https://pypi.org/project/pip/

## Prerequisities

* `click`, Click is a Python package for creating beautiful command line interfaces in a composable way with as little code as necessary. It’s the “Command Line Interface Creation Kit”. It’s highly configurable but comes with sensible defaults out of the box.

* `os`, This module provides a portable way of using operating system dependent functionality

* `glob`, The glob module finds all the pathnames matching a specified pattern according to the rules used by the Unix shell, although results are returned in arbitrary order. No tilde expansion is done, but *, ?, and character ranges expressed with [] will be correctly matched.

* `numpy`, NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.  

* `sklearn`, scikit-learn is a Python module for machine learning built on top of SciPy and is distributed under the 3-Clause BSD license.

* `scipy`, SciPy (pronounced “Sigh Pie”) is a Python-based ecosystem of open-source software for mathematics, science, and engineering

* `matplotlib`, Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

Use `pip install name of module or library` to install them.
 
##Using the Script for complete execution

To get a complete execution using just one file, follow the format

`python all_main.py -bio <PATH OF CSV FILE WITH BIOMASS> -ODa <PATH OF CSV FILE ODa> -d f -o <FILE NAME IN WICH WILL BE STORED THE DATA>`

To get a complete execution using just one file, follow the format

`python all_main.py -bio <PATH OF MAIN DIRECTORY WITH MULTIPLE csv BIOMASS FILES/*> -ODa <PATH OF MAIN DIRECTORY WITH MULTIPLE csv ODa FILES/*> -d dir -o <FILE NAME IN WICH WILL BE STORED THE DATA>`

You can also write `python all_main.py --help` to get some help :D 


