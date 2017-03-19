"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Tools for plotting ensemble benchmarks.
"""

from __future__ import division, print_function

import numpy as np

import matplotlib.pyplot as plt
from seaborn import diverging_palette, heatmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA
from matplotlib.colors import ListedColormap
from seaborn import color_palette


def diagnose(est, **kwargs):
    """Plot the time to fit, the average CPU usage, and peak memory usage.

    Parameters
    ----------
    est : instance
        estimator instance to benchmark

    **kwargs : optional
        optional arguments to pass to the make_regression function.
    """

