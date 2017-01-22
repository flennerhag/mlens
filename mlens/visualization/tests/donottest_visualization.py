#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML-ENSEMBLE
author: Sebastian Flennerhag
date: 13/01/2017
"""

import numpy as np
from pandas import DataFrame, Series
from mlens.visualization import corrmat, clustered_corrmap, corr_X_y
from mlens.visualization import IO_plot_comp,  IO_plot, exp_var_plot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# training data
np.random.seed(100)
X = np.random.random((1000, 10))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

X = DataFrame(X)
y = Series(y)


def test_corrmat():
    out = corrmat(X.corr(), figsize=(11, 9), annotate=True,
                  linewidths=.5, cbar_kws={"shrink": .5})
    assert out is not None


def test_clustered_corrmat():
    clustered_corrmap(X, KMeans, show=False)


def test_corr_X_y():
    out = corr_X_y(X, y, show=False)
    assert out is not None


def test_IO_plot_comp():
    out = IO_plot_comp(X, y)
    assert out is not None


def test_IO_plot():
    IO_plot(X, y, PCA(n_components=2), show=False)


def test_exp_var_plot():
    exp_var_plot(X, PCA(), show=False)
