#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
"""

from __future__ import division, print_function

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.decomposition import KernelPCA


def IO_plot_comp(X, y, figsize=(10, 8)):
    '''
    Function for comparing PCA analysis across 2 and 3 dimensions and
    linear and rbf kernels. See IO_plot for details.
    '''
    comp = ['linear', 'rbf']
    f = plt.figure(figsize=figsize)
    ax = []
    kwarg = {}

    for dim, frame in [(2, 221), (3, 223)]:
        if dim is 3:
            kwarg = {'projection': '3d'}
        for i, kr in enumerate(comp):
            ax.append(f.add_subplot(frame + i, **kwarg))
            ax[-1] = IO_plot(X, y, KernelPCA(dim, kr), ax=ax[-1], show=False)
            ax[-1].set_title(kr + ', ' + str(dim) + 'd')
    return f, ax


def IO_plot(X, y, estimator, figsize=(10, 8), show=True, ax=None):
    '''
    Function to plot a PCA analysis of 1, 2, or 3 dims.

    Input:
    ---------
    X : array-like
        Input matrix
    y : array-like
        label vector
    estimator : object
        PCA estimator, not initiated, assumes an sklearn API.
    dim : int <= 3
        subspace dimensions.

    Returns:
        shows figure
    '''

    Z = estimator.fit_transform(X.values)

    # prep figure if no axis supplied
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
        if estimator.n_components == 3:
            ax = f.add_subplot(111, projection='3d')

    if estimator.n_components == 1:
        ax.scatter(Z, c=y)
    elif estimator.n_components == 2:
        ax.scatter(Z[:, 0], Z[:, 1], c=y)
    elif estimator.n_components == 3:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y)
    else:
        raise ValueError('dim too large to visualize.')
    if show:
        plt.show()
    return ax


def exp_var_plot(X, estimator, figsize=(10, 8), show=True, ax=None):
    '''
    Function to plot the explained variance using PCA.

    Input:
    ---------
    X : array-like
        Input matrix
    estimator : object
        PCA estimator, assumes an sklearn API.

    Returns:
        shows figure
    '''
    estimator.set_params(**{'n_components': None})
    ind_var_exp = estimator.fit(X).explained_variance_ratio_
    cum_var_exp = np.cumsum(ind_var_exp)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.step(range(len(ind_var_exp)), cum_var_exp, where='mid')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(ind_var_exp))
    plt.title('Explained variance ratio', fontsize=18)
    if show:
        plt.show()
    if ax is not None:
        return ax
