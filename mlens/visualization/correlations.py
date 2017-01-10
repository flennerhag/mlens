#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
Correlation plots
"""

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from seaborn import diverging_palette, heatmap
import numpy as np
from scipy.stats import pearsonr


def corrmat(corr, figsize=(11, 9), annotate=True,
            linewidths=.5, cbar_kws={"shrink": .5}, **kwargs):
    '''
    Function for generating color-coded correlation triangle.

    Parameters
    ----------
    corr : array-like, shape = [n_features, n_features]
        Input correlation matrix. Use pandas DataFrame for axis labels
    figsize : tuple, shape (int, int)
        Size of figure
    linewidths : float
        with of line separating color points
    cbar_kws : dict
        Optional arguments to color bar. Shrink is preset to fit standard
        figure frame.
    kwargs : kwargs
        Other pptional arguments to sns heatmap

    Returns
    --------
    heatmap : obj
        Matplotlib figure containing heatmap
    '''
    # Determine annotation
    do_annot = {True: corr*100, False: None}
    annot = do_annot[annotate]

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    heatmap(corr, mask=mask, cmap=cmap, vmin=corr.min().min(),
            annot=annot, fmt='2.0f',
            vmax=corr.max().max(), square=True, linewidths=linewidths,
            cbar_kws=cbar_kws, ax=ax, **kwargs)
    return ax


def clustered_corrmap(X, cls=None, label_attr_name='labels_',
                      fig_size=(20, 20), title_fontsize=24,
                      title_name='Feature correlation heatmap', **kwargs):
    '''
    Function for plotting a clustered correlation heatmap

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        dataset to plot
    cls : obj
        cluster object that accepts fit and stores labels as attribute
    label_attr_name : str
        name of attribute that contains cluster label
    '''

    # find closely associated features
    fa = cls(**kwargs)
    fa.fit(X.corr())

    # sort features on cluster membership
    corr_list = [tup[0] for tup in sorted(zip(X.columns.tolist(),
                                              getattr(fa, label_attr_name)),
                                          key=lambda x: x[1])]
    plt.figure(figsize=fig_size)
    heatmap(X.loc[:, corr_list].corr(), vmax=1.0, square=True)
    plt.title(title_name, fontsize=title_fontsize)
    plt.show()


def corr_X_y(X, y, top=25):
    '''
    Function for plotting how features in an input matrix X correlates with
    y. Output figure shows all correlations as well as top pos and neg.

    Parameters
    ----------
    top : int
        number of features to show in top pos and neg graphs

    Returns
    -------
    f, gs : object, object
        figure, axis_grid
    '''

    correls = X.apply(lambda x: pearsonr(x, y)[0]).sort_values(ascending=False)

    f = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2)

    ax0 = plt.subplot(gs[0, 0])
    ax0.bar(range(top), correls.iloc[:top], align='center')
    ax0.set_xlim(-0.5, top)
    ax0.set_title('Top %i positive pairwise correlation coefficients' % top,
                  fontsize=16)
    plt.xticks(range(top), correls.index.tolist()[:top], rotation=90)

    ax1 = plt.subplot(gs[0, 1])
    ax1.bar(range(top), correls.iloc[-top:], align='center')
    ax1.set_xlim(-1, top+1)
    ax1.set_title('Top %i negative pairwise correlation coefficients' % top,
                  fontsize=16)
    plt.xticks(range(top), correls.index.tolist()[-top:], rotation=90)

    ax2 = plt.subplot(gs[1, :])
    ax2.bar(range(len(correls)), correls, align='center')
    ax2.set_xlim(-1, len(correls)+1)
    ax2.set_title('Pairwise correlation coefficients', fontsize=16)

    plt.tight_layout()
    plt.show()

    return f, gs
