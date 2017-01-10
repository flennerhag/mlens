#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 22:57:30 2016

@author: Sebastian Flennerhag
@kaggle: California Housing datase
@content: library with models for building ensemble
"""

# ===================== Dependencies =====================
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.decomposition import KernelPCA
from scipy.stats import pearsonr


# ===================== Data plots =====================
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
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=corr.min().min(),
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
    figure(figsize=fig_size)
    heatmap(X.loc[:, corr_list].corr(), vmax=1.0, square=True)
    title(title_name, fontsize=title_fontsize)
    show()


# ===================== PCA reduction =====================
def IO_plot_comp(X, y, figsize=(10, 8)):
    comp=['linear', 'rbf']
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
            ax = f.add_subplot(111, projection = '3d')

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
        f, ax = plt.subplots(figsize=figsize)
    ax.step(range(len(ind_var_exp)), cum_var_exp, where='mid')
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(ind_var_exp))
    plt.title('Explained variance ratio', fontsize=18)
    if show:
        plt.show()
    if ax is not None:
        return ax

def corr_X_y(X, y, top=25):
    correls = X.apply(lambda x: pearsonr(x, y)[0]).sort_values(ascending=False)
    
    
    f = plt.figure(figsize=(12,8))
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
