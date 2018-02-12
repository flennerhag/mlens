"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Explained variance plots.
"""

from __future__ import division, print_function

import numpy as np

import warnings

try:
    from pandas import DataFrame, Series
except ImportError:
    DataFrame = Series = None
    warnings.warn("Pandas not installed. Visualization module may not work "
                  "as intended.", ImportWarning)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.decomposition import KernelPCA
    from matplotlib.colors import ListedColormap
    from seaborn import color_palette
except:
    warnings.warn("Matplotlib and Seaborn not installed. Cannot load "
                  "visualization module.", ImportWarning)


def pca_comp_plot(X, y=None, figsize=(10, 8),
                  title='Principal Components Comparison', title_font_size=14,
                  show=True, **kwargs):
    """Function for comparing PCA analysis.

    Function compares across 2 and 3 dimensions and
    linear and rbf kernels.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        input matrix to be used for prediction.

    y : array-like of shape = [n_samples, ] or None (default = None)
        training labels to be used for color highlighting.

    figsize : tuple (default = (10, 8))
        Size of figure.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    show : bool (default = True)
        whether to print figure :obj:`matplotlib.pyplot.show`.

    **kwargs : optional
        optional arguments to pass to :class:`mlens.visualization.pca_plot`.

    Returns
    -------
    ax :
        axis object.

    See Also
    --------
    :class:`mlens.visualization.pca_plot`
    """
    comp = ['linear', 'rbf']
    f = plt.figure(figsize=figsize)

    ax = []
    subplot_kwarg = {}

    for dim, frame in [(2, 221), (3, 223)]:

        if dim is 3:
            # Need to specify projection
            subplot_kwarg = {'projection': '3d'}

        for i, kernel in enumerate(comp):
            # Create subplot
            ax.append(f.add_subplot(frame + i, **subplot_kwarg))

            # Build figure
            ax[-1] = pca_plot(X, KernelPCA(dim, kernel), y,
                              ax=ax[-1], show=False, **kwargs)

            ax[-1].set_title('%s kernel, %i dims' % (kernel, dim))

            # Whiten background if dim is 3
            if dim is 3:
                ax[-1].set_facecolor((1, 1, 1))

    if show:
        f.suptitle(title, fontsize=title_font_size)
        plt.show()

    return ax


def pca_plot(X, estimator, y=None, cmap=None, figsize=(10, 8),
             title='Principal Components Analysis', title_font_size=14,
             show=True, ax=None, **kwargs):
    """Function to plot a PCA analysis of 1, 2, or 3 dims.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        matrix to perform PCA analysis on.

    estimator : instance
        PCA estimator. Assumes a Scikit-learn API.

    y : array-like of shape = [n_samples, ] or None (default = None)
        training labels to be used for color highlighting.

    cmap : object, optional
        cmap object to pass to :obj:`matplotlib.pyplot.scatter`.

    figsize : tuple (default = (10, 8))
        Size of figure.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    show : bool (default = True)
        whether to print figure :obj:`matplotlib.pyplot.show`.

    ax : object, optional
        axis to attach plot to.

    **kwargs : optional
        arguments to pass to :obj:`matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : optional
        if ``ax`` was specified, returns ``ax`` with plot attached.
    """
    Z = X.values if isinstance(X, (DataFrame, Series)) else X

    Z = estimator.fit_transform(Z)

    # prep figure if no axis supplied
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
        if estimator.n_components == 3:
            ax = f.add_subplot(111, projection='3d')

    if cmap is None:
        cmap = ListedColormap(color_palette('husl'))

    if estimator.n_components == 1:
        ax.scatter(Z, c=y, cmap=cmap, **kwargs)
    elif estimator.n_components == 2:
        ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap=cmap, **kwargs)
    elif estimator.n_components == 3:
        ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=y, cmap=cmap, **kwargs)
    else:
        raise ValueError("'n_components' is too large to visualize. "
                         "Set to one of [1, 2, 3].")

    if show:
        plt.title(title, fontsize=title_font_size)
        plt.show()

    return ax


def exp_var_plot(X, estimator, figsize=(10, 8), buffer=0.01, set_labels=True,
                 title='Explained variance ratio', title_font_size=14,
                 show=True, ax=None, **kwargs):
    """Function to plot the explained variance using PCA.

    Parameters
    ----------
    X : array-like of shape = [n_samples, n_features]
        input matrix to be used for prediction.

    estimator : class
        PCA estimator, not initiated, assumes a Scikit-learn API.

    figsize : tuple (default = (10, 8))
        Size of figure.

    buffer : float (default = 0.01)
        For creating a buffer around the edges of the graph. The buffer
        added is calculated as ``num_components`` * ``buffer``,
        where ``num_components`` determine the length of the x-axis.

    set_labels : bool
        whether to set axis labels.

    title : str
        figure title if shown.

    title_font_size : int
        title font size.

    show : bool (default = True)
        whether to print figure using :obj:`matplotlib.pyplot.show`.

    ax : object, optional
        axis to attach plot to.

    **kwargs : optional
        optional arguments passed to the :obj:`matplotlib.pyplot.step`
        function.

    Returns
    -------
    ax : optional
        if ``ax`` was specified, returns ``ax`` with plot attached.
    """
    estimator.set_params(**{'n_components': None})
    ind_var_exp = estimator.fit(X).explained_variance_ratio_
    cum_var_exp = np.cumsum(ind_var_exp)
    x = range(1, len(ind_var_exp) + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.step(x, cum_var_exp, **kwargs)

    buffer_ = buffer * len(ind_var_exp)
    ax.set_ylim(0, 1 + buffer_ / 3)
    ax.set_xlim(1 - buffer_, len(ind_var_exp) + buffer_)
    ax.set_xticks([i for i in x])

    if set_labels:
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained variance ratio')

    if show:
        plt.title(title, fontsize=title_font_size)
        plt.show()

    return ax
