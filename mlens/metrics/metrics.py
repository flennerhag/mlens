"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Scoring functions
"""

from __future__ import division, print_function

import numpy as np
from ..externals.sklearn.scorer import make_scorer


def score_matrix(y_preds, y_true, scorer, column_names=None, prefix=None):
    """DEPRECATED

    Function for scoring a matrix"""
    if column_names is None:
        column_names = ['preds_%i' % (i + 1) for i in range(y_preds.shape[1])]

    if prefix is None:
        return {col: scorer(y_true, y_preds[:, i]) for i, col in
                enumerate(column_names)}
    else:
        return {prefix + '-' + col: scorer(y_true, y_preds[:, i]) for i, col in
                enumerate(column_names)}


def set_scores(inst, layer_output, scorer_attribute_name='scorer'):
    """Utility function for generating score dict if a scorer was provided.

    Parameters
    ----------
    inst : instance
        instance to check for scorer attribute.

    layer_output : dict
        the ``dict`` output from a ``fit`` call of a ``LayerContainer`` class.

    scorer_attribute_name : str (default = 'scorer')
        name of the instance's attribute holding the scorer.

    Returns
    -------
    score_dict: dict
        a flattened dictionary of cross-validated scores across layers with
        keys formatted as 'layer-n-est_name'.
    """
    if getattr(inst, scorer_attribute_name, None) is None:
        return

    score_dict = {}
    for layer, out in layer_output.items():
        if out[0] is None:
            continue

        for key, score in out[0].items():
            score_dict['%s-%s' % (layer, key)] = score

    return score_dict


def rmse_scoring(y, p):
    """Root Mean Square Error := sqrt(mse), mse := (1/n) * sum((y-p)**2).

    Parameters
    ----------
    y : array-like
        ground truth.

    p : array-like
        predicted labels.

    Returns
    -------
    z: float
        root mean squared error.
    """
    z = y - p
    return np.mean(np.multiply(z, z)) ** (1 / 2)

rmse = make_scorer(rmse_scoring, greater_is_better=False)


def mape_scoring(y, p):
    """Mean Average Percentage Error := mean(abs((y - p) / y))."""
    return np.mean(np.abs((y - p) / y))


def mape_log_rescaled_scoring(y, p):
    """Log transform inputs y and p before calling ``mape_scoring``."""
    return mape_scoring(np.exp(y), np.exp(p))

mape = make_scorer(mape_scoring, greater_is_better=False)
mape_log = make_scorer(mape_log_rescaled_scoring, greater_is_better=False)


def wape_scoring(y, p):
    """Weighted Mean Average Percentage Error := sum(abs(y - p)) / sum(y)"""
    return np.sum(np.abs(y - p)) / np.sum(y)


def wape_log_rescaled_scoring(y, p):
    """Log transform"""
    return mape_scoring(np.exp(y), np.exp(p))

wape = make_scorer(wape_scoring, greater_is_better=False)
wape_log = make_scorer(wape_log_rescaled_scoring, greater_is_better=False)
