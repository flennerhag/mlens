#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/01/2017
Scoring functions
"""

from __future__ import division, print_function

import numpy as np
from pandas import DataFrame
from ..externals import make_scorer


def score_matrix(y_preds, y_true, scorer, column_names=None, prefix=None):
    """Function for scoring a matrix"""
    y_preds = y_preds if not isinstance(y_preds, DataFrame) else y_preds.values

    if column_names is None:
        column_names = ['preds_%i' % (i + 1) for i in range(y_preds.shape[1])]

    if prefix is None:
        return {col: scorer(y_true, y_preds[:, i]) for i, col in enumerate(column_names)}
    else:
        return {prefix + '-' + col: scorer(y_true, y_preds[:, i]) for i, col in
                enumerate(column_names)}


def rmse_scoring(y, p):
    """Root Mean Square Error := sqrt(mse), mse := (1/n) * sum((y-p)**2)

    Parameters
    ----------
    y : array-like
        ground truth
    p : array-like
        predicted labels

    Returns
    ---------
    z: float
        root mean squared error
    """
    return np.mean((y-p)**2)**(1/2)

rmse = make_scorer(rmse_scoring, greater_is_better=False)


def mape_scoring(y, p):
    """Mean Average Percentage Error := mean(abs((y - p) / y))"""
    return np.mean(np.abs((y-p)/y))


def mape_log_rescaled_scoring(y, p):
    """Log transform"""
    return mape_scoring(np.exp(y), np.exp(p))

mape = make_scorer(mape_scoring, greater_is_better=False)
mape_log = make_scorer(mape_log_rescaled_scoring, greater_is_better=False)


def wape_scoring(y, p):
    """Weighted Mean Average Percentage Error := sum(abs(y - p)) / sum(y)"""
    return np.sum(np.abs(y-p)) / np.sum(y)


def wape_log_rescaled_scoring(y, p):
    """Log transform"""
    return mape_scoring(np.exp(y), np.exp(p))

wape = make_scorer(wape_scoring, greater_is_better=False)
wape_log = make_scorer(wape_log_rescaled_scoring, greater_is_better=False)
