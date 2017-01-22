#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML-ENSEMBLE
author: Sebastian Flennerhag
date: 11/01/2017
Scoring functions not in the sklearn library
"""

from __future__ import division, print_function

import numpy as np
from pandas import DataFrame
from sklearn.metrics import make_scorer


def score_matrix(M, y, score, column_names=None):
    """Function for scoring a matrix"""
    if isinstance(M, DataFrame):
        return dict(M.apply(lambda x: score(y, x)))
    else:
        return {col: score(y, M[:, i]) for i, col in enumerate(column_names)}


def rmse_scoring(y, p):
    """Root Mean Square Error := sqrt(mse), mse := (1/n) * sum( (y-p)**2 )"""
    return np.mean((y-p)**2)**(1/2)

rmse = make_scorer(rmse_scoring, greater_is_better=False)


def mape_scoring(y, p):
    """Mean Average Percentage Error := mean( abs( (y - p) / y) )"""
    return np.mean(np.abs((y-p)/y))


def mape_log_rescaled_scoring(y, p):
    """Log transform"""
    return mape_scoring(np.exp(y), np.exp(p))

mape = make_scorer(mape_scoring, greater_is_better=False)
mape_log = make_scorer(mape_log_rescaled_scoring, greater_is_better=False)


def wape_scoring(y, p):
    """Weighted Mean Average Percentage Error := mean( abs( (y - p) / y) )"""
    return np.sum(np.abs(y-p)) / np.sum(y)


def wape_log_rescaled_scoring(y, p):
    """Log transform"""
    return mape_scoring(np.exp(y), np.exp(p))

wape = make_scorer(wape_scoring, greater_is_better=False)
wape_log = make_scorer(wape_log_rescaled_scoring, greater_is_better=False)
