#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 10/01/2017
Support functions for fitting and predicting
"""
import numpy as np
import warnings
from ..utils import _slice
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning


def _fit_estimator(X, y, case, est_name, est):
    """ utlity function for fitting estimator and logging its information """
    try:
        est.fit(X, y)
        return [case, est_name, est]
    except Exception as e:
        msg = "Estimator [%s] not fitted. Details: \n%r"
        warnings.warn(msg % (est, e), FitFailedWarning)


def _construct_matrix(preds, n, columns):
    """ Helper function to assemble prediction matrix from prediction output"""
    colmap = {col: i for i, col in enumerate(columns)}

    M = np.empty((n, len(columns)))

    for tup in preds:
        col, i, p = tup
        j = colmap[col]
        M[i, j] = p

    return M


def _fit_and_predict(tup, estimator, est_name):
    """ Fits ests on part of training set to predict out of sample"""
    xtrain, xtest, ytrain, _, idx, case = tup

    try:
        est = clone(estimator)
        p = est.fit(xtrain, ytrain).predict(xtest)
        return [case + '-' + est_name, idx, p]
    except Exception as e:
        msg = "Estimator [%s] not fitted. Details: \n%r"
        warnings.warn(msg % (est, e), FitFailedWarning)
