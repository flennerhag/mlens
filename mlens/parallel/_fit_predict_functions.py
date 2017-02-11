#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/1/2017
licence: MIT
Support function for parallelized fitting and prediction of estimators
"""

from __future__ import division, print_function

from functools import wraps
from time import time
import numpy as np
import warnings
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning


def _fit_score(est, est_name, params, scoring, tup, draw, error_score=-99):
    """Score an estimator with given parameters on train and test set"""
    try:
        xtrain, xtest, ytrain, ytest, p_name = tup
    except ValueError:
        xtrain, xtest, ytrain, ytest = tup
        p_name = None

    try:
        est = clone(est)
        est.set_params(**params)
        t0 = time()
        est = est.fit(xtrain, ytrain)

        t = time() - t0
        test_sc = scoring(est, xtest, ytest)
        train_sc = scoring(est, xtrain, ytrain)

    except Exception as e:
        msg = "Could not fit estimator [%s]. Score set to %s. Details: \n%r"
        warnings.warn(msg % (est_name, str(error_score), e), FitFailedWarning)

        t = 0
        train_sc, test_sc = error_score, error_score

    if p_name not in [None, '']:
        est_name += '-' + p_name

    return [est_name, test_sc, train_sc, t, draw + 1, params]


def _fit_estimator(tup):
    """Utility function for fitting estimator and logging its information"""
    y, (X, case), (est_name, estimator) = tup
    try:
        estimator = estimator.fit(X, y)
        return [case, est_name, estimator]
    except Exception as e:
        msg = "Estimator [%s] not fitted. Details: \n%r"
        warnings.warn(msg % (est_name, e), FitFailedWarning)
        return [None, None, None]


def _predict_folds(tup, fit):
    """Fits ests on part of training set to predict out of sample"""
    (xtrain, xtest, ytrain, _, idx, case), (est_name, estimator) = tup

    try:
        est = clone(estimator)
        if fit:
            est.fit(xtrain, ytrain)
        p = est.predict(xtest)
        out = [case + '-' + est_name] if case not in [None, ''] else [est_name]
        out += [idx, p]
        return out
    except Exception as e:
        msg = "Estimator [%s] not fitted. Details: \n%r"
        warnings.warn(msg % (est_name, e), FitFailedWarning)
        return [None, None, None]


def _predict_folds_decorator(fit):
    """Decorator for _predict_folds to alter fit argument"""
    def wrapper(tup):
        return _predict_folds(tup, fit)
    return wrapper


def _predict(tup):
    """Predicts on data using estimator"""
    (X, case), (est_name, estimator) = tup
    p = estimator.predict(X)
    out = [case + '-' + est_name] if case not in [None, ''] else [est_name]
    out.append(p)
    return out


def _construct_matrix(preds, n, columns, folds):
    """Helper function to assemble prediction matrix from prediction output"""
    colmap = {col: i for i, col in enumerate(columns)}

    M = np.empty((n, len(columns)))

    if folds:
        for (col, i, p) in preds:
            if col is not None:
                j = colmap[col]
                M[i, j] = p
    else:
        for (col, p) in preds:
            if col is not None:
                j = colmap[col]
                M[:, j] = p
    return M
