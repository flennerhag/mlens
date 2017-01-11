#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
data: 10/01/2017
Base functions for any parallel processing
"""
from ._fit_predict_functions import _fit_score, _fit_and_predict
from ._fit_predict_functions import _fit_estimator, _construct_matrix
from ..utils import name_columns
from pandas import DataFrame
from sklearn.externals.joblib import Parallel, delayed


def cross_validate(estimators, param_sets, X, y, cv, scoring, dout=None,
                   n_jobs=-1, verbose=False):
    """ Run parallellized cross-validated grid search on premade folds """

    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(_fit_score)(est, est_name, params, scoring,
                                       *tup, i, True, True)
                   for tup in dout
                   for est_name, est in estimators.items()
                   for i, params in enumerate(param_sets[est_name]))
    return out


def folded_predictions(data, estimator_cases, n, as_df=False,
                       n_jobs=-1, verbose=False):
    """ Function for parallelized function fitting """

    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(_fit_and_predict)(tup, est, est_name)
                   for tup in data
                   for est_name, est in estimator_cases[tup[-1]])

    columns = name_columns(estimator_cases)

    M = _construct_matrix(out, n, columns)

    if as_df:
        M = DataFrame(M, columns=columns)

    return M


def fit_estimators(data, y, estimator_cases, n_jobs=-1, verbose=False):
    """ Function for parallelized function fitting """
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(_fit_estimator)(X, y, case, est_name, est)
                   for X, case in data
                   for est_name, est in estimator_cases[case])

    fitted_estimators = {case: [] for case in estimator_cases.keys()}
    for case, est_name, est in out:
        fitted_estimators[case].append((est_name, est))

    return fitted_estimators
