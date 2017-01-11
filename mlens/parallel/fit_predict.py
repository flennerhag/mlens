#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
data: 10/01/2017
Base functions for any parallel processing
"""
from ._fit_predict_functions import _fit_score, _fit_and_predict, _predict
from ._fit_predict_functions import _fit_estimator, _construct_matrix
from pandas import DataFrame
from sklearn.externals.joblib import Parallel, delayed


def _parallel_estimation(function, data, estimator_cases,
                         y=None, n_jobs=-1, verbose=False):
    """ Backend function for estimator evaluation """
    return Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(function)(tup, est, est_name, y)
                   for tup in data
                   for est_name, est in estimator_cases[tup[-1]])


def base_predict(data, estimator_cases, n, folded_preds, columns, as_df=False,
                 n_jobs=-1, verbose=False):
    """ Function for parallelized function fitting """
    if folded_preds:
        # we only call folded prediction when fitting, so run fit_and_predict
        function = _fit_and_predict
    else:
        function = _predict

    out = _parallel_estimation(function, data, estimator_cases,
                               n_jobs=n_jobs, verbose=verbose)

    M = _construct_matrix(out, n, columns, folded_preds)

    if as_df:
        M = DataFrame(M, columns=columns)
    return M


def fit_estimators(data, y, estimator_cases, n_jobs=-1, verbose=False):
    """ Function for parallelized estimator fitting """

    out = _parallel_estimation(_fit_estimator, data, estimator_cases, y,
                               n_jobs, verbose)

    fitted_estimators = {case: [] for case in estimator_cases.keys()}
    for case, est_name, est in out:
        fitted_estimators[case].append((est_name, est))

    return fitted_estimators


def cross_validate(estimators, param_sets, dout, cv, scoring,
                   n_jobs=-1, verbose=False):
    """ Run parallellized cross-validated grid search on premade folds """

    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(_fit_score)(est, est_name, params, scoring,
                                       *tup, i, True, True)
                   for tup in dout
                   for est_name, est in estimators.items()
                   for i, params in enumerate(param_sets[est_name]))
    return out
