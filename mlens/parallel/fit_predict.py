#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
data: 10/01/2017
Base functions for any parallel processing
"""

from ..model_selection._cross_validate import fit_score
from ..ensemble._fit_predict import _fit_and_predict, _construct_matrix,
from ..ensemble._fit_predict import _fit_estimator
from pandas import DataFrame
from sklearn.externals.joblib import Parallel, delayed


def cross_validate(estimators, param_sets, X, y, cv, scoring, dout=None,
                   n_jobs=-1, verbose=False):
    """ Run parallellized cross-validated grid search on premade folds """

    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(fit_score)(est, est_name, params, scoring,
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

    columns = [case + '-' + est_name
               for case, estimators in estimator_cases.items()
               for est_name, _ in estimators]

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
