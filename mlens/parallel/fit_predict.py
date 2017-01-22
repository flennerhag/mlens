#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Base functions for any parallel processing
"""

from __future__ import division, print_function

from ._fit_predict_functions import _fit_score, _fit_and_predict, _predict
from ._fit_predict_functions import _fit_estimator, _construct_matrix
from pandas import DataFrame
from sklearn.externals.joblib import Parallel, delayed


def _pre_check_estimators(out, case_est_base_columns):
    try:
        case_est_names, _, _ = zip(*out)
    except ValueError:
        case_est_names, _ = zip(*out)

    return [ce for ce in case_est_base_columns if ce in case_est_names]


def _parallel_estimation(function, data, estimator_cases,
                         const=None, n_jobs=-1, verbose=False):
    """Backend function for estimator evaluation.

    Functions used for parallel estimation must accept only on argument,
    that the function itself unpacks.

    Parameters
    ----------
    function : obj
        function to be evaluated in parallel loop. Function should accept only
        one argument, a tuple for unpacking. The tuple is unpacked as one of:
            - data_tuple, estimator_info = tuple
            - const_tuple, data_tuple, estimator_info = tuple
        each tuple in turn can be furter unpacked if desired:
            (xtrain, xtest, ytest, ytrain, p_name), (est, est_name) = tuple
    data : list
        a list of lists, where the last element in each list is a key
        in the dict estimator_cases: [Xtrain [, Xtest, ytrain, ytest], key]
    estimator_cases : dict
        dictionary that maps preprocessing cases to a list of estimators to be
        fitted on the generated data
    const : tuple
        a tuple of optional arguments to be passed to function
    n_jobs : int
        level of parallellization
    verbose : int
        verbosity of paralellization process
    """
    inp = const if const is not None else tuple()

    return Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(function)(inp + (tup, est))
                   for tup in data
                   for est in estimator_cases[tup[-1]])


def base_predict(data, estimator_cases, n, folded_preds, columns, as_df=False,
                 n_jobs=-1, verbose=False):
    """Function for parallelized function fitting"""
    if folded_preds:
        # we only call folded prediction when fitting, so run fit_and_predict
        function = _fit_and_predict
    else:
        function = _predict

    out = _parallel_estimation(function, data, estimator_cases,
                               n_jobs=n_jobs, verbose=verbose)

    fitted_estimators = _pre_check_estimators(out, columns)
    M = _construct_matrix(out, n, fitted_estimators, folded_preds)

    if as_df:
        M = DataFrame(M, columns=fitted_estimators)

    return M, fitted_estimators


def fit_estimators(data, y, estimator_cases, n_jobs=-1, verbose=False):
    """Function for parallelized estimator fitting"""
    out = _parallel_estimation(_fit_estimator, data, estimator_cases, (y,),
                               n_jobs, verbose)

    fitted_estimators = {}
    for case, est_name, est in out:
        # Filter out unfitted models - these have case, est_name, est = None
        if case is not None:
            # Instantiate list
            if case not in fitted_estimators:
                fitted_estimators[case] = []

            fitted_estimators[case].append((est_name, est))

    return fitted_estimators


def cross_validate(estimators, param_sets, dout, scoring, error_score=-99,
                   n_jobs=-1, verbose=False):
    """Run parallellized cross-validated grid search on premade folds"""
    out = Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(_fit_score)(est, est_name, params, scoring,
                                       tup, i, error_score)
                   for tup in dout
                   for est_name, est in estimators.items()
                   for i, params in enumerate(param_sets[est_name]))
    return out
