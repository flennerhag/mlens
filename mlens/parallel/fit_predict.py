#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Base functions for any parallel processing
"""

from __future__ import division, print_function

from ._fit_predict_functions import (_fit_score, _fit_ests,
                                     _construct_matrix, _predict,
                                     _predict_folds, _fit_ests_folds)
from pandas import DataFrame
from joblib import Parallel, delayed


def _pre_check_estimators(out, case_est_base_columns):
    """Returns ordered list of the names of successfully fittest estimators"""
    try:
        case_est_names, _, _ = zip(*out)
    except ValueError:
        case_est_names, _ = zip(*out)

    return [ce for ce in case_est_base_columns if ce in case_est_names]


def _construct_estimator_dict(out):
    """Returns a dictionary of fitted estimators"""
    fitted_estimators = {}
    for key, est_name, est in out:
        # Filter out unfitted models
        if est_name is not None:
            # Add key
            if key not in fitted_estimators:
                fitted_estimators[key] = []

            fitted_estimators[key].append((est_name, est))
    return fitted_estimators


def _parallel_estimation(function, data, estimator_cases,
                         optional_args=None, n_jobs=-1, verbose=False):
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
    optional_args : tuple
        a tuple of optional arguments to be passed to function
    n_jobs : int
        level of parallellization
    verbose : int
        verbosity of paralellization process
    """
    optional_args = optional_args if optional_args is not None else tuple()

    return Parallel(n_jobs=n_jobs, verbose=verbose)(
                   delayed(function)(optional_args + (tup, est))
                   for tup in data
                   for est in estimator_cases[tup[-1]])


def base_predict(data, estimator_cases, n, folded_preds, fit, columns,
                 combine_keys, as_df=False, n_jobs=-1, verbose=False):
    """Generate a matrix M of predictions from m estimators

    Parameters
    -----------
    data: obj, list-like
        object to be passed to function for parallel estimation. Standard
        format is a nested list of inputs, i.e.
        [[Xtrain, Xtest, ytrain, ytest, test_idx, preprocess_case_name], ...]
    estimator_cases: dict
        dictionary that maps estimators on each preprocessing case.
        Each entry is a list of tuples, {'': [(est_name, est), ...], ...}
    n: int
        shape of test set
    folded_preds: bool
        whether predictions should be generated using the _predict_folds
        function. Otherwise the _predict function is used
    fit: bool
        whether estimators should be fitted before making predictions. Set to
        false if estimators already fitted.
    columns: list
        list of column names. Used to map prediction scores to estimators,
        and if as_df is True, to map columns headers.
    combine_keys: bool
        whether to use the last element in each list in `data` to create
        unique keys: i.e. each output tuple will have keys of the form
        `'preprocess_case_name-est_name keys'`. If set to False, each output
        tuple will have keys of the form `'est_name'`.
    as_df: bool
        whether the output matrix M should be returned as a pandas DataFrame
        columns names correspond to the estimator that generated the
        respective predictions
    n_jobs: int
        number of CPU cores to use for parallel estimation
    verbose: bool, int
        degree of printed messages

    Returns
    ---------
    M: array-like, shape=[n_samples, n_estimators]
        Matrix of estimator preditions. Either a numpy array of a pandas
        Dataframe.
    fitted_estimator_names: list
        list of estimator names for estimators with successfull predictions
        runs
    """
    # Determine prediction case
    if folded_preds:
        function = _predict_folds
    else:
        # Use estimators fitted on full training set to predict test set
        function = _predict

    out = _parallel_estimation(function, data, estimator_cases,
                               (fit, combine_keys),
                               n_jobs=n_jobs, verbose=verbose)

    fitted_estimator_names = _pre_check_estimators(out, columns)
    M = _construct_matrix(out, n, fitted_estimator_names, folded_preds)

    if as_df:
        M = DataFrame(M, columns=fitted_estimator_names)

    return M, fitted_estimator_names


def fit_estimators(data, estimator_cases, y, n_jobs=-1, verbose=False):
    """Function for parallelized estimator fitting

    Parameters
    -----------
    data: obj, list-like
        object to be passed to function for parallel estimation. Standard
        format is a nested list of training inputs for each preprocessing
        case, i.e. [[X_preprocessed_1, preprocess_case_name_1], ...]
    y: array-like, None
        the output values to train each preprocessed input set on.
        If None, assumes data is a list of training folds.
    estimator_cases: dict
        dictionary that maps estimators on each preprocessing case.
        Each entry is a list of tuples, {'': [(est_name, est), ...], ...}
    n_jobs: int
        number of CPU cores to use for parallel estimation
    verbose: bool, int
        degree of printed messages

    Returns
    ----------
    fitted_estimators: list
        list of fitted estimator instances
    """
    # Fit estimators
    if y is None:
        out = _parallel_estimation(_fit_ests_folds, data, estimator_cases,
                                   n_jobs=n_jobs, verbose=verbose)
    else:
        out = _parallel_estimation(_fit_ests, data, estimator_cases, (y,),
                                   n_jobs, verbose)
    return _construct_estimator_dict(out)


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
