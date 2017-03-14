"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Functions for processing an ensemble layer
"""

from __future__ import division, print_function

from ..base import check_fit_overlap
from ..base import clone_base_estimators, clone_preprocess_cases
from ..base import name_columns
from ..metrics import score_matrix
from ..parallel import (preprocess_folds, preprocess_pipes,
                        fit_estimators, base_predict)
import sys


def _layer_preprocess(X, y, parallel, layer_preprocess, method_is_fit, n_jobs,
                      verbose):
    """Method for generating predictions for inputs."""
    if (layer_preprocess is None) or (len(layer_preprocess) == 0):
        return [[X, '']], None
    else:

        out = preprocess_pipes(layer_preprocess, X, y,
                               parallel, fit=method_is_fit,
                               return_estimators=method_is_fit,
                               n_jobs=n_jobs, verbose=verbose)
        if method_is_fit:
            pipes, Z, cases = zip(*out)
            fitted_prep = [(case, pipe) for case, pipe in
                           zip(cases, pipes)]
            return [[z, case] for z, case in zip(Z, cases)], fitted_prep
        else:
            return [[z, case] for z, case in out], None


def _gen_in_layer(layer, X, y, parallel, folds, shuffle, random_state, scorer,
                  as_df, folded_preds, n_jobs, printout, verbose,
                  layer_msg=''):
    """Generate training data layer.

    Function for generating training data for next layer from an ingoing layer.
    """
    preprocess = clone_preprocess_cases(layer.preprocessing)
    estimators = clone_base_estimators(layer.estimators)
    columns = name_columns(estimators)

    Min = preprocess_folds(preprocess, X, y, parallel, folds=folds, fit=True,
                           shuffle=shuffle, random_state=random_state,
                           n_jobs=n_jobs, verbose=verbose)

    M, est_names = base_predict(Min, estimators, parallel, n=X.shape[0],
                                folded_preds=folded_preds,
                                function_args=(True, True), columns=columns,
                                as_df=as_df, n_jobs=n_jobs, verbose=verbose)

    if scorer is not None:
        cols = est_names
        scores = score_matrix(M, y, scorer, cols, layer_msg)
    else:
        scores = None

    return M, scores, est_names


def _fit_layer_estimators(layer, X, y, parallel, n_jobs, printout, verbose):
    """Fits preprocessing pipelines and layer estimator on full data set."""
    preprocess, estimators = layer.preprocessing, layer.estimators

    Min, preprocessing = \
        _layer_preprocess(X, y, parallel, clone_preprocess_cases(preprocess),
                          True, n_jobs, verbose)

    return (fit_estimators(Min, clone_base_estimators(estimators), y,
                           parallel, n_jobs, verbose), preprocessing)


def fit_layer(layer, X, y, parallel, folds, shuffle, random_state, scorer,
              as_df, folded_preds, n_jobs, printout, verbose, layer_msg=''):
    """Fit ensemble layer.

    Function for fitting a layer and generating training data for next layer.
    `fit_layer` starts by generating the layer's predictions, M, by temporarily
    fitting the layer's preprocessing pipes and estimators on folds. It then
    fits the final preprocessing pipes and estimators on the full training set.
    """
    M, scores, est_names = \
        _gen_in_layer(layer, X, y, parallel, folds, shuffle, random_state,
                      scorer, as_df, folded_preds, n_jobs, printout, verbose,
                      layer_msg)

    fitted_estimators, fitted_preprocessing = \
        _fit_layer_estimators(layer, X, y, parallel, n_jobs, printout, verbose)

    # Check that success in folded fits overlap with success in full fit
    fitted_est_names = name_columns(fitted_estimators)
    check_fit_overlap(fitted_est_names, est_names, layer_msg)

    return fitted_estimators, fitted_preprocessing, (M, scores)


def predict_layer(layer, X, y, parallel, as_df, n_jobs, verbose, printout=None,
                  layer_msg=None):
    """Predict ensemble layer.

    Function for predicting a layer and generating training data for next
    layer. Predict layer wraps a preprocessing step and a prediction step into
    a single function call, and returns the prediction matrix M.
    """
    columns = name_columns(layer.estimators_)

    if verbose:
        print('Processing layer %s' % layer_msg, file=getattr(sys, printout))
        getattr(sys, printout).flush()

    out, _ = _layer_preprocess(X, y, parallel, layer.preprocessing_,
                               False, n_jobs, verbose)

    out = base_predict(out, layer.estimators_, parallel, n=X.shape[0],
                       folded_preds=False, function_args=(True,),
                       columns=columns, as_df=as_df, n_jobs=n_jobs,
                       verbose=verbose)
    return out[0]
