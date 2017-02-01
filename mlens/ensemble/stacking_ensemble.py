#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/01/2017
licence: MIT
Stacked ensemble class for full control over the entire model's parameters.
Scikit-learn API allows full integration, including grid search and pipelining.
"""

from __future__ import division, print_function

from sklearn.base import clone, BaseEstimator, TransformerMixin, RegressorMixin
from ._setup import name_estimators, name_base, _split_base
from ._clone import _clone_base_estimators, _clone_preprocess_cases
from ._support import _check_estimators, name_columns
from ..utils import print_time
from ..metrics import score_matrix
from ..parallel import (preprocess_folds, preprocess_pipes,
                        fit_estimators, base_predict)
from sklearn.externals import six
from time import time
import sys

# TODO: make the preprocessing of folds optional as it can take a lot of memory


def _gen_in_layer(layer, X, y, folds, shuffle, random_state, scorer, as_df,
                  folded_preds, n_jobs, printout, verbose, layer_msg='layer'):
    """Generate training data layer

    Function for generating training data for next layer from an ingoing layer
    """
    if verbose >= 1:
        print('> fitting %s' % layer_msg, file=printout)
        printout.flush()

    preprocess, estimators, columns = layer

    # Fit temporary base pipelines and make k-fold out of sample preds
    # Parellelized preprocessing for all folds
    if verbose >= 2:
        print('>> preprocessing folds', file=printout)
        printout.flush()

    Min = preprocess_folds(_clone_preprocess_cases(preprocess),
                           X, y, folds=folds, fit=True, shuffle=shuffle,
                           random_state=random_state, n_jobs=n_jobs,
                           verbose=verbose)

    # Parellelized k-fold predictions for meta estimator training set
    if verbose >= 2:
        print('>> fitting ingoing layer', file=printout)
        printout.flush()

    M, fitted_estimator_names = \
        base_predict(Min, _clone_base_estimators(estimators), n=X.shape[0],
                     folded_preds=folded_preds, columns=columns, as_df=as_df,
                     n_jobs=n_jobs, verbose=verbose)
    del Min

    if scorer is not None:
        cols = [] if as_df else fitted_estimator_names
        scores = score_matrix(M, y, scorer, cols)
    else:
        scores = None

    if verbose >= 2:
        print('>> fit complete.', file=printout)
        printout.flush()

    return M, scores, fitted_estimator_names


def fit_layer(layer, X, y, folds, shuffle, random_state, scorer, as_df,
              folded_preds, n_jobs, printout, verbose, layer_msg='layer'):
    """Fit ensemble layer

    Function for fitting a layer and generating training data for next layer
    """
    out = _gen_in_layer(layer, X, y, folds, shuffle, random_state, scorer,
                        as_df, folded_preds, n_jobs, printout, verbose,
                        layer_msg)

    fitted_estimators, fitted_preprocessing = \
        _fit_layer_estimators(layer, X, y, n_jobs, printout, verbose)

    return out + (fitted_estimators, fitted_preprocessing)


def _fit_layer_estimators(layer, X, y, n_jobs, printout, verbose):
    """Fits preprocessing pipelines and layer estimator on full dataset"""
    if verbose >= 1:
        print('> fitting layer estimators', file=printout)

    # Parallelized fitting of preprocessing pipelines
    if verbose >= 2:
        print('>> preprocessing layer training data', file=printout)
        printout.flush()

    preprocess, estimators, _ = layer

    Min, preprocessing = \
        _layer_preprocess(X, y, _clone_preprocess_cases(preprocess), True,
                          n_jobs, verbose)

    # Parallelized fitting of base estimators (on full training data)
    if verbose >= 2:
        print('>> fitting base estimators', file=printout)
        printout.flush()

    return (fit_estimators(Min, y,  _clone_base_estimators(estimators), n_jobs,
                           verbose), preprocessing)


def _layer_preprocess(X, y, layer_preprocess, method_is_fit, n_jobs, verbose):
    """Method for generating predictions for inputs"""
    if (layer_preprocess is None) or (len(layer_preprocess) == 0):
        return [[X, '']], None
    else:

        out = preprocess_pipes(layer_preprocess, X, y, fit=method_is_fit,
                               return_estimators=method_is_fit,
                               n_jobs=n_jobs, verbose=verbose)
        if method_is_fit:
            pipes, Z, cases = zip(*out)
            fitted_prep = [(case, pipe) for case, pipe in
                           zip(cases, pipes)]
            return [[z, case] for z, case in zip(Z, cases)], fitted_prep
        else:
            return [[z, case] for z, case in out], None


class StackingEnsemble(BaseEstimator, RegressorMixin, TransformerMixin):

    """Stacking Ensemble

    Meta estimator class that blends a set of base estimators via a meta
    estimator. In difference to standard stacking, where the base estimators
    predict the same data they were fitted on, this class uses k-fold splits of
    the the training data make base estimators predict out-of-sample training
    data. Since base estimators predict training data as in-sample, and test
    data as out-of-sample, standard stacking suffers from a bias in that the
    meta estimators fits based on base estimator training error, but predicts
    based on base estimator test error. This blends overcomes this by splitting
    up the training set in the fitting stage, to create near identical for both
    training and test set. Thus, as the number of folds is increased, the
    training set grows closer in remeblance of the test set, but at the cost of
    increased fitting time.

    Parameters
    -----------
    meta_estimator : obj
        estimator to fit on base_estimator predictions. Must accept fit and
        predict method.
    base_pipelines : dict, list
        base estimator pipelines. If no preprocessing, pass a list of
        estimators, possible as named tuples [('est-1', est), (...)]. If
        preprocessing is desired, pass a dictionary with pipeline keys:
        {'pipe-1': [preprocessing], [estimators]}, where
        [preprocessing] should be a list of transformers, possible as named
        tuples, and estimators should be a list of estimators to fit on
        preprocesssed data, possibly as named tuples. General format should be
        {'pipe-1', [('step-1', trans), (...)], [('est-1', est), (...)]}, where
        named each step is optional. Each transformation step and estimators
        must accept fit and transform / predict methods respectively
    folds : int, obj, default=2
        number of folds to use for constructing meta estimator training set.
        Either pass a KFold class object that accepts as ``split`` method,
        or the number of folds in standard KFold
    shuffle : bool, default=True
        whether to shuffle data for creating k-fold out of sample predictions
    as_df : bool, default=False
        whether to fit meta_estimator on a dataframe. Useful if meta estimator
        allows feature importance analysis
    scorer : func, default=None
        scoring function. If a function is provided, base estimators will be
        scored on the training set assembled for fitting the meta estimator.
        Since those predictions are out-of-sample, the scores represent valid
        test scores. The scorer should be a function that accepts an array of
        true values and an array of predictions: score = f(y_true, y_pred). The
        scoring function of an sklearn scorer can be retrieved by ._score_func
    random_state : int, default=None
        seed for creating folds during fitting (if shuffle=True)
    verbose : bool, int, default=False
        level of verbosity of fitting:
            verbose = 0 prints minimum output
            verbose = 1 give prints for meta and base estimators
            verbose = 2 prints also for each stage (preprocessing, estimator)
    n_jobs : int, default=-1
        number of CPU cores to use for fitting and prediction

    Attributes
    -----------
    scores_ : dict
        scored base of base estimators on the training set, estimators are
        named according as pipeline-estimator.
    base_estimators_ : list
        fitted base estimators
    base_columns_ : list
        ordered list of base estimators as they appear in the input matrix to
        the meta estimators. Useful for mapping sklearn feature importances,
        which comes as ordered ndarrays.
    preprocess_ : dict
        fitted preprocessing pipelines

    Methods
    --------
    fit : X, y=None
        Fits ensemble on provided data
    predict : X
        Use fitted ensemble to predict on X
    get_params : None
        Method for generating mapping of parameters. Sklearn API
    """

    def __init__(self, meta_estimator, base_pipelines, folds=2, shuffle=True,
                 as_df=False, scorer=None, random_state=None,
                 verbose=False, n_jobs=-1):

        self.base_pipelines = base_pipelines
        self.meta_estimator = meta_estimator
        self.named_meta_estimator = name_estimators([meta_estimator], 'meta-')
        self.named_base_pipelines = name_base(base_pipelines)

        self.preprocess, self.base_estimators = _split_base(base_pipelines)

        self.folds = folds
        self.shuffle = shuffle
        self.as_df = as_df
        self.scorer = scorer
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit ensemble

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction
        y : array-like, shape=[n_samples, ]
            output vector to trained estimators on

        Returns
        --------
        self : obj
            class instance with fitted estimators
        """
        self.meta_estimator_ = clone(self.meta_estimator)
        self.base_estimators_ = _clone_base_estimators(self.base_estimators)
        self.preprocess_ = _clone_preprocess_cases(self.preprocess)
        self.base_columns_ = name_columns(self.base_estimators_)

        if self.verbose > 0:
            printout = sys.stdout if self.verbose > 50 else sys.stderr
            print('Fitting ensemble\n', file=printout)
            printout.flush()
            ts = time()
        else:
            printout = None

        # ========== Fit meta estimator ==========
        layer = (self.preprocess, self.base_estimators, self.base_columns_)

        (M, scores, self._fitted_estimators_,
         self.base_estimators_, self.preprocess_) = \
            fit_layer(layer, X, y, self.folds, self.shuffle, self.random_state,
                      self.scorer, self.as_df, True, self.n_jobs, printout,
                      self.verbose, layer_msg='layer')

        if self.scorer is not None:
            self.scores_ = scores

        self.meta_estimator_.fit(M, y)

        if self.verbose > 0:
            print_time(ts, 'Fit complete', file=printout)

        return self

    def predict(self, X, y=None):
        """Predict with fitted ensemble

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction

        Returns
        --------
        y : array-like, shape=[n_samples, ]
            predictions for provided input array
        """
        Min, _ = _layer_preprocess(X, y, self.preprocess_, False, self.n_jobs,
                                   self.verbose)

        M, fitted_estimator_names = \
            base_predict(Min, self.base_estimators_, X.shape[0],
                         folded_preds=False, columns=self._fitted_estimators_,
                         as_df=self.as_df, n_jobs=self.n_jobs, verbose=False)

        _check_estimators(fitted_estimator_names, self._fitted_estimators_)

        return self.meta_estimator_.predict(M)

    def get_params(self, deep=True):
        """Sklearn API for retrieveing all (also nested) model parameters"""
        if not deep:
            return super(StackingEnsemble, self).get_params(deep=False)
        else:
            out = {'folds': self.folds,
                   'shuffle': self.shuffle,
                   'as_df': self.as_df,
                   'scorer': self.scorer,
                   'random_state': self.random_state,
                   'verbose': self.verbose,
                   'n_jobs': self.n_jobs}

            out.update(self.named_base_pipelines.copy())
            for name, step in six.iteritems(self.named_base_pipelines):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_meta_estimator.copy())
            for name, step in six.iteritems(self.named_meta_estimator):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
