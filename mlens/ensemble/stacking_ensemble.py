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
from ._setup import name_estimators, name_base, _check_names
from ._clone import _clone_base_estimators, _clone_preprocess_cases
from ..utils import print_time, name_columns
from ..metrics import score_matrix
from ..parallel import preprocess_folds, preprocess_pipes
from ..parallel import fit_estimators, base_predict
from sklearn.externals import six
from time import time
import sys

# TODO: make the preprocessing of folds optional as it can take a lot of memory
# TODO: Refactor the fitting method so we can build blend and stacking from
#       same class shell


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

        # if preprocessing, seperate base estimators and preprocessing pipes
        if isinstance(base_pipelines, dict):
            self.preprocess = [(case, _check_names(p[0])) for case, p in
                               base_pipelines.items()]
            self.base_estimators = [(case, _check_names(p[1])) for case, p in
                                    base_pipelines.items()]
        # else, ensure base_estimators are named
        else:
            self.preprocess = []
            self.base_estimators = [('', _check_names(base_pipelines))]

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
        self._fit_meta_estimator(X, y, printout)

        # ========== Fit preprocessing pipes and base estimators ==========
        self._fit_base(X, y, printout)

        if self.verbose > 0:
            print_time(ts, '\nFit complete', file=printout)

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
        data = self._preprocess(X, y, False)
        M = base_predict(data, self.base_estimators_, X.shape[0],
                         folded_preds=False, columns=self.base_columns_,
                         as_df=self.as_df, n_jobs=self.n_jobs, verbose=False)

        return self.meta_estimator_.predict(M)

    def _fit_meta_estimator(self, X, y, printout):
        """Create K-fold predicts as meta estimator training data"""
        if self.verbose >= 1:
            print('> fitting meta estimator', file=printout)
            printout.flush()

        # Fit temporary base pipelines and make k-fold out of sample preds
        # Parellelized preprocessing for all folds
        if self.verbose >= 2:
            print('>> preprocessing folds', file=printout)
            printout.flush()

        data = preprocess_folds(_clone_preprocess_cases(self.preprocess),
                                X, y, folds=self.folds, fit=True,
                                shuffle=self.shuffle,
                                random_state=self.random_state,
                                n_jobs=self.n_jobs, verbose=self.verbose)

        # Parellelized k-fold predictions for meta estiamtor training set
        if self.verbose >= 2:
            print('>> fitting base estimators', file=printout)
            printout.flush()

        M = base_predict(data, _clone_base_estimators(self.base_estimators),
                         n=X.shape[0], folded_preds=True,
                         columns=self.base_columns_, as_df=self.as_df,
                         n_jobs=self.n_jobs, verbose=self.verbose)
        data = None  # discard

        if self.scorer is not None:
            cols = [] if self.as_df else name_columns(self.base_estimators_)
            self.scores_ = score_matrix(M, y, self.scorer, cols)

        if self.verbose >= 2:
            print('>> fitting meta estimator', file=printout)
            printout.flush()

        self.meta_estimator_.fit(M, y)

    def _fit_base(self, X, y, printout):
        """Fits preprocessing pipelines and base estimator on full dataset"""
        if self.verbose >= 1:
            print('\n> fitting base estimators', file=printout)

        # Parallelized fitting of preprocessing pipelines
        if self.verbose >= 2:
            print('>> preprocessing data', file=printout)
            printout.flush()

        data = self._preprocess(X, y, True)

        # Parallelized fitting of base estimators (on full training data)
        if self.verbose >= 2:
            print('>> fitting base estimators', file=printout)
            printout.flush()

        self.base_estimators_ = fit_estimators(data, y, self.base_estimators_,
                                               self.n_jobs, self.verbose)

    def _preprocess(self, X, y, method_is_fit):
        """Method for generating predictions for inputs"""
        if len(self.preprocess_) == 0:
            return [[X, '']]
        else:

            out = preprocess_pipes(self.preprocess_, X, y, fit=method_is_fit,
                                   return_estimators=method_is_fit,
                                   n_jobs=self.n_jobs, verbose=self.verbose)
            if method_is_fit:
                pipes, Z, cases = zip(*out)
                self.preprocess_ = [(case, pipe) for case, pipe in
                                    zip(cases, pipes)]
                return [[z, case] for z, case in zip(Z, cases)]
            else:
                return [[z, case] for z, case in out]

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
