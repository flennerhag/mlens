#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 10/02/2017
licence: MIT
Class for generating new features in the form of predictions from a given set
of models. Prediction are generated using KFold out-of-sample predictions.
"""

from __future__ import division, print_function

from pandas import DataFrame, concat
from numpy import hstack

from sklearn.base import BaseEstimator, TransformerMixin
from ..base import name_estimators
from ..base import _clone_base_estimators
from ..base import _check_estimators
from ..utils import print_time
from ..parallel import preprocess_folds, fit_estimators, base_predict
from ..externals import six
from time import time
import sys


class PredictionFeature(BaseEstimator, TransformerMixin):

    """Prediction Feature

    Transformer that appends columns of predictions from a set of estimators
    to a matrix.

    Parameters
    -----------
    estimators : obj
        estimators to use for generating predictions. One feature of
        predictions is generated per estimator
    folds : int, obj, default=2
        number of folds to use for constructing prediction feature set.
        Either pass a KFold class object that accepts as ``split`` method,
        or the number of folds in standard KFold
    shuffle : bool, default=True
        whether to shuffle data for creating k-fold out of sample predictions
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
    estimators_ : list
        fitted estimator

    Methods
    --------
    fit : X, y=None
        Fits estimators on provided data
    predict : X
        Use fitted estimators to create matrix of predictions,
        shape [n_samples, n_estimators]
    transform : X
        Use fitted estimators to generate and concatenate predictions to X
    get_params : None
        Method for generating mapping of parameters. Sklearn API
    """

    def __init__(self, estimators, folds=2, shuffle=True, scorer=None,
                 random_state=None, verbose=False, n_jobs=-1):

        self.estimators = [('', estimators)]
        self.named_estimator = name_estimators(estimators)

        self.folds = folds
        self.shuffle = shuffle
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

        # Store training set id
        self._train_id_ = id(X)

        if self.verbose > 0:
            printout = sys.stdout if self.verbose > 50 else sys.stderr
            print('Fitting estimators\n', file=printout)
            printout.flush()
            ts = time()
        else:
            printout = None

        # Fit estimators for training set
        Min = preprocess_folds(None, X, y, folds=self.folds, fit=False,
                               shuffle=self.shuffle,
                               random_state=self.random_state,
                               n_jobs=self.n_jobs, verbose=self.verbose)

        # >> Generate mapping between folds and estimators
        Min = [tup[:-1] + [i] for i, tup in enumerate(Min)]
        ests_ = {i: _clone_base_estimators(self.estimators)['']
                 for i in range(len(Min))}
        self.train_ests_ = fit_estimators(Min, ests_, None,
                                          self.n_jobs, self.verbose)

        # Fit estimators for test set
        self.test_ests_ = \
            fit_estimators([[X, '']], _clone_base_estimators(self.estimators),
                           y, self.n_jobs, self.verbose)

        fitted_test_ests = [est_name for est_name, _ in self.test_ests_['']]
        self._fitted_ests = fitted_test_ests

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

        as_df = isinstance(X, DataFrame)

        if id(X) == self._train_id_:
            # Use cv folds to generate predictions
            Min = preprocess_folds(None, X, y, folds=self.folds, fit=False,
                                   shuffle=self.shuffle,
                                   random_state=self.random_state,
                                   n_jobs=self.n_jobs, verbose=self.verbose)

            Min = [tup[:-1] + [i] for i, tup in enumerate(Min)]
            folded_preds = True
            estimators = self.train_ests_
        else:
            # Predict using estimators fitted on full training data
            Min = [[X, '']]
            folded_preds = False
            estimators = self.test_ests_

        # Generate predictions matrix
        M, fitted_estimator_names = \
            base_predict(Min, estimators, n=X.shape[0],
                         folded_preds=folded_preds, fit=False,
                         columns=self._fitted_ests, combine_keys=False,
                         as_df=as_df, n_jobs=self.n_jobs,
                         verbose=self.verbose)
        _check_estimators(self._fitted_ests, fitted_estimator_names)
        return M

    def transform(self, X, y=None):

        M = self.predict(X, y)

        if isinstance(X, DataFrame):
            return concat((X, M), 1)
        else:
            return hstack((X, M))

    def get_params(self, deep=True):
        """Sklearn API for retrieveing all (also nested) model parameters"""
        if not deep:
            return super(PredictionFeature, self).get_params(deep=False)
        else:
            out = {'folds': self.folds,
                   'shuffle': self.shuffle,
                   'random_state': self.random_state,
                   'verbose': self.verbose,
                   'n_jobs': self.n_jobs}

            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_base_pipelines):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out
