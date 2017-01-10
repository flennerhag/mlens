#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 11/01/2017
Stacked ensemble class for full control over the entire model's parameters.
Scikit-learn API allows full integration, including grid search and pipelining.
"""

from sklearn.base import clone, BaseEstimator, TransformerMixin, RegressorMixin
import numpy as np
from pandas import DataFrame, Series
from ._setup import name_estimators, name_base, _check_names
from ._clone import _clone_base_estimators, _clone_preprocess_cases
from ..utils import print_time
from ..parallel import preprocess_folds, preprocess_pipes, folded_predictions
from ._fit_predict import fit_estimators
from sklearn.externals import six
from time import time


class Ensemble(BaseEstimator, RegressorMixin, TransformerMixin):
    '''
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
    folds : int, default=10
        number of folds to use for constructing meta estimator training set
    shuffle : bool, default=True
        whether to shuffle data for creating k-fold out of sample predictions
    as_df : bool, default=False
        whether to fit meta_estimator on a dataframe. Useful if meta estimator
        allows feature importance analysis
    verbose : bool, int, default=False
        level of verbosity of fitting
    n_jobs : int, default=10
        number of CPU cores to use for fitting and prediction
    '''

    def __init__(self, meta_estimator, base_pipelines, folds=10,
                 shuffle=True, as_df=False, verbose=False, n_jobs=-1):

        self.base_pipelines = base_pipelines
        self.meta_estimator = meta_estimator

        self.named_meta_estimator = name_estimators([meta_estimator], 'meta-')
        self.named_base_pipelines = name_base(base_pipelines)

        # if preprocessing, seperate pipelines
        if isinstance(base_pipelines, dict):
            self.preprocess = [(case, _check_names(p[0])) for case, p in
                               base_pipelines.items()]
            self.base_estimators = [(case, _check_names(p[1])) for case, p in
                                    base_pipelines.items()]
        else:
            self.preprocess = []
            self.base_estimators = [(case, p) for case, p in base_pipelines]

        self.folds = folds
        self.shuffle = self.shuffle
        self.as_df = as_df
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        '''
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
        '''
        self.meta_estimator_ = clone(self.meta_estimator)
        self.base_estimators_ = _clone_base_estimators(self.base_estimators)
        self.preprocess_ = _clone_preprocess_cases(self.preprocess)

        if self.verbose > 0:
            print('Fitting ensemble')
            ts = time()

        # ========== Fit meta estimator ==========
        # Fit temporary base pipelines and make k-fold out of sample preds

        # Parellelized preprocessing for all folds
        data = preprocess_folds(_clone_preprocess_cases(self.preprocess),
                                X, y, self.folds, self.shuffle, True,
                                self.n_jobs, self.verbose)

        # Parellelized k-fold predictions for meta estiamtor training set
        M = folded_predictions(data,
                               _clone_base_estimators(self.base_estimators),
                               X.shape[0], self.as_df, self.n_jobs,
                               self.verbose)

        self.meta_estimator_.fit(M, y)

        # ========== Fit preprocessing and base estimator ==========

        # Parallelized fitting of preprocessing pipelines
        out = preprocess_pipes(self.preprocess_, X, y, return_estimators=True,
                               n_jobs=self.n_jobs, verbose=self.verbose)
        pipes, Z, cases = zip(*out)

        self.preprocess_ = [(case, pipe) for case, pipe in zip(cases, pipes)]

        # Parallelized fitting of base estimators (on full training data)
        data = [[z, case] for z, case in zip(Z, cases)]
        self.base_estimators_ = fit_estimators(data, y, self.base_estimators_,
                                               self.n_jobs, self.verbose)

        if self.verbose > 0:
            print_time(ts, 'Fit complete')

        return self

    def predict(self, X, y=None):
        '''
        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction
        Returns
        --------
        y : array-like, shape=[n_samples, ]
            predictions for provided input array
        '''
        if hasattr(self, 'base_estimators_'):
            M = self.base_estimators_.predict(X)
        else:
            M = None

        if hasattr(self, 'base_feature_pipelines_'):
            M = self._predict_pipeline(M, X, fitted=True)

        return self.meta_estimator_.predict(M)

    def get_params(self, deep=True):
        ''' Sklearn API for retrieveing all (also nested) model parameters'''
        if not deep:
            return super(Ensemble, self).get_params(deep=False)
        else:
            out = self.__dict__

            out.update(self.named_base_estimators.copy())
            for name, step in six.iteritems(self.named_base_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_base_feature_pipelines.copy())
            for name, step in six.iteritems(self.named_base_feature_pipelines):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_meta_estimator.copy())
            for name, step in six.iteritems(self.named_meta_estimator):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out


class PredictionFeatures(BaseEstimator, TransformerMixin):
    '''
    Class that creates features out of predictions from a set of estimators.
    While predictions for test set always are out of sample, predictions for
    training sets are often made in sample. Thus a weakness in standard
    stacking is that models are implicitly trained to adjust training errors,
    but predicts based on test errors. If these differs materially, the
    meta estimator can be severaly biased. This class implements a K-folds
    estimation option, such that K sub-estimators are fitted for each estimator
    and the training set is constructed out of these K estimators. Test set is
    constructed using estimators fitted on full training data to minimize
    noise. Class is implemented using Sklearn API, and can be used together
    with FeatureUnion, GridSearch and Pipeline utility classes. Use the
    join_X option to join prediction features to input training data.

    Parameters
    ----------

    estimators : list
        list of models to train on the dataset.
    kfold_train_preds : bool
        whether to create training set features as out of sample predictions
        using K sub-estimators (for each estimator) to predict left out folds
    folds : int, default=10
        number of folds to for constructing kfold_train_fit.
    join_X : bool, default=True
        whether prediction feature(s) should be joined to input features X
        in transform method. If set to False, transform and predict coincide -
        which can be useful for sklearn compatability
    verbose : boolean, int, default=False
        level of verbosity during fitting. Use integers to regulate level of
        printed messages during parallel fitting and grid search.
    n_jobs : int, default=-1
        number of CPU cores to use. Set to -1 for automatically deterning max
        number of CPUs. See sklearn documentation for details.
    '''

    def __init__(self, estimators=[], folds=10, join_X=True, as_df=None,
                 kfold_train_preds=True, verbose=False, n_jobs=-1):

        self.estimators = estimators
        self.named_estimators = name_estimators(estimators)

        self.folds = folds
        self.join_X = join_X
        self.as_df = as_df
        self.kfold_train_preds = kfold_train_preds
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.xtrain_id = id(X)  # store X to ID training set in transform

        # Fit on full training set for test set predictions
        if self.verbose > 0:
            msg = 'Initiating test set estimator fits for %i estimators\n'
            print(msg % (len(self.estimators)))
            ts = time()

        self.fitted_estimators_ = fit_estimators

        if self.verbose > 0:
            print_time(ts, 'Estimators fitted')

        return self

    def predict(self, X, y=None, as_df=False):
        '''
        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction
        as_df : bool, default=False
            whether prediction array should be returned as as Pandas DataFrame
        Returns
        --------
        P : array-like, shape=[n_samples, n_estimators]
            prediction array with a vector of preds for each fitted estimator
        '''

        try:
            idx = X.index
        except AttributeError:
            idx = range(X.shape[0])

        if self._use_kfold_preds(X):
            # Use sub-estimators for prediction
            subout = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                           delayed(predict_estimator)(X, *tup)
                           for tup in self.fitted_subestimators_)
            P = self._gen_df(subout, idx)

        else:
            # Use main estimators for prediction
            out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                           delayed(predict_estimator)(X, *tup)
                           for tup in self.fitted_estimators_)

            (ests, preds) = zip(*out)

            P = DataFrame(np.array(preds).T,
                          columns=['Pred_' + est for est in ests], index=idx)
        # Check return type
        if self.as_df or ((self.as_df is None) and isinstance(X, DataFrame)):
            return P
        else:
            return P.values

    def transform(self, X, y=None):
        '''
        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction

        Returns
        --------
        Xout : array-like, shape=[n_samples, (n_features +) n_estimators]
            input matrix with prediction feature(s) for each fitted estimator
            if join_X if False, returns only prediction features
        '''

        P = self.predict(X)

        if self.join_X:
            if isinstance(X, DataFrame):
                return X.join(P)
            else:
                return np.hstack((X, P))
        else:
            return P

    def _use_kfold_preds(self, X):
        ''' Check whether kfold predictions is required'''
        if self.kfold_train_preds:
            # Check if X is identical to xtrain
            # Note that if xtrain is a subset of X,
            # this implementation will not make a
            # distnction
            if isinstance(X, (DataFrame, Series)):
                return X.equals(self.xtrain)
            else:
                return (X == self.xtrain).all()
        else:
            return False

    def _gen_df(self, out, idx):
        ''' Utility function for reshaping prediction run into matrix'''
        n, m = len(idx), len(self.estimators)
        temp = DataFrame(np.zeros((n, m)), index=idx,
                         columns=['Pred_' + est_name for est_name in
                                  self.named_estimators])

        for tup in out:
            (est_name, pred, idx) = tup
            pos = temp.columns.get_loc('Pred_' + est_name)
            temp.iloc[idx, pos] = pred

        return temp

    def get_params(self, deep=True):
        ''' Sklearn API compatibility'''
        if not deep:
            return super(PredictionFeatures, self).get_params(deep=False)
        else:
            out = self.__dict__
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
        return out
