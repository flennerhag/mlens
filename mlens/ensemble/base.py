#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/01/2017
licence: MIT
Base ensemble class for layer API and genereic `fit` and `predict` calls on
instances.
"""

from __future__ import division, print_function

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from ..base import _split_base, _check_estimators
from ..utils import check_is_fitted, check_inputs
from ._layer import fit_layer, predict_layer


# TODO: make the preprocessing of folds optional as it can take a lot of memory
class LayerMixin(BaseEstimator, RegressorMixin, TransformerMixin):

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

    def add(self, layer):
        """Add a layer of base estimator pipelines to the ensemble

        Parameters
        -----------
        base_pipelines : dict, list
            base estimator pipelines. If no preprocessing, pass a list of
            estimators, possible as named tuples [('est-1', est), (...)]. If
            preprocessing is desired, pass a dictionary with pipeline keys:
            {'pipe-1': [preprocessing], [estimators]}, where
            [preprocessing] should be a list of transformers, possible as named
            tuples, and estimators should be a list of estimators to fit on
            preprocesssed data, possibly as named tuples. General format should
            be {'pipe-1', [('step-1', trans), (...)], [('est-1', est), (...)]},
            where named each step is optional. Each transformation step and
            estimators must accept fit and transform/predict methods

        Returns
        -----------
        self : obj,
            ensemble instance with layer initiated
        """
        if self.layers is None:
            # Initiate counter
            self._n_layers = 1
            self.layers = {}
        else:
            self._n_layers += 1

        self.layers['layer-' + str(self._n_layers)] = _split_base(layer)

        return self

    def fit_layers(self, X, y):
        """Fit layers of ensemble

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
        X, y, self.random_state = check_inputs(X, y, self.random_state)

        self.layers_ = {}
        if self.scorer is not None:
            self.scores_ = {}

        for layer_name, layer in self.layers.items():

            X, scores, ests_names_, ests_, prep_ = \
                fit_layer(layer, X, y, self.folds, self.shuffle,
                          self.random_state, self.scorer, self.as_df,
                          True, self.n_jobs, self.printout, self.verbose,
                          layer_msg=layer_name)

            self.layers_[layer_name] = (prep_, ests_, ests_names_)

            if self.scorer is not None:
                self.scores_.update(scores)

        return X

    def predict_layers(self, X, y=None):
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
        check_is_fitted(self, 'layers_')
        X, y, _ = check_inputs(X, y, None)

        for layer_name, layer in self.layers_.items():

            X, ests_names_ = predict_layer(X, y, layer, as_df=self.as_df,
                                           n_jobs=self.n_jobs,
                                           verbose=self.verbose)

            _check_estimators(ests_names_, layer[2])

        return X
