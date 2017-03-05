"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/01/2017
licence: MIT
Stacked ensemble class for full control over the entire model's parameters.
Scikit-learn API allows full integration, including grid search and pipelining.
"""

from __future__ import division, print_function

from .base import BaseEnsemble
from ..utils import print_time
from ..externals.six import iteritems

from sklearn.base import clone
from sklearn.pipeline import _name_estimators
from time import time
import sys


class StackingEnsemble(BaseEnsemble):

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
    layers_ : list
        fitted ensemble layers

    Methods
    --------
    add : tuple
        Method for adding a layer. Layers are added sequentially
    add_meta : obj
        Method for adding the final meta estimator. Can be added at any time.
    fit : data, labels=None
        Fits ensemble on provided data
    predict : data
        Use fitted ensemble to predict on X
    get_params : None
        Method for generating mapping of parameters. Sklearn API
    """

    def __init__(self, folds=2, shuffle=True, as_df=False, scorer=None,
                 random_state=None, verbose=False, n_jobs=-1,
                 layers=None, meta_estimator=None):

        self.folds = folds
        self.shuffle = shuffle
        self.as_df = as_df
        self.scorer = scorer
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self._init_layers(layers)
        self.meta_estimator = meta_estimator

    def add_meta(self, meta_estimator):
        """Add final estimator used to combine last layer's predictions

        Parameters
        -----------
        meta_estimator : obj
            estimator to fit on base_estimator predictions. Must accept fit and
            predict method.

        Returns
        -----------
        self : obj,
            ensemble instance with meta estimator initiated
        """
        self.meta_estimator = meta_estimator
        return self

    def fit(self, data, labels):
        """Fit ensemble

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction
        labels : array-like, shape=[n_samples, ]
            output vector to trained estimators on

        Returns
        --------
        self : obj
            class instance with fitted estimators
        """
        ts = self._print_start()

        data = self.fit_layers(data, labels)

        self.meta_estimator_ = clone(self.meta_estimator).fit(data, labels)

        if self.verbose > 0:
            print_time(ts, 'Fit complete', file=self.printout)

        return self

    def predict(self, data, labels=None):
        """Predict with fitted ensemble

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction
        labels : array-like, None
            pass through for scikit-learn compatability

        Returns
        --------
        labels : array-like, shape=[n_samples, ]
            predictions for provided input array
        """
        data = self.predict_layers(data, labels)
        return self.meta_estimator_.predict(data)

    def _print_start(self):
        if self.verbose > 0:
            self.printout = sys.stdout if self.verbose > 50 else sys.stderr
            print('Fitting ensemble\n', file=self.printout)
            self.printout.flush()
            ts = time()
            return ts
        else:
            self.printout = None
            return

    def get_params(self, deep=True):
        """Sklearn API for retrieving all (also nested) model parameters"""
        # Ensemble parameters
        out = {  # Instantiated settings
               'folds': self.folds,
               'shuffle': self.shuffle,
               'as_df': self.as_df,
               'scorer': self.scorer,
               'random_state': self.random_state,
               'verbose': self.verbose,
               'n_jobs': self.n_jobs,
                 # Layers
               'layers': self.layers,
               'meta_estimator': self.meta_estimator}

        if deep is False:
            return out
        else:
            # Get parameters of the estimators in each layer
            for layer_nm, layer in iteritems(self.layers):
                for meta_step in layer:
                    for step_nm, step in meta_step:
                        for nm, est in step:
                            # Register the estimator
                            out['%s-%s-%s' % (layer_nm, step_nm, nm)] = est
                            for k, v in iteritems(est.get_params(deep=True)):
                                # Register the estimator parameters
                                out['%s-%s-%s__%s' % (layer_nm,
                                                      step_nm,
                                                      nm, k)] = v

            # Get meta estimator parameters
            for name, est in _name_estimators([self.meta_estimator]):
                out['meta-%s' % name] = est
                for key, value in iteritems(est.get_params(deep=True)):
                    out['meta-%s__%s' % (name, key)] = value
            return out
