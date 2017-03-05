"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 11/01/2017
licence: MIT
Base ensemble class for layer API and generic `fit` and `predict` calls on
instances.
"""

from __future__ import division, print_function
from collections import OrderedDict as Odict
import sys

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from ..base import _split_base, _check_estimators
from ..utils import check_is_fitted, check_inputs
from ._layer import fit_layer, predict_layer


# TODO: make the preprocessing of folds optional as it can take a lot of memory
class BaseEnsemble(BaseEstimator, RegressorMixin, TransformerMixin):

    """BaseEnsemble class

    Base class for adding and processing layers

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

    Methods
    --------
    fit : X, y=None
        Fits ensemble on provided data
    predict : X
        Use fitted ensemble to predict on X
    get_params : None
        Method for generating mapping of parameters. Sklearn API
    """

    def _init_layers(self, layers):
        """Set up empty layer dictionary and layer counter"""
        if layers is None:
            self._n_layers = 0
            self.layers = Odict()
        else:
            self.layers = layers

    def add(self, layer):
        """Add a layer of base estimator pipelines to the ensemble

        Parameters
        -----------
        layer : dict, list
            base estimator pipelines for given layer. If no preprocessing, pass a list of
            estimators, possible as named tuples [('est-1', est), (...)]. If
            preprocessing is desired, pass a dictionary with pipeline keys:
            {'pipe-1': [preprocessing], [estimators]}, where
            [preprocessing] should be a list of transformers, possible as named
            tuples, and estimators should be a list of estimators to fit on
            preprocessed data, possibly as named tuples. General format should
            be {'pipe-1', [('step-1', trans), (...)], [('est-1', est), (...)]},
            where named each step is optional. Each transformation step and
            estimators must accept fit and transform/predict methods

        Returns
        -----------
        self : obj,
            ensemble instance with layer initiated
        """
        self._n_layers += 1
        self.layers['layer-' + str(self._n_layers)] = _split_base(layer)
        return self

    def fit_layers(self, data, labels):
        """Fit layers of ensemble

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
        self.random_state = check_inputs(data, labels, self.random_state)
        self.layers_ = Odict()

        if self.scorer is not None:
            self.scores_ = {}

        for layer_name, layer in self.layers.items():
            data = self._partial_fit(data, labels, layer, layer_name)

        return data

    def predict_layers(self, data, labels=None):
        """Predict with fitted ensemble

        Parameters
        ----------
        data : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction
        labels : array-like, default=None
            pass through for pipeline compatibility

        Returns
        --------
        data : array-like, shape=[n_samples, n_estimators_final_layer]
            predictions from final layer
        """
        check_is_fitted(self, 'layers_')
        check_inputs(data, labels, None)

        for layer_name, layer in self.layers_.items():
            data = self._partial_predict(data, labels, layer, layer_name)

        return data

    def _partial_fit(self, data, labels, layer, layer_name):
        """Method for fitting a given layer"""
        data, scores, ests_names_, ests_, prep_ = \
            fit_layer(layer, data, labels, self.folds, self.shuffle, self.random_state, self.scorer, self.as_df,
                      True, self.n_jobs, self.printout, self.verbose, layer_msg=layer_name)

        self.layers_[layer_name] = (prep_, ests_, ests_names_)

        if self.scorer is not None:
            self.scores_.update(scores)

        return data

    def _partial_predict(self, data, labels, layer, layer_name):
        """Method for generating new prediction with a given fitted layer"""
        data, est_names_ = predict_layer(data, labels, layer, as_df=self.as_df, n_jobs=self.n_jobs,
                                         verbose=self.verbose, layer_msg=layer_name, printout=self.printout)

        _check_estimators(est_names_, layer[2])

        return data
