"""ML-ENSEMBLE

author: Sebastian Flennerhag
copyright: 2017
licence: MIT
Stacked ensemble class for full control over the entire model's parameters.
Scikit-learn API allows full integration, including grid search and pipelining.
"""

from __future__ import division, print_function

from .base import BaseEnsemble
from ._layer import fit_layer, predict_layer
from ..utils import print_time
from ..metrics import set_scores

from sklearn.base import clone
from time import time
import sys


class StackingEnsemble(BaseEnsemble):

    """Stacking Ensemble.

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
    training set grows closer in resemblance of the test set, but at the cost
    of increased fitting time.

    Parameters
    -----------
    folds : int or object (default = 2)
        number of folds to use for constructing meta estimator training set.
        Either pass a KFold class object that accepts as ``split`` method,
        or the number of folds in standard KFold.

    shuffle : bool (default = True)
        whether to shuffle data for creating k-fold out of sample predictions.

    as_df : bool (default = False)
        whether to fit meta_estimator on a pandas DataFrame. Useful if meta
        estimator allows feature importance analysis.

    scorer : object (default = None)
        scoring function. If a function is provided, base estimators will be
        scored on the training set assembled for fitting the meta estimator.
        Since those predictions are out-of-sample, the scores represent valid
        test scores. The scorer should be a function that accepts an array of
        true values and an array of predictions: score = f(y_true, y_pred).

    random_state : int (default = None)
        seed for creating folds during fitting (if shuffle = True).

    verbose : bool or int (default = False)
        level of verbosity of fitting:
            - verbose = 0 prints minimum output
            - verbose = 1 give prints for each layer
            - verbose = 2 prints also for each stage (preprocess, estimation).

    n_jobs : int (default = -1)
        number of CPU cores to use for fitting and prediction.

    Attributes
    -----------
    scores_ : dict
        scored base of base estimators on the training set, estimators are
        named according as pipeline-estimator.

    layers : object
        container class for layers.

    Methods
    -----------
    add : estimators (list or dict), preprocessing (list of dict)
        Method for adding a layer. If no preprocessing is desired, or if the
        same preprocessing applies to all estimators in a layer, both
        `estimators` and `preprocessing` can be lists of instances, or
        lists of named tuples of instances (i..e [('name', est), ...].
        
        If a preprocessing mapping is desired, both `estimators` and
        `preprocessing` must be dictionaries with overlapping keys, where the
        value for each key is a list of instances (possibly as named tuples)
        belonging to that preprocessing case.
        
        Layers are connected sequentially in the order they are added.

    add_meta : instance
        The meta estimator to fit on the predictions of the final layer.

    fit : X, y (default = None)
        Fits ensemble on provided input data.

    predict : X
        Use fitted ensemble to predict on X.

    get_params :
        Method for generating mapping of parameters through the Sklearn API.
    """

    def __init__(self,
                 folds=2,
                 shuffle=True,
                 as_df=False,
                 scorer=None,
                 random_state=None,
                 verbose=False,
                 n_jobs=-1,
                 layers=None,
                 meta_estimator=None):

        self.folds = folds
        self.shuffle = shuffle
        self.as_df = as_df
        self.scorer = scorer
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

        self.printout = 'stdout' if self.verbose >= 50 else 'stderr'

        self._init_layers(layers)
        self.meta_estimator = meta_estimator
        
    def add(self, estimators, preprocessing=None):
        """Add layer to ensemble.
        
        Parameters
        ----------
        preprocessing: dict of lists or list, optional (default = [])
            preprocessing pipelines for given layer. If
            the same preprocessing applies to all estimators, `preprocessing`
            can be a list of transformer instances. The list can contain the
            instances directly, or named tuples of transformers:

            ```
            option_1 = [transformer_1, transformer_2]
            option_2 = [('trans-1', transformer_1), ('trans-2', transformer_2)]
            ```

             If different preprocessing pipelines are desired, a dictionary
             that maps preprocessing pipelines must be passed. The names of the
             preprocessing dictionary must correspond to the names of the
             estimator dictionary.

             ```
             preprocessing_cases = {'case-1': [trans_1, trans_2].
                                    'case-2': [alt_trans_1, alt_trans_2]}

             estimators = {'case-1': [est_a, est_b].
                           'case-2': [est_c, est_d]}
             ```

             The lists for each dictionary entry can be both a list of
             transformers and a list of named tuples of transformers,
             as in `option_1` and `option_2` respectively.

        estimators: dict of lists or list
            estimators constituting the layer. If no preprocessing,
            or preprocessing applies to all estimators, a list of estimators
            can be passed. The list can either contain estimator instances,
            or named tuples of estimator instances:

            ```
            option_1 = [estimator_1, estimator_2]
            option_2 = [('est-1', estimator_1), ('est-2', estimator_2)]
            ```

             If different preprocessing pipelines are desired, a dictionary
             that maps estimators to preprocessing pipelines must be passed.
             The names of the estimator dictionary must correspond to the
             names of the estimator dictionary:

             ```
             preprocessing_cases = {'case-1': [trans_1, trans_2].
                                    'case-2': [alt_trans_1, alt_trans_2]}

             estimators = {'case-1': [est_a, est_b].
                           'case-2': [est_c, est_d]}
             ```

             The lists for each dictionary entry can be both a list of
             estimators and a list of named tuples of estimators,
             as in `option_1` and `option_2` respectively.
                  
        Returns
        ----------
        self : instance
            ensemble instance with layer instantiated.
        """
        fit_params = {'folds': self.folds,
                      'shuffle': self.shuffle,
                      'random_state': self.random_state,
                      'scorer': self.scorer,
                      'as_df': self.as_df,
                      'folded_preds': True,
                      'n_jobs': self.n_jobs,
                      'printout': self.printout,
                      'verbose': self.verbose}
        
        predict_params = {'as_df': self.as_df,
                          'n_jobs': self.n_jobs,
                          'verbose': self.verbose,
                          'printout': self.printout}
        
        return self._add(estimators=estimators,
                         preprocessing=preprocessing,
                         fit_function=fit_layer,
                         fit_params=fit_params,
                         predict_function=predict_layer,
                         predict_params=predict_params)

    def add_meta(self, meta_estimator):
        """Add final estimator used to combine last layer's predictions.

        Parameters
        -----------
        meta_estimator : instances
            estimator to fit on base_estimator predictions. Must accept fit and
            predict method.

        Returns
        -----------
        self : instance
            ensemble instance with meta estimator initiated.
        """
        self.meta_estimator = meta_estimator
        return self

    def fit(self, X, y=None):
        """Fit ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like of shape = [n_samples, ] or None (default = None)
            output vector to trained estimators on.

        Returns
        ----------
        self : instance
            class instance with fitted estimators.
        """
        ts = self._print_start()

        out, X = \
            self._fit_layers(X, y, return_final=True, verbose=self.verbose)

        self.scores_ = set_scores(self, out)
        
        self.meta_estimator_ = clone(self.meta_estimator).fit(X, y)

        if self.verbose > 0:
            print_time(ts, 'Ensemble fitted', file=getattr(sys, self.printout))

        return self

    def predict(self, X, y=None):
        """Predict with fitted ensemble.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            input matrix to be used for prediction.

        y : array-like, None (default = None)
            pass through for scikit-learn compatibility.

        Returns
        --------
        y_pred : array-like, shape=[n_samples, ]
            predictions for provided input array.
        """
        X = self._predict_layers(X, y, verbose=self.verbose)
        return self.meta_estimator_.predict(X)
        
    def _print_start(self):
        if self.verbose > 0:
            print('Fitting ensemble\n', file=getattr(sys, self.printout))
            getattr(sys, self.printout).flush()
            ts = time()
            return ts
        return
