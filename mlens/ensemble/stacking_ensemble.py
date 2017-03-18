"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Stacked ensemble class for full control over the entire model's parameters.
Scikit-learn API allows full integration, including grid search and pipelining.
"""

from __future__ import division

from .base import BaseEnsemble
from ._layer import predict_layer
from ..parallel.stacking import folded_fit, predict_on_full
from ..utils import print_time, safe_print, check_inputs, check_ensemble_build
from ..metrics import set_scores

from sklearn.base import clone
from sklearn.model_selection import KFold
from time import time


class StackingEnsemble(BaseEnsemble):

    r"""Stacking Ensemble class.

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
    ----------
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

    raise_on_exception : bool (default = False)
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer.

    array_check : int (default = 2)
        level of strictness in checking input arrays.

            - ``array_check = 0`` will not check ``X`` or ``y``
            - ``array_check = 1`` will check ``X`` and ``y`` for \
            inconsistencies and warn when format looks suspicious, \
            but retain original format.
            - ``array_check = 2`` will impose Scikit-learn array checks, \
            which converts ``X`` and ``y`` to numpy arrays and raises \
            an error if conversion fails.

    verbose : int or bool (default = False)
        level of verbosity.

            - ``verbose = 0`` silent (same as ``verbose = False``)
            - ``verbose = 1`` messages at start and finish \
            (same as ``verbose = True``)
            - ``verbose = 2`` messages for each layer

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    n_jobs : int (default = -1)
        number of CPU cores to use for fitting and prediction.

    Attributes
    ----------
    scores\_ : dict
        if ``scorer`` was passed to instance, ``scores_`` contains dictionary
        with cross-validated scores assembled during ``fit`` call. The fold
        structure used for scoring is determined by ``folds``.

    layers : instance
        container instance for layers see
        :py:class:`mlens.ensemble.base.LayerContainer` for further
        information.

    Examples
    --------

    Instantiate ensembles with no/same preprocessing with estimator lists.

    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = StackingEnsemble()
    >>> ensemble.add([SVR(), Lasso()]).add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)

    Instantiate ensembles with different preprocessing pipelines through dicts.

    >>> from sklearn.datasets import load_boston
    >>> from sklearn. preprocessing import MinMaxScaler, StandardScaler
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> preprocessing_cases = {'mm': [MinMaxScaler()],
    ...                        'sc': [StandardScaler()]}
    >>>
    >>> estimators_per_case = {'mm': [SVR()],
    ...                        'sc': [Lasso()]}
    >>>
    >>> ensemble = StackingEnsemble()
    >>> ensemble.add(estimators_per_case, preprocessing_cases)
    >>> ensemble.add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    """

    def __init__(self,
                 folds=2,
                 shuffle=True,
                 as_df=False,
                 scorer=None,
                 random_state=None,
                 raise_on_exception=False,
                 array_check=2,
                 verbose=False,
                 n_jobs=-1,
                 layers=None,
                 meta_estimator=None):

        self.folds = folds
        self.shuffle = shuffle
        self.as_df = as_df
        self.scorer = scorer
        self.random_state = random_state
        self.raise_on_exception = raise_on_exception
        self.array_check = array_check
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.layers = layers
        self.meta_estimator = meta_estimator
        self.printout = 'stdout' if self.verbose >= 50 else 'stderr'

    def add(self, estimators, preprocessing=None):
        """Add layer to ensemble.

        Parameters
        ----------
        preprocessing: dict of lists or list, optional (default = None)
            preprocessing pipelines for given layer. If
            the same preprocessing applies to all estimators, ``preprocessing``
            should be a list of transformer instances. The list can contain the
            instances directly, named tuples of transformers,
            or a combination of both. ::

                option_1 = [transformer_1, transformer_2]
                option_2 = [("trans-1", transformer_1),
                            ("trans-2", transformer_2)]
                option_3 = [transformer_1, ("trans-2", transformer_2)]

            If different preprocessing pipelines are desired, a dictionary
            that maps preprocessing pipelines must be passed. The names of the
            preprocessing dictionary must correspond to the names of the
            estimator dictionary. ::

                preprocessing_cases = {"case-1": [trans_1, trans_2],
                                       "case-2": [alt_trans_1, alt_trans_2]}

                estimators = {"case-1": [est_a, est_b],
                              "case-2": [est_c, est_d]}

            The lists for each dictionary entry can be any of ``option_1``,
            ``option_2`` and ``option_3``.

        estimators: dict of lists or list
            estimators constituting the layer. If ``preprocessing`` is
            ``None`` or ``list``, ``estimators`` should be a ``list``.
            The list can either contain estimator instances,
            named tuples of estimator instances, or a combination of both. ::

                option_1 = [estimator_1, estimator_2]
                option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
                option_3 = [estimator_1, ("est-2", estimator_2)]

            If different preprocessing pipelines are desired, a dictionary
            that maps estimators to preprocessing pipelines must be passed.
            The names of the estimator dictionary must correspond to the
            names of the estimator dictionary. ::

                preprocessing_cases = {"case-1": [trans_1, trans_2],
                                       "case-2": [alt_trans_1, alt_trans_2]}

                estimators = {"case-1": [est_a, est_b],
                              "case-2": [est_c, est_d]}

            The lists for each dictionary entry can be any of ``option_1``,
            ``option_2`` and ``option_3``.

        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        if isinstance(self.folds, int):
            kf = KFold(self.folds, self.shuffle, self.random_state)
        else:
            kf = self.folds

        return self._add(estimators=estimators,
                         preprocessing=preprocessing,
                         fit_function=folded_fit,
                         fit_params=None,
                         predict_function=predict_on_full,
                         predict_params=None,
                         indexer=kf,
                         verbose=self.verbose)

    def add_meta(self, meta_estimator):
        """Add final estimator used to combine last layer's predictions.

        Parameters
        -----------
        meta_estimator : instances
            estimator to fit on base_estimator predictions.
            Must accept ``fit`` and ``predict`` methods.

        Returns
        -------
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
        -------
        self : instance
            class instance with fitted estimators.
        """
        # Ensemble build check
        if not check_ensemble_build(self):
            # No layers instantiated. Return vacuous fit.
            return self

        # Inputs check.
        X, y = check_inputs(X, y, self.array_check)

        # Layer estimation
        ts = self._print_start()

        out, X = \
            self._fit_layers(X, y, return_final=True)

        self.scores_ = set_scores(self, out)

        # Meta estimator fit
        self.meta_estimator_ = clone(self.meta_estimator).fit(X, y)

        if self.verbose > 0:
            print_time(ts, 'Ensemble fitted', file=self.printout)

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
        -------
        y_pred : array-like, shape=[n_samples, ]
            predictions for provided input array.
        """
        # Ensemble build check
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is false
            return

        # Inputs check
        X, y = check_inputs(X, y, self.array_check)

        # Process Layers
        X = self._predict_layers(X, y)

        # Final prediction
        return self.meta_estimator_.predict(X)

    def _print_start(self):
        """Utility for printing initial message and launching timer."""
        if self.verbose > 0:
            safe_print('Fitting ensemble\n', file=self.printout,
                       flush=True)
            ts = time()
            return ts
        return
