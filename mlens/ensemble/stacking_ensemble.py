"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Stacked ensemble class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .base import BaseEnsemble
from ..base import FullIndex
from ..utils import print_time, safe_print, check_inputs, check_ensemble_build
from ..metrics import set_scores

from ..externals.validation import check_random_state

from time import time


class StackingEnsemble(BaseEnsemble):
    r"""Stacking Ensemble class.

    Blends a set of base estimators via a meta estimator using K-Fold
    training set estimation. For every layer, the prediction matrix used as
    training set by the subsequent layer or meta estimator is constructed by
    partitioning the data into K folds, with each transformer / estimator
    fitted of K - 1 folds and used to predict the left out fold. This process
    is repeated until every sample in the training set has been predicted.

    Every transformer / estimator is additionally fitted on the full data set,
    and these fully fitted estimators are used to generate predictions once the
    ensemble is fitted. Hence, the K-fold estimation technique builds the
    training set used by subsequent layer in a manner as close to the
    predictions it would see during prediction. This method ensures all
    estimators (except the initial layer) are fitted on test errors of the
    estimators in the preceding layer.

    The final layer's predictions are combined through a meta learner specified
    by the user.

    Stacking is a time consuming method, as it requires several fittings of
    every estimator in the ensemble. With large sets of data, other ensembles
    that fits the ensemble through various combinations of subsets can be
    much faster at little loss of performance. However, when data is noisy or
    of high variance, the :class:`StackingEnsemble` ensure all information is
    used during fitting.

    Parameters
    ----------
    folds : int (default = 2)
        number of folds to use during fitting. Note: this parameter can be
        specified on a layer-specific basis in the :attr:`add` method.

    shuffle : bool (default = True)
        whether to shuffle data before generating folds.

    random_state : int (default = None)
        random seed if shuffling inputs.

    scorer : object (default = None)
        scoring function. If a function is provided, base estimators will be
        scored on the training set assembled for fitting the meta estimator.
        Since those predictions are out-of-sample, the scores represent valid
        test scores. The scorer should be a function that accepts an array of
        true values and an array of predictions: ``score = f(y_true, y_pred)``.

    raise_on_exception : bool (default = True)
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

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
        container instance for layers see :class:`LayerContainer` for further
        information.

    See Also
    --------
    :class:`BlendEnsemble`

    Examples
    --------

    Instantiate ensembles with no/same preprocessing with estimator lists.

    >>> from mlens.ensemble import StackingEnsemble
    >>> from mlens.metrics.metrics import rmse_scoring
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = StackingEnsemble()
    >>> ensemble.add([SVR(), Lasso()]).add(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse_scoring(y, preds)
    6.9553583775881407

    Instantiate ensembles with different preprocessing pipelines through dicts.

    >>> from mlens.ensemble import StackingEnsemble
    >>> from mlens.metrics.metrics import rmse_scoring
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
    >>> ensemble.add(estimators_per_case, preprocessing_cases).add(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse_scoring(y, preds)
    7.8413294010791557
    """

    def __init__(self,
                 folds=2,
                 shuffle=False,
                 random_state=None,
                 scorer=None,
                 raise_on_exception=True,
                 array_check=2,
                 verbose=False,
                 n_jobs=-1,
                 layers=None,
                 meta_estimator=None):

        self.folds = folds
        self.shuffle = shuffle
        self.scorer = scorer
        self.random_state = random_state
        self.raise_on_exception = raise_on_exception
        self.array_check = array_check
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.layers = layers
        self.meta_estimator = meta_estimator
        self.printout = 'stdout' if self.verbose >= 50 else 'stderr'

    def add(self, estimators, preprocessing=None, folds=None):
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

        estimators: dict of lists or list or instance
            estimators constituting the layer. If preprocessing is none and the
            layer is meant to be the meta estimator, it is permissible to pass
            a single instantiated estimator. If ``preprocessing`` is
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

        folds : int, optional
            Use if a different number of folds is desired than what the
            ensemble was instantiated with.

        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        c = folds if folds is not None else self.folds
        return self._add(
                estimators=estimators,
                cls='stack',
                preprocessing=preprocessing,
                indexer=FullIndex(c, raise_on_exception=self.raise_on_exception),
                verbose=self.verbose)

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

        if self.shuffle:
            r = check_random_state(self.random_state)
            r.shuffle(X)
            r.shuffle(y)

        # Layer estimation
        ts = self._print_start()

        out = self._fit_layers(X, y)

        self.scores_ = set_scores(self, out)

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
            # No layers instantiated, but raise_on_exception is False
            return

        # Inputs check
        X, y = check_inputs(X, y, self.array_check)

        if self.shuffle:
            r = check_random_state(self.random_state)
            r.shuffle(X)
            r.shuffle(y)

        y = self._predict_layers(X, y)

        if y.shape[1] == 1:
            # Meta estimator is treated as a layer and thus a prediction
            # matrix with shape [n_samples, 1] is created. Ravel before return
            y = y.ravel()

        return y

    def _print_start(self):
        """Utility for printing initial message and launching timer."""
        if self.verbose > 0:
            safe_print('Fitting ensemble\n', file=self.printout, flush=True)
            ts = time()
            return ts
        return
