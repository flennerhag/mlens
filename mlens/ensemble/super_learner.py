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


class SuperLearner(BaseEnsemble):
    r"""Super Learner class.

    The Super Learner (also known as the Stacking Ensemble)is an
    supervised ensemble algorithm that uses K-fold estimation to map a
    training set (X, y) into a prediction set (Z, y), where the predictions
    in Z are constructed using K-Fold splits of X to ensure Z reflects test
    errors, and that applies a user-specified meta learner to predict
    y from Z. The algorithm in sudo code follows:

        #. Specify a library of L base learners
        #. Fit all base learners on X and store them
        #. Split X into K folds, fit every learner in L on the training set\
            and predict test set. Repeat until all folds have been predicted.
        #. Construct a matrix Z by stacking the predictions per fold
        #. Fit the meta learner on Z and store the learner

    The ensemble can be used for prediction by mapping a new test set T into a
    prediction set Z' using the L learners fitted in (2), and then mapping Z'
    to y' using the fitted meta learner from (5).

    The Super Learner does asymptotically as well as (up to a constant) the
    Oracle selector. For the theory behind the Super Learner, see
    [1]_ and [2]_ as well as references therein.

    Stacking K-fold predictions to cover an entire training set is a time
    consuming method and can be prohibitively costly for large datasets.
    With large data, other ensembles that fits an ensemble on subsets
    can achieve similar performance at a fraction of the training time.
    However, when data is noisy or of high variance,
    the :class:`SuperLearner` ensure all information is
    used during fitting.

    References
    ----------

    .. [1] van der Laan, Mark J.; Polley, Eric C.; and Hubbard, Alan E.,
    "Super Learner" (July 2007). U.C. Berkeley Division of Biostatistics
    Working Paper Series. Working Paper 222.
    http://biostats.bepress.com/ucbbiostat/paper222

    .. [2] Polley, Eric C. and van der Laan, Mark J.,
    "Super Learner In Prediction" (May 2010). U.C. Berkeley Division of
    Biostatistics Working Paper Series. Working Paper 266.
    http://biostats.bepress.com/ucbbiostat/paper266

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

    Instantiate ensembles with no preprocessing: use list of estimators

    >>> from mlens.ensemble import SuperLearner
    >>> from mlens.metrics.metrics import rmse_scoring
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = SuperLearner()
    >>> ensemble.add([SVR(), ('can name some or all est', Lasso())]).add(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse_scoring(y, preds)
    6.9553583775881407

    Instantiate ensembles with different preprocessing pipelines through dicts.

    >>> from mlens.ensemble import SuperLearner
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
    ...                        'sc': [('can name some or all ests', Lasso())]}
    >>>
    >>> ensemble = SuperLearner()
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
                 layers=None):

        self.folds = folds
        self.shuffle = shuffle
        self.scorer = scorer
        self.random_state = random_state
        self.raise_on_exception = raise_on_exception
        self.array_check = array_check
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.layers = layers
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
        if not check_ensemble_build(self):
            # No layers instantiated. Return vacuous fit.
            return self

        X, y = check_inputs(X, y, self.array_check)

        if self.shuffle:
            r = check_random_state(self.random_state)
            r.shuffle(X)
            r.shuffle(y)

        ts = self._print_start()

        self._fit_layers(X, y)
# TODO:        self.scores_ = set_scores(self)

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
        if not check_ensemble_build(self):
            # No layers instantiated, but raise_on_exception is False
            return

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
