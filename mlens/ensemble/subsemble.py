"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:licence: MIT

Subsemble class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .base import BaseEnsemble
from ..base import FullIndex, SubsetIndex


class Subsemble(BaseEnsemble):
    r"""Subsemble class.

    Subsemble is a supervised ensemble algorithm that uses subsets of
    the full data to fit a layer, and within each subset K-fold estimation
    to map a training set (X, y) into a prediction set (Z, y), where Z is a
    matrix of prediction from each estimator on each subset (thus of shape
    shape [n_samples, (n_partitions * n_estimators)]). Z is constructed
    using K-Fold splits of each partition of X to ensure Z reflects test
    errors within each partition. A final user-specified meta learner is
    fitted to the final ensemble layer's prediction, to learn the best
    combination of subset-specific estimator predictions.
    The algorithm in sudo code follows:

        1. For each layer in the ensemble, do:
            a. Specify a library of L base learners
            b. Specify a partition strategy and partition X into J partitions
            c. Fit all base learners on every partition and store them
            d. Split every partition into K folds, fit every learner in L on \
                the training set. Collect the kth fold from all partitions \
                    and predict.
            e. Construct a matrix Z by stacking the predictions
        2. Fit the meta learner on Z and store the learner

    The ensemble can be used for prediction by mapping a new test set T into a
    prediction set Z' using the L learners fitted in (1.a), and then mapping Z'
    to y' using the fitted meta learner from (2).

    The Subsemble does asymptotically as well as (up to a constant) the
    Oracle selector. For the theory behind the Subsemble, see
    [1]_ and references therein.

    By partitioning the data into subset and fitting on those, a Subsemble
    can reduce training time considerably if estimators does not scale
    linearly. Moreover, Subsemble allows estimators to learn different
    patterns from each subset, and so can improve the overall performance
    by achieving a tighter fit on each subset. Since all observations in the
    training set are predicted, no information is lost between layers.
    Subsemble is performant on small,  medium and large data and is a
    competitive alternative to the :class:`SuperLearner`.


    References
    ----------
    .. [1] Sapp, S., van der Laan, M. J., & Canny, J. (2014).
    Subsemble: an  ensemble method for combining subset-specific algorithm
    fits. Journal of Applied Statistics, 41(6), 1247-1259.
    http://doi.org/10.1080/02664763.2013.864263

    Notes
    -----
    This implementation splits X into partitions sequentially, i.e. without
    randomizing indices, unless the ``shuffle`` parameter is set to ``True``.
    In this case, any input data ``X`` will be shuffled before partitioned.
    However, the splitting is naive, in that no learning is involved in
    splitting data. Supervised partitioning is under development.

    See Also
    --------
    :class:`BlendEnsemble`, :class:`SuperLearner`

    Parameters
    ----------
    partitions : int (default = 2)
        number of partitions to split data into. For each layer,
        increasing partitions increases the number of estimators in the
        ensemble by a factor equal to the number of estimators.
        Note: this parameter can be specified on a layer-specific basis in the
        :attr:`add` method.

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

    Examples
    --------

    Instantiate ensembles with no preprocessing: use list of estimators

    >>> from mlens.ensemble import Subsemble
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = Subsemble()
    >>> ensemble.add([SVR(), ('can name some or all est', Lasso())])
    >>> ensemble.add(SVR(), meta=True)
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    9.2393246953908577

    Instantiate ensembles with different preprocessing pipelines through dicts.

    >>> from mlens.ensemble import Subsemble
    >>> from mlens.metrics.metrics import rmse
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
    >>> ensemble = Subsemble()
    >>> ensemble.add(estimators_per_case, preprocessing_cases).add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    9.0115741283454458
    """

    def __init__(self,
                 partitions=2,
                 folds=2,
                 shuffle=False,
                 random_state=None,
                 scorer=None,
                 raise_on_exception=True,
                 array_check=2,
                 verbose=False,
                 n_jobs=-1,
                 layers=None):

        super(Subsemble, self).__init__(
                shuffle=shuffle, random_state=random_state,
                scorer=scorer, raise_on_exception=raise_on_exception,
                verbose=verbose, n_jobs=n_jobs, layers=layers,
                array_check=array_check)

        self.partitions = partitions
        self.folds = folds

    def add_meta(self, estimators):
        """Add meta estimator."""
        return self.add(estimators, meta=True)

    def add(self, estimators, preprocessing=None, meta=False,
            partitions=None, folds=None, proba=False):
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

        meta : bool
            indicator if the layer added is the final meta estimator. This will
            prevent folded or blended fits of the estimators and only fit them
            once on the full input data.

        partitions : int, optional
            number of partitions to split data into. Increasing partitions
            increases the number of estimators in the layer by a factor equal
            to the number of estimators. Specifying this parameter overrides
            the ensemble-wide parameter.

        folds : int, optional
            Use if a different number of folds is desired than what the
            ensemble was instantiated with.

        proba : bool (default = False)
            whether to call ``predict_proba`` on base learners.

        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        if meta:
            cls = 'full'
            idx = FullIndex()
        else:
            cls = 'subset'
            p = partitions if partitions is not None else self.partitions
            c = folds if folds is not None else self.folds
            idx = SubsetIndex(p, c,
                              raise_on_exception=self.raise_on_exception)

        return self._add(cls=cls,
                         estimators=estimators,
                         preprocessing=preprocessing,
                         indexer=idx,
                         proba=proba,
                         verbose=self.verbose)
