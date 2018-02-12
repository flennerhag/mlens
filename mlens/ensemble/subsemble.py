"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Subsemble class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .. import config
from .base import BaseEnsemble
from ..index import FullIndex, SubsetIndex, ClusteredSubsetIndex
from ..utils import kwarg_parser


class Subsemble(BaseEnsemble):

    r"""Subsemble class.

    Subsemble is a supervised ensemble algorithm that uses subsets of
    the full data to fit a layer, and within each subset K-fold estimation
    to map a training set :math:`(X, y)` into a prediction set :math:`(Z, y)`,
    where :math:`Z` is a matrix of prediction from each estimator on each
    subset (thus of shape ``[n_samples, (partitions * n_estimators)]``).
    :math:`Z` is constructed using K-Fold splits of each partition of `X` to
    ensure :math:`Z` reflects test errors within each partition. A final
    user-specified meta learner is fitted to the final ensemble layer's
    prediction, to learn the best combination of subset-specific estimator
    predictions. By passing a ``partition_estimator``, the partitions can be
    learnt. The algorithm in sudo code :

        #. For each layer in the ensemble, do:

            #. Specify a library of :math:`L` base learners
            #. Specify a partition strategy and partition :math:`X` into
               :math:`J` subsets.
            #. For each partition do:

                #. Fit all base learners and store them
                #. Create :math:`K` folds
                #. For each fold, do:

                   #. Fit all base learners on the training folds
                   #. Collect *all* test folds, across partitions, and predict.

            #. Assemble a cross-validated prediction matrix
               :math:`Z \in \mathbb{R}^{(n \times (L \times J))}`  by
               stacking predictions made in the cross-validation step.

        #. Fit the meta learner on :math:`Z` and store the learner.

    The ensemble can be used for prediction by mapping a new test set :math:`T`
    into a prediction set :math:`Z'` using the learners fitted in
    (1.3.1), and then using :math:`Z'` to generate final predictions through
    the fitted meta learner from (2).

    The Subsemble does asymptotically as well as (up to a constant) the
    Oracle selector. For the theory behind the Subsemble, see
    [#]_ and references therein.

    By partitioning the data into subset and fitting on those, a Subsemble
    can reduce training time considerably if estimators does not scale
    linearly. Moreover, Subsemble allows estimators to learn different
    patterns from each subset, and so can improve the overall performance
    by achieving a tighter fit on each subset. Since all observations in the
    training set are predicted, no information is lost between layers.

    This implementation allows very general partition estimators. The user
    must ensure that the partition estimator behaves as desired. To alter
    the expected behavior, see the ``kwd`` parameter under the :attr:`add` method
    and the :class:`mlens.base.ClusteredSubsetIndex`. Also see
    the `advanced tutorials <http://mlens.readthedocs.io/en/latest/ensemble_tutorial.html#advanced-subsemble-techniques>`_
    for example use cases.

    References
    ----------
    .. [#] Sapp, S., van der Laan, M. J., & Canny, J. (2014).
       Subsemble: an  ensemble method for combining subset-specific algorithm
       fits. Journal of Applied Statistics, 41(6), 1247-1259.
       http://doi.org/10.1080/02664763.2013.864263

    See Also
    --------
    :class:`BlendEnsemble`, :class:`SuperLearner`


    .. note :: All parameters can be overriden in the :attr:`add` method unless
        otherwise specified. Notably, the ``backend`` and ``n_jobs`` cannot
        be altered in the :attr:`add` method.

    Parameters
    ----------
    partitions : int (default = 2)
        number of partitions to split data into. For each layer,
        increasing partitions increases the number of estimators in the
        ensemble by a factor equal to the number of estimators.
        Note: this parameter can be specified on a layer-specific basis in the
        :attr:`add` method.

    partition_estimator : instance, optional
        To use a supervised or unsupervised estimator to learn partitions,
        pass an instantiated estimator as ``partition_estimator``. The
        estimator must accept a ``fit`` call for fitting the training data,
        and a ``predict`` call that *assigns cluster partitions labels*.
        For instance, clustering estimator or classifiers (where their class
        predictions will be used for partitioning). The number of partitions
        by the estimator must correspond to the ``partitions`` argument.
        Specific estimators can be added to each layer by passing the
        estimator during the call to the ensemble's :attr:`add` method.

    folds : int (default = 2)
        number of folds to use during fitting. Note: this parameter can be
        specified on a layer-specific basis in the :attr:`add` method.

    shuffle : bool (default = False)
        whether to shuffle data before before processing each layer. This
        parameter can be overridden in the :attr:`add` method if different test
        sizes is desired for each layer.

    random_state : int (default = None)
        random seed for shuffling inputs. Note that the seed here is used to
        generate a unique seed for each layer. Can be overridden in the
        :attr:`add` method.

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

            - ``array_check = 1`` will check ``X`` and ``y`` for
              inconsistencies and warn when format looks suspicious,
              but retain original format.

            - ``array_check = 2`` will impose Scikit-learn array checks,
              which converts ``X`` and ``y`` to numpy arrays and raises
              an error if conversion fails.

    verbose : int or bool (default = False)
        level of verbosity.

            * ``verbose = 0`` silent (same as ``verbose = False``)

            * ``verbose = 1`` messages at start and finish (same as
              ``verbose = True``)

            * ``verbose = 2`` messages for each layer

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    n_jobs : int (default = -1)
        Degree of concurrency in estimation. Set to -1 to maximize
        paralellization, while 1 runs on a single process (or thread
        equivalent). Cannot be overriden in the :attr:`add` method.

        .. note::

            A high degree of partitioning can incur a thread overload that can
            in certain cases overwhelm OpenBLAS. If any of your estimators
            rely on OpenBLAS and you experience crashed, set
            ``n_jobs`` to a lower (i.e. -2). In these cases, this will actually
            not impact performance since the issues stems from having too many
            threads active, so lowering the count avoids the bottleneck.
            Reference: https://github.com/xianyi/OpenBLAS/issues/889

    backend : str or object (default = 'threading')
        backend infrastructure to use during call to
        :class:`mlens.externals.joblib.Parallel`. See Joblib for further
        documentation. To set global backend, set ``mlens.config._BACKEND``.
        Cannot be overriden in the :attr:`add` method.

    model_selection: bool (default=False)
        Whether to use the ensemble in model selection mode. If ``True``,
        this will alter the ``transform`` method. When calling ``transform``
        on new data, the ensemble will call ``predict``, while calling
        ``transform`` with the training data reproduces predictions from the
        ``fit`` call. Hence the ensemble can be used as a pure transformer
        in a preprocessing pipeline passed to the :class:`Evaluator`, as
        training folds are faithfully reproduced as during a ``fit``call and
        test folds are transformed with the ``predict`` method.

    sample_size: int (default=20)
        size of training set sample
        (``[min(sample_size, X.size[0]), min(X.size[1], sample_size)]``)

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
    9.2393246...

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
    9.0115741...
    """

    def __init__(
            self, partitions=2, partition_estimator=None, folds=2,
            shuffle=False, random_state=None, scorer=None,
            raise_on_exception=True, array_check=2, verbose=False, n_jobs=-1,
            backend=None, model_selection=False, sample_size=20, layers=None):
        super(Subsemble, self).__init__(
            shuffle=shuffle, random_state=random_state, scorer=scorer,
            raise_on_exception=raise_on_exception, verbose=verbose,
            n_jobs=n_jobs, layers=layers, model_selection=model_selection,
            sample_size=sample_size, array_check=array_check, backend=backend)

        self.__initialized__ = 0  # Unlock parameter setting
        self.partition_estimator = partition_estimator
        self.partitions = partitions
        self.folds = folds
        self.__initialized__ = 1  # Protect against param resets

    def add_meta(self, estimator, **kwargs):
        """Add meta estimator.

        Parameters
        ----------
        estimator : instance
            estimator instance.

        **kwargs : optional
            optional keyword arguments.
        """
        return self.add(estimator, meta=True, **kwargs)

    def add(self, estimators, preprocessing=None, meta=False,
            partitions=None, partition_estimator=None, folds=None, proba=False,
            propagate_features=None, **kwargs):
        r"""Add layer to ensemble.

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

        partition_estimator : instance, optional
            To use a supervised or unsupervised estimator to learn partitions,
            pass an instantiated estimator as ``partition_estimator``. The
            estimator must accept a ``fit`` call for fitting the training data,
            and a ``predict`` call that *assigns cluster partitions labels*.
            For instance, clustering estimator or classifiers (where class
            predictions will be used for partitioning). The number of
            partitions by the estimator must correspond to the layer's
            ``partitions`` argument. Passing an estimator here supersedes any
            other estimator previously passed.

        folds : int, optional
            Use if a different number of folds is desired than what the
            ensemble was instantiated with.

        proba : bool (default = False)
            whether to call ``predict_proba`` on base learners.

        propagate_features : list, optional
            List of column indexes to propagate from the input of
            the layer to the output of the layer. Propagated features are
            concatenated and stored in the leftmost columns of the output
            matrix. The ``propagate_features`` list should define a slice of
            the numpy array containing the input data, e.g. ``[0, 1]`` to
            propagate the first two columns of the input matrix to the output
            matrix.

        **kwargs : optional
            optional keyword arguments to instantiate ensemble with. In
            particular, keywords for clustered subsemble learning

                * **fit_estimator** *(Bool, default = True)* -
                  whether to call ``fit`` on the partition estimator.

                * **attr** *(str, default = 'predict')* -
                  the method attribute to call for generating partition ids
                  for the input data.

                * **partition_on** *(str, default = 'X')* -
                  the input data for the ``attr`` method.
                  One of ``'X'``, ``'y'`` or ``'both'``.


        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        if meta:
            idx = FullIndex()
        else:
            # Parse arguments for the indexer
            p = partitions if partitions is not None else self.partitions
            e = partition_estimator if partition_estimator is not None \
                else self.partition_estimator
            c = folds if folds is not None else self.folds

            idx = ClusteredSubsetIndex if e is not None else SubsetIndex
            args = (e, p, c) if e is not None else (p, c)

            kwargs_idx, kwargs = kwarg_parser(idx.__init__, kwargs)

            if 'raise_on_exception' not in kwargs_idx:
                kwargs_idx['raise_on_exception'] = self.raise_on_exception

            idx = idx(*args, **kwargs_idx)

        return super(Subsemble, self).add(
            estimators=estimators, preprocessing=preprocessing, indexer=idx,
            proba=proba, propagate_features=propagate_features, **kwargs)
