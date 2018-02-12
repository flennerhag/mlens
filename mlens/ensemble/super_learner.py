"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Super Learner class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .base import BaseEnsemble
from ..index import FoldIndex, FullIndex


class SuperLearner(BaseEnsemble):

    r"""Super Learner class.

    The Super Learner (also known as the Stacking Ensemble)is an
    supervised ensemble algorithm that uses K-fold estimation to map a
    training set :math:`(X, y)` into a prediction set :math:`(Z, y)`,
    where the predictions
    in :math:`Z` are constructed using K-Fold splits of :math:`X` to ensure
    :math:`Z` reflects test errors, and that applies a user-specified meta
    learner to predict :math:`y` from :math:`Z`. The algorithm in sudo code
    follows:

        #. Specify a library :math:`L` of base learners

        #. Fit all base learners on :math:`X` and store the fitted estimators.

        #. Split :math:`X` into :math:`K` folds, fit every learner in
           :math:`L` on the training set and predict test set. Repeat until
           all folds have been predicted.

        #. Construct a matrix :math:`Z` by stacking the predictions per fold.

        #. Fit the meta learner on :math:`Z` and store the learner

    The ensemble can be used for prediction by mapping a new test set :math:`T`
    into a prediction set :math:`Z'` using the learners fitted in (2),
    and then mapping :math:`Z'` to :math:`y'` using the fitted meta learner
    from (5).

    The Super Learner does asymptotically as well as (up to a constant) an
    Oracle selector. For the theory behind the Super Learner, see
    [#]_ and [#]_ as well as references therein.

    Stacking K-fold predictions to cover an entire training set is a time
    consuming method and can be prohibitively costly for large datasets.
    With large data, other ensembles that fits an ensemble on subsets
    can achieve similar performance at a fraction of the training time.
    However, when data is noisy or of high variance,
    the :class:`SuperLearner` ensure all information is
    used during fitting.

    References
    ----------
    .. [#] van der Laan, Mark J.; Polley, Eric C.; and Hubbard, Alan E.,
       "Super Learner" (July 2007). U.C. Berkeley Division of Biostatistics
       Working Paper Series. Working Paper 222.
       http://biostats.bepress.com/ucbbiostat/paper222

    .. [#] Polley, Eric C. and van der Laan, Mark J.,
       "Super Learner In Prediction" (May 2010). U.C. Berkeley Division of
       Biostatistics Working Paper Series. Working Paper 266.
       http://biostats.bepress.com/ucbbiostat/paper266

    Notes
    -----
    This implementation uses the agnostic meta learner approach, where the
    user supplies the meta learner to be used. For the original Super Learner
    algorithm (i.e. learn the best linear combination of the base learners),
    the user can specify a linear regression as the meta learner.

    See Also
    --------
    :class:`BlendEnsemble`, :class:`Subsemble`


    .. note :: All parameters can be overriden in the :attr:`add` method unless
        otherwise specified. Notably, the ``backend`` and ``n_jobs`` cannot
        be altered in the :attr:`add` method.

    Parameters
    ----------
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
        Degree of parallel processing. Set to -1 for maximum parallelism and
        1 for sequential processing. Cannot be overriden in the :attr:`add` method.

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

    >>> from mlens.ensemble import SuperLearner
    >>> from mlens.metrics.metrics import rmse
    >>> from sklearn.datasets import load_boston
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.svm import SVR
    >>>
    >>> X, y = load_boston(True)
    >>>
    >>> ensemble = SuperLearner()
    >>> ensemble.add([SVR(), ('can name some or all est', Lasso())])
    >>> ensemble.add_meta(SVR())
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    6.955358...

    Instantiate ensembles with different preprocessing pipelines through dicts.

    >>> from mlens.ensemble import SuperLearner
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
    >>> ensemble = SuperLearner()
    >>> ensemble.add(estimators_per_case, preprocessing_cases).add(SVR(), meta=True)
    >>>
    >>> ensemble.fit(X, y)
    >>> preds = ensemble.predict(X)
    >>> rmse(y, preds)
    7.841329...
    """

    def __init__(
            self, folds=2, shuffle=False, random_state=None, scorer=None,
            raise_on_exception=True, array_check=2, verbose=False, n_jobs=-1,
            backend='threading', model_selection=False, sample_size=20, layers=None):
        super(SuperLearner, self).__init__(
            shuffle=shuffle, random_state=random_state, scorer=scorer,
            raise_on_exception=raise_on_exception, verbose=verbose,
            n_jobs=n_jobs, layers=layers, backend=backend,
            array_check=array_check, model_selection=model_selection,
            sample_size=sample_size)

        self.__initialized__ = 0  # Unlock parameter setting
        self.folds = folds
        self.__initialized__ = 1  # Protect against param resets

    def add_meta(self, estimator, **kwargs):
        """Meta Learner.

        Meta learner to be used for final predictions.

        Parameters
        ----------
        estimator : instance
            estimator instance.

        **kwargs : optional
            optional keyword arguments.
        """
        return self.add(estimators=estimator, meta=True, **kwargs)

    def add(self, estimators, preprocessing=None,
            proba=False, meta=False, propagate_features=None, **kwargs):
        """Add layer to ensemble.

        Parameters
        ----------
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

        proba : bool
            whether layer should predict class probabilities. Note: setting
            ``proba=True`` will attempt to call an the estimators
            ``predict_proba`` method.

        propagate_features : list, optional
            List of column indexes to propagate from the input of
            the layer to the output of the layer. Propagated features are
            concatenated and stored in the leftmost columns of the output
            matrix. The ``propagate_features`` list should define a slice of
            the numpy array containing the input data, e.g. ``[0, 1]`` to
            propagate the first two columns of the input matrix to the output
            matrix.

        meta : bool (default = False)
            indicator if the layer added is the final meta estimator. This will
            prevent folded or blended fits of the estimators and only fit them
            once on the full input data.

        **kwargs : optional
            optional keyword arguments.

        Returns
        -------
        self : instance
            ensemble instance with layer instantiated.
        """
        c = kwargs.pop('folds', self.folds)

        if meta:
            idx = FullIndex()
        else:
            idx = FoldIndex(c, raise_on_exception=self.raise_on_exception)

        return super(SuperLearner, self).add(
            estimators=estimators, indexer=idx, preprocessing=preprocessing,
            proba=proba, propagate_features=propagate_features, **kwargs)
