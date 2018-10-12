"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Temporal ensemble class. Fully integrable with Scikit-learn.
"""

from __future__ import division

from .base import BaseEnsemble
from ..index import TemporalIndex, FullIndex


class TemporalEnsemble(BaseEnsemble):

    r"""Temporal ensemble class.

    The temporal ensemble class uses a time series cross-validation
    strategy to create training and test folds that preserve temporal
    ordering in the data. The cross validation strategy is unrolled
    through time. For instance:

    ====  =================  ==========
    fold  train obs          test obs
    ====  =================  ==========
    0     0, 1, 2, 3         4
    1     0, 1, 2, 3, 4      5
    2     0, 1, 2, 3, 4, 5   6
    ====  =================  ==========

    Different estimators in the ensemble can operate on different time scales,
    allow efficient combinations of different temporal patterns in one model.

    See Also
    --------
    :class:`SuperLearner`, :class:`BlendEnsemble`, :class:`SequentialEnsemble`


    .. note :: All parameters can be overriden in the :attr:`add` method unless
        otherwise specified. Notably, the ``backend`` and ``n_jobs`` cannot
        be altered in the :attr:`add` method.

    Parameters
    ----------
    step_size : int (default=1)
        number of samples to use in each test fold. The final window
        size may be smaller if too few observations remain.

    burn_in : int (default=None)
        number of samples to use for first training fold. These observations
        will be dropped from the output. Defaults to ``step_size``.

    window: int (default=None)
        number of previous samples to use in each training fold, except first
        which is determined by ``burn_in``. If ``None``, will use all previous
        observations.

    lag: int (default=0)
        distance between the most recent training point in the training fold and
        the first test point. For ``lag>0``, the training fold and the test fold
        will not be contiguous.

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
    >>> from sklearn.linear_model import LinearRegression
    >>> from mlens.ensemble import TemporalEnsemble
    >>> import numpy as np
    >>>
    >>> x = np.linspace(0, 1, 100)
    >>> y = x[1:]
    >>> x = x[:-1]
    >>> x = x.reshape(-1, 1)
    >>>
    >>> ens = TemporalEnsemble(window=1)
    >>> ens.add(LinearRegression())
    >>>
    >>> ens.fit(x, y)
    >>> p = ens.predict(x)
    >>>
    >>>
    >>> print("{:5} | {:5}".format('pred', 'truth'))
    >>> for i in range(5, 10):
    ...     print("{:.3f} | {:.3f}".format(p[i], y[i]))
    >>>
    pred  | truth
    0.061 | 0.061
    0.071 | 0.071
    0.081 | 0.081
    0.091 | 0.091
    0.101 | 0.101
    """

    def __init__(
            self, step_size=1, burn_in=None, window=None, lag=0, scorer=None,
            raise_on_exception=True, array_check=None, verbose=False, n_jobs=-1,
            backend='threading', model_selection=False, sample_size=20, layers=None):
        super(TemporalEnsemble, self).__init__(
            shuffle=False, random_state=None, scorer=scorer,
            raise_on_exception=raise_on_exception, verbose=verbose,
            n_jobs=n_jobs, layers=layers, backend=backend,
            array_check=array_check, model_selection=model_selection,
            sample_size=sample_size)

        self.__initialized__ = 0  # Unlock parameter setting
        self.step_size = step_size
        self.burn_in = burn_in
        self.window = window
        self.lag = lag
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
        s = kwargs.pop('step_size', self.step_size)
        b = kwargs.pop('burn_in', self.burn_in)
        w = kwargs.pop('window', self.window)
        l = kwargs.pop('lag', self.lag)
        if meta:
            idx = FullIndex()
        else:
            idx = TemporalIndex(
                s, b, w, l, raise_on_exception=self.raise_on_exception)

        return super(TemporalEnsemble, self).add(
            estimators=estimators, indexer=idx, preprocessing=preprocessing,
            proba=proba, propagate_features=propagate_features, **kwargs)
