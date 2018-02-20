"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Scikit-learn estimators of computational graph nodes. Estimator classes are
full Scikit-learn estimators that can be used in conjunction with other
standard estimators.
"""
from ..parallel import Layer, make_group
from ..parallel.base import ParamMixin
from ..parallel.wrapper import EstimatorMixin
from ..parallel._base_functions import check_stack
from ..externals.sklearn.base import BaseEstimator as _BaseEstimator
from ..externals.sklearn.base import clone, TransformerMixin


class BaseEstimator(EstimatorMixin, ParamMixin, _BaseEstimator):

    """Base class for estimators

    Ensure proper initialization of Mixins.
    """

    def __init__(self):
        self.__static__ = list()
        self._static_fit_params = dict()

    @property
    def __fitted__(self):
        """Fitted status"""
        if not self._backend:
            return False
        return self._check_static_params()

    @property
    def data(self):
        return self._backend.data

    @property
    def raw_data(self):
        return self._backend.raw_data


class LearnerEstimator(BaseEstimator):

    """Learner estimator

    Wraps an estimator in a cross-validation strategy.

    Parameters
    __________
    estimator: obj
        estimator to construct learner from.

    indexer: str, obj
        a cross-validation strategy. Either a :mod`mlens.index` indexer
        instance or a string.

    name: str, optional
        name of learner. If ``preprocessing`` is not ``None``,
        the name of the transformer will be prepended. If not specified, the
        name of the learner will be the name of the estimator in lower case.

    scorer: func, optional
        function to use for scoring predictions during cross-validated
        fitting.

    verbose: bool, int (default = False)
        whether to report completed fits.

    backend: str (default='threading')
        parallel backend. One of ``'treading'``, ``'multiprocessing'`` and
        ``'sequential'``.

    n_jobs: int (default=-1)
        degree of concurrency. Set to ``-1`` for maximum and ``1`` for
        sequential processing.
    """

    def __init__(self, estimator, indexer, verbose=False, scorer=None,
                 backend=None, n_jobs=-1, dtype=None):
        super(LearnerEstimator, self).__init__()

        self.estimator = estimator
        self.indexer = indexer
        self.verbose = verbose
        self.scorer = scorer
        self.backend = backend
        self.n_jobs = n_jobs
        self.dtype = dtype
        self._backend = None

        self.__static__.extend(['estimator', 'indexer'])

    def _build(self):
        """Build backend"""
        lr_kwargs = {'attr': 'predict', 'verbose': self.verbose,
                     'scorer': self.scorer, 'backend': self.backend,
                     'n_jobs': self.n_jobs, 'dtype': self.dtype}

        idx = clone(self.indexer)
        est = clone(self.estimator)
        group = make_group(idx, est, None, learner_kwargs=lr_kwargs)

        self._backend = group.learners[0]
        self._store_static_params()


class TransformerEstimator(TransformerMixin, BaseEstimator):
    """Transformer estimator

    Wraps an preprocessing pipeline in a cross-validation strategy.

    Parameters
    __________
    preprocessing: obj
        preprocessing pipeline to construct transformer from.

    indexer: str, obj
        a cross-validation strategy. Either a :mod`mlens.index` indexer
        instance or a string.

    name: str
        name of transformer.

    verbose: bool, int (default = False)
        whether to report completed fits.

    backend: str (default='threading')
        parallel backend. One of ``'treading'``, ``'multiprocessing'`` and
        ``'sequential'``.

    n_jobs: int (default=-1)
        degree of concurrency. Set to ``-1`` for maximum and ``1`` for
        sequential processing.
    """

    def __init__(self, preprocessing, indexer, verbose=False,
                 backend=None, n_jobs=-1, dtype=None):
        super(TransformerEstimator, self).__init__()

        self.preprocessing = preprocessing
        self.indexer = indexer
        self.verbose = verbose
        self.backend = backend
        self.n_jobs = n_jobs
        self.dtype = dtype
        self._backend = None

        self.__static__.extend(['preprocessing', 'indexer'])

    def _build(self):
        """Build backend"""
        tr_kwargs = {'backend': self.backend,
                     'n_jobs': self.n_jobs, 'verbose': self.verbose,
                     'dtype': self.dtype}

        idx = clone(self.indexer)
        prep = clone(self.preprocessing)

        group = make_group(idx, None, prep, transformer_kwargs=tr_kwargs)
        self._backend = group.transformers[0]
        self._store_static_params()


class LayerEnsemble(BaseEstimator):

    """One-layer ensemble

    An ensemble of estimators and preprocessing pipelines in one layer. Takes
    an input X and generates an output P. Assumes all preprocessing pipelines
    and all estimators are independent.

    Parameters
    ----------
    groups: list,
        list of :class:`Group` instances to build layer with.

    proba: bool (default = False)
        whether to call `predict_proba` on the estimators in the layer when
        predicting.

    propagate_features: list, range, optional
        Features to propagate from the input array to the output array.
        Carries input features to the output of the layer, useful for
        propagating original data through several stacked layers. Propagated
        features are stored in the left-most columns.

    verbose: int or bool (default = False)
        level of verbosity.

            - ``verbose = 0`` silent (same as ``verbose = False``)

            - ``verbose = 1`` messages at start and finish \
              (same as ``verbose = True``)

            - ``verbose = 2`` messages for preprocessing and estimators

            - ``verbose = 3`` messages for completed job

        If ``verbose >= 10`` prints to ``sys.stderr``, else ``sys.stdout``.

    shuffle: bool (default = False)
        Whether to shuffle data before fitting layer.

    random_state: obj, int, optional
        Random seed number to use for shuffling inputs
    """

    def __init__(
            self, groups, propagate_features=None, shuffle=False, n_jobs=-1,
            backend=None, verbose=False, random_state=None, dtype=None):
        super(LayerEnsemble, self).__init__()
        self.groups = groups if isinstance(groups, list) else [groups]
        self.propagate_features = propagate_features
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.backend = backend
        self.n_jobs = n_jobs
        self.dtype = dtype
        self._backend = None

        self.__static__.extend(['groups'])

    def _build(self):
        """Build Backend"""
        self._backend = Layer(
            propagate_features=self.propagate_features, dtype=self.dtype,
            shuffle=self.shuffle, random_state=self.random_state,
            verbose=self.verbose, stack=clone(self.groups))
        self._store_static_params()

    def push(self, *groups):
        """Push more groups onto stack

        Parameters
        ----------
        *groups : objects
            :class:`Group` instance(s) to push onto stack. Order is preserved
            (see pop).
        """
        check_stack(groups, self.groups)
        for group in groups:
            self.groups.append(group)
        return self

    def pop(self, idx):
        """Pop group

        Parameters
        ----------
        idx : int
            index of group to be popped in stack (0-based indexing).
            Equivalent to [group_0, group_1, ..., group_n].pop(idx)
        """
        return self.groups.pop(idx)

    def get_params(self, deep=True):
        out = super(LayerEnsemble, self).get_params(deep=deep)
        if not deep:
            return out

        for group in self.groups:
            for g in group:
                for key, value in g.get_params(deep=deep).items():
                    out["%s__%s" % (g.name, key)] = value
                    out[g.name] = g
        return out
