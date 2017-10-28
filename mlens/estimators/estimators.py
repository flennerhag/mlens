"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Front-end classes
"""
from ..parallel import Layer, make_learners
from ..parallel.base import AttributeMixin
from ..parallel.wrapper import EstimatorMixin
from ..externals.sklearn.base import BaseEstimator, clone, TransformerMixin


class LearnerEstimator(EstimatorMixin, AttributeMixin, BaseEstimator):

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

    def __init__(self, estimator, indexer, name=None,
                 verbose=False, scorer=None, backend=None, n_jobs=-1):
        # Set __initialized__ to False to unblock __setattr__
        self.__initialized__ = 0

        # Params
        self.estimator = estimator
        self.indexer = indexer
        self.name = name
        self.verbose = verbose
        self.scorer = scorer
        self.backend = backend
        self.n_jobs = n_jobs

        # Backend
        lr_kwargs = {'attr': 'predict', 'verbose': self.verbose,
                     'scorer': self.scorer, 'backend': self.backend,
                     'n_jobs': self.n_jobs}

        if self.name:
            est = [(self.name, clone(self.estimator))]
        else:
            est = clone(self.estimator)
        group = make_learners(
            clone(self.indexer), est, None, learner_kwargs=lr_kwargs)

        _learner = group.learners[0]
        self._backend = _learner

        # Set __initialized__ to True to protect __setattr__ of parameters
        self.__initialized__ = 1

    @property
    def data_(self):
        return self._backend.data

    @property
    def raw_data_(self):
        return self._backend.raw_data


class TransformerEstimator(
        TransformerMixin, EstimatorMixin, AttributeMixin, BaseEstimator):
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

    def __init__(self, preprocessing, indexer, name=None, verbose=False,
                 backend=None, n_jobs=-1):
        # Set __initialized__ to False to unblock __setattr__
        self.__initialized__ = 0

        # Params
        self.preprocessing = preprocessing
        self.indexer = indexer
        self.name = name
        self.verbose = verbose
        self.backend = backend
        self.n_jobs = n_jobs

        # Backend
        tr_kwargs = {'backend': self.backend,
                     'n_jobs': self.n_jobs, 'verbose': self.verbose}

        if isinstance(self.preprocessing, list):
            prep = list()
            for u in self.preprocessing:
                if isinstance(u, (tuple, list)):
                    prep.append((u[0], clone(u[1])))
                else:
                    prep.append(clone(u))
        else:
            prep = clone(self.preprocessing)

        group = make_learners(
            clone(self.indexer), None, prep, transformer_kwargs=tr_kwargs)
        self._backend = group.transformers[0]

        # Set __initialized__ to True to protect __setattr__ of parameters
        self.__initialized__ = 1

    @property
    def data_(self):
        return self._backend.data

    @property
    def raw_data_(self):
        return self._backend.raw_data


class LayerEnsemble(EstimatorMixin, AttributeMixin, BaseEstimator):

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

    def __init__(self, groups, propagate_features=None, shuffle=False,
                 random_state=None, verbose=False, name=None, backend=None,
                 n_jobs=-1):
        self.__initialized__ = 0
        self.groups = groups if isinstance(groups, list) else [groups]
        self.propagate_features = propagate_features
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.backend = backend
        self.n_jobs = n_jobs

        name = name if name else 'layer'
        self.name = name
        self._backend = Layer(
            name=self.name, propagate_features=self.propagate_features,
            shuffle=self.shuffle, random_state=self.random_state,
            verbose=self.verbose, groups=self.groups)
        self.__initialized__ = 1

    def push(self, *groups):
        """Push more groups onto stack

        Parameters
        ----------
        *groups : objects
            :class:`Group` instance(s) to push onto stack. Order is preserved
            (see pop).
        """
        self._backend.push(*groups)

    def pop(self, idx):
        """Pop group

        Parameters
        ----------
        idx : int
            index of group to be popped in stack (0-based indexing).
            Equivalent to [group_0, group_1, ..., group_n].pop(idx)
        """
        self._backend.pop(idx)

    def get_params(self, deep=True):
        out = super(LayerEnsemble, self).get_params(deep=deep)
        if not deep:
            return out

        for group in self.groups:
            for g in group:
                for key, value in g.get_params(deep=deep).items():
                    out["%s.%s" % (g.name, key)] = value
        return out
