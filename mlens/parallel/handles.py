"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Handles for mlens.parallel objects.
"""
from .base import AttributeMixin
from .learner import Learner, Transformer
from ._base_functions import mold_objects, transform
from ..utils import format_name, check_instances
from ..utils.formatting import _check_instances
from ..externals.sklearn.base import BaseEstimator, clone

GLOBAL_GROUP_NAMES = list()
GLOBAL_PIPELINE_NAMES = list()


class Pipeline(BaseEstimator):

    """Transformer pipeline

    Pipeline class for wrapping a preprocessing pipeline of transformers.

    Parameters
    ----------
    pipeline: list, instance
        A transformer or a list of transformers. Valid input::

                option_1 = transformer_1
                option_2 = [transformer_1, transformer_2]
                option_3 = [("tr-1", transformer_1), ("tr-2", transformer_2)]
                option_4 = [transformer_1, ("tr-2", transformer_2)]

    name: str, optional
        name of pipeline.

    return_y: bool (default=False)
        If True, both X and y will be returned in a transform call.
    """

    def __init__(self, pipeline, name=None, return_y=False):
        self.name = format_name(name, 'pipeline', GLOBAL_PIPELINE_NAMES)
        self.pipeline = _check_instances(pipeline) if pipeline else None
        self.return_y = return_y
        self._pipeline = None

    def _run(self, fit, process, X, y=None):
        """Run job on pipeline."""
        out = self._check_empty(process, X, y)
        if out is not False:
            return out

        if fit:
            self._pipeline = [(tr_name, clone(tr))
                              for tr_name, tr in self.pipeline]

        for tr_name, tr in self._pipeline:
            if fit:
                tr.fit(X, y)

            if len(self._pipeline) > 1 or process:
                X, y = transform(tr, X, y)

        if process:
            if self.return_y:
                return X, y
            return X
        return self

    def _check_empty(self, process, X, y=None):
        """Check if empty pipeline and return vacuously"""
        # TODO: disallow None pipeline (modify model selection case handling)
        if self.pipeline:
            return False
        if not process:
            return self
        if self.return_y:
            return X, y
        return X

    def fit(self, X, y=None):
        """Fit pipeline.

        Note that the :class:`Pipeline` accepts both X and y arguments, and
        can return both X and y, depending on the transformers. The
        pipeline itself does no checks on the input.

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            Input data

        y: array-like of shape [n_samples, ]
            Targets

        Returns
        -------
        self: instance
            Fitted pipeline
        """
        return self._run(True, False, X, y)

    def transform(self, X, y=None):
        """Transform pipeline.

        Note that the :class:`Pipeline` accepts both X and y arguments, and
        can return both X and y, depending on the transformers. The
        pipeline itself does no checks on the input.

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            Input data

        y: array-like of shape [n_samples, ]
            Targets

        Returns
        -------
        X_processed: array-like of shape [n_samples, n_preprocessed_features]
            Preprocessed input data

        y: array-like of shape [n_samples, ], optional
            Preprocessed targets
        """
        return self._run(False, True, X, y)

    def fit_transform(self, X, y=None):
        """Fit and transform pipeline.

        Note that the :class:`Pipeline` accepts both X and y arguments, and
        can return both X and y, depending on the transformers. The
        pipeline itself does no checks on the input.

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            Input data

        y: array-like of shape [n_samples, ]
            Targets

        Returns
        -------
        X_processed: array-like of shape [n_samples, n_preprocessed_features]
            Preprocessed input data

        y: array-like of shape [n_samples, ], optional
            Preprocessed targets
        """
        return self._run(True, True, X, y)

    def get_params(self, deep=True):
        out = super(Pipeline, self).get_params(deep)
        if not deep:
            return out

        for tr_name, tr in self.pipeline:
            for k, v in tr.get_params(deep=True).items():
                out['%s__%s' % (tr_name, k)] = v
                out[tr_name] = tr
        return out


class Group(AttributeMixin, BaseEstimator):

    """Group

    Lightweight class for pairing a set of independent learners with
    a set of transformers that all share the same cross-validation strategy.
    All instances will share *the same* indexer. Allows cloning.

    Parameters
    ----------
    indexer: inst
        A :mod:`mlens.index` indexer to build learner and transformers on.

    learners: list, inst, optional
        :class:`Learner` instance(s) to build on indexer.

    transformers: list, inst, optional
        :class:`Transformer` instance(s) to build on indexer.

    name: str, optional
        name of group
    """

    def __init__(self, indexer, learners=None, transformers=None, name=None):
        self.__initialized__ = 0  # Unblock __setattr__

        self.name = format_name(name, 'group', GLOBAL_GROUP_NAMES)
        learners, transformers = mold_objects(learners, transformers)

        # Enforce common indexer
        self.indexer = indexer
        for o in learners + transformers:
            o.set_indexer(self.indexer)

        self.learners = learners
        self.transformers = transformers

        self.__initialized__ = 1  # Lock __setattr__

    def __iter__(self):
        for tr in self.transformers:
            yield tr
        for lr in self.learners:
            yield lr

    @property
    def __fitted__(self):
        """Fitted status"""
        return all([o.__fitted__ for o in self.learners + self.transformers])

    def get_params(self, deep=True):
        out = super(Group, self).get_params(deep)
        if not deep:
            return out

        out.update(self.indexer.get_params(deep))
        for obj in self:
            out.update(obj.get_params(deep))
        return out


def make_learners(indexer, estimators, preprocessing,
                  learner_kwargs=None, transformer_kwargs=None):
    """Set learners and preprocessing pipelines in layer"""
    preprocessing, estimators = check_instances(estimators, preprocessing)

    if learner_kwargs is None:
        learner_kwargs = {}
    if transformer_kwargs is None:
        transformer_kwargs = {}

    transformers = [Transformer(estimator=Pipeline(tr, return_y=True),
                                name=case_name, **transformer_kwargs)
                    for case_name, tr in preprocessing]

    learners = [Learner(estimator=est, preprocess=case_name,
                        name=learner_name, **learner_kwargs)
                for case_name, learner_name, est in estimators]

    return Group(indexer=indexer, learners=learners, transformers=transformers)
