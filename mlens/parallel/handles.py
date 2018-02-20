"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Handles for mlens.parallel.
"""
from .base import BaseEstimator
from .learner import Learner, Transformer
from ._base_functions import mold_objects, transform
from ..utils import format_name, check_instances
from ..utils.formatting import _check_instances
from ..externals.sklearn.base import clone, BaseEstimator as _BaseEstimator

GLOBAL_GROUP_NAMES = list()
GLOBAL_PIPELINE_NAMES = list()


class Pipeline(_BaseEstimator):

    """Transformer pipeline

    Pipeline class for wrapping a preprocessing pipeline of transformers.

    .. versionadded: 0.2.0

    Parameters
    ----------
    pipeline : list, instance
        A :class:`~mlens.parallel.Transformer` instance or a list of
        :class:`~mlens.parallel.Transformer`
        instances. Accepted input formats::

            option_1 = transformer_1
            option_2 = [transformer_1, transformer_2]
            option_3 = [("tr-1", transformer_1), ("tr-2", transformer_2)]
            option_4 = [transformer_1, ("tr-2", transformer_2)]

    name : str, optional
        name of pipeline.

    return_y : bool, default = False
        If True, both X and y will be returned in a
        :func:`~mlens.parallel.handles.Pipeline.transform` call.
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
        # TODO: remove ability to set None pipelines: need to modify Evalutor

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
        X : array-like of shape [n_samples, n_features]
            Input data

        y : array-like of shape [n_samples, ]
            Targets

        Returns
        -------
        self : instance
            Fitted pipeline
        """
        return self._run(True, False, X, y)

    def transform(self, X, y=None):
        """Transform pipeline.

        Note that the :class:`Pipeline` accepts both X and y arguments, and
        can return both X and y, depending on the transformers.
        Pipeline itself does not checks the input.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            Input data

        y : array-like of shape [n_samples, ]
            Targets

        Returns
        -------
        X_processed : array-like of shape [n_samples, n_preprocessed_features]
            Preprocessed input data

        y : array-like of shape [n_samples, ], optional
            Original or preprocessed targets, depending on the transformers.
        """
        return self._run(False, True, X, y)

    def fit_transform(self, X, y=None):
        """Fit and transform pipeline.

        Note that the :class:`Pipeline` accepts both X and y arguments, and
        can return both X and y, depending on the transformers. The
        pipeline itself does no checks on the input.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            Input data

        y : array-like of shape [n_samples, ]
            Targets

        Returns
        -------
        X_processed : array-like of shape [n_samples, n_preprocessed_features]
            Preprocessed input data

        y : array-like of shape [n_samples, ], optional
            Preprocessed targets
        """
        return self._run(True, True, X, y)

    def get_params(self, deep=True):
        out = super(Pipeline, self).get_params(deep)
        if not deep:
            return out

        if self.pipeline:
            for tr_name, tr in self.pipeline:
                for k, v in tr.get_params(deep=True).items():
                    out['%s__%s' % (tr_name, k)] = v
                    out[tr_name] = tr
        return out


class Group(BaseEstimator):

    """A handle for learners and transformers that share a common indexer.

    Lightweight class for pairing a set of independent learners with
    a set of transformers that all share the same cross-validation strategy.
    A :class:`Group` instance is an acceptable caller to
    :class:`~mlens.parallel.ParallelProcessing`.

    .. versionadded:: 0.2.0

    .. note::
        All instances will share *the same* indexer. If instances have a
        different indexer, that indexer will be replaced.

    .. seealso::
        To run a :class:`Group` instance, see :func:`~mlens.parallel.wrapper.run`.
        To handle several groups, use the :class:`~mlens.parallel.layer.Layer`
        class.

    Parameters
    ----------
    indexer : inst, optional
        A :obj:`~mlens.index` indexer to build learner and transformers on.
        If not passed, the first indexer of the learners will be enforced
        on all instances.

    learners : list, inst, optional
        :class:`~mlens.parallel.learner.Learner` instance(s) attached to
        indexer. Note that :class:`Group` overrides previous
        ``indexer`` parameter settings.

    transformers : list, inst, optional
        :class:`~mlens.parallel.learner.Transformer` instance(s) attached to
        indexer. Note that :class:`Group` overrides previous
        ``indexer`` parameter settings.

    name : str, optional
        name of group

    **kwargs : optional
        Optional keyword arguments to the
        :class:`~mlens.parallel.base.BaseParallel` backend.
    """

    def __init__(self, indexer=None, learners=None, transformers=None,
                 name=None, **kwargs):
        name = format_name(name, 'group', GLOBAL_GROUP_NAMES)
        super(Group, self).__init__(name=name, **kwargs)

        learners, transformers = mold_objects(learners, transformers)
        if not indexer:
            indexer = learners[0].indexer

        # Enforce common indexer
        self.indexer = indexer
        for o in learners + transformers:
            o.set_indexer(self.indexer)

        self.learners = learners
        self.transformers = transformers

        self.__static__.extend(['indexer', 'learners', 'transformers'])

    def __iter__(self):
        # We update optional backend kwargs that might have been passed
        # to ensure these are passed to the instances
        backend_kwargs = {
            param: getattr(self, param)
            for param in ['dtype', 'verbose', 'raise_on_exception']
            if hasattr(self, param)
        }
        for tr in self.transformers:
            tr.set_params(**backend_kwargs)
            yield tr
        for lr in self.learners:
            lr.set_params(**backend_kwargs)
            yield lr

    @property
    def __fitted__(self):
        """Fitted status"""
        if not self._check_static_params():
            return False
        return all([o.__fitted__ for o in self.learners + self.transformers])

    def get_params(self, deep=True):
        out = super(Group, self).get_params(deep)
        if not deep:
            return out
        for item in self:
            for k, v in item.get_params(deep=deep).items():
                out['%s__%s' % (item.name, k)] = v
            out[item.name] = item
        return out


def make_group(indexer, estimators, preprocessing,
               learner_kwargs=None, transformer_kwargs=None, name=None):
    """Creating a :class:`Group` from a set learners and transformers

    Utility function for creating mapping a set of estimators and
    preprocessing pipelines to a :class:`Group` of
    :class:`~mlens.parallel.learner.Learner` and
    :class:`~mlens.parallel.learner.Transformer` instances.

    Parameters
    ----------
    indexer : instance or None, default = None
        Indexer instance to use. See :obj:`~mlens.index` for details.

    estimators : dict of lists or list of estimators.
        If ``preprocessing`` is ``None`` or ``list``, ``estimators`` should
        be a ``list``. The list can either contain estimator instances,
        named tuples of estimator instances, or a combination of both. ::

            option_1 = [estimator_1, estimator_2]
            option_2 = [("est-1", estimator_1), ("est-2", estimator_2)]
            option_3 = [estimator_1, ("est-2", estimator_2)]

        If different preprocessing pipelines are desired, a dictionary
        that maps estimators to preprocessing pipelines must be passed.
        The names of the estimator dictionary must correspond to the
        names of the estimator dictionary. ::

            preprocessing_cases = {"case-1": [trans_1, trans_2].
                                   "case-2": [alt_trans_1, alt_trans_2]}

            estimators = {"case-1": [est_a, est_b].
                          "case-2": [est_c, est_d]}

        The lists for each dictionary entry can be any of ``option_1``,
        ``option_2`` and ``option_3``.

    preprocessing : dict of lists or list, optional, default = None
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

            preprocessing_cases = {"case-1": [trans_1, trans_2].
                                   "case-2": [alt_trans_1, alt_trans_2]}

            estimators = {"case-1": [est_a, est_b].
                          "case-2": [est_c, est_d]}

        The lists for each dictionary entry can be any of ``option_1``,
        ``option_2`` and ``option_3``.

    transformer_kwargs : dict, optional
        Keyword arguments to pass to the
        :class:`~mlens.parallel.learner.Transformer` instances.

    learner_kwargs : dict, optional
        Keyword arguments to pass to the
        :class:`~mlens.parallel.learner.Learner` instances.

    name : str, optional
        Name of group. Should be unique.

    """
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

    group = Group(indexer=indexer, learners=learners,
                  transformers=transformers, name=name)
    return group
