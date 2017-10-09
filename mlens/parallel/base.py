"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Base classes for parallel estimation
"""
from abc import abstractmethod
import numpy as np

from .manager import ParallelProcessing
from ._base_functions import mold_objects
from .. import config
from ..externals.sklearn.base import BaseEstimator


class Group(BaseEstimator):

    """Group

    Lightweight class for pairing an estimator to a set of transformers and
    learners.
    """

    def __init__(self, indexer, learners, transformers):
        learners = mold_objects(learners, 'Learner')
        transformers = mold_objects(transformers, 'Transformer')

        self.indexer = indexer
        self.learners = learners
        self.transformers = transformers
        for obj in self.learners + self.transformers:
            obj.indexer = self.indexer

    def __iter__(self):
        for tr in self.transformers:
            yield tr
        for lr in self.learners:
            yield lr

    @property
    def __fitted__(self):
        """Fitted status"""
        return all([o.__fitted__ for o in self.learners + self.transformers])


class IndexMixin(object):

    """Indexer mixin

    Mixin for handling indexers.
    """

    def _check_indexer(self, indexer):
        """Check consistent indexer classes"""
        cls = indexer.__class__.__name__.lower()
        if 'index' not in cls:
            ValueError("Passed indexer does not appear to be valid indexer")

        lcls = [idx.__class__.__name__.lower() for idx in self._get_indexers()]
        if lcls:
            if 'blendindex' in lcls and cls != 'blendindex':
                raise ValueError(
                "Layer has blend indexers, but passed indexer if of full type")
            elif 'blendindex' not in lcls and cls == 'blendindex':
                raise ValueError(
               "Layer has full size indexers, passed indexer is of blend type")

    def _get_indexers(self):
        """Return list of indexers"""
        indexers = [getattr(self, 'indexer', None)]
        if None in indexers:
            indexers = getattr(self, 'indexers', [None])
        if None in indexers:
            raise AttributeError("No indexer or indexers attribute available")
        return indexers

    def _fit_indexers(self, X, y, job):
        indexers = self._get_indexers()
        for indexer in indexers:
            indexer.fit(X, y, job)


class OutputMixin(IndexMixin):

    """Output Mixin

    Mixin class for interfacing with ParallelProcessing when outputs are
    desired. Also implements generic fit, predict and transform methods.

    .. Note::
       To use this mixin the instance inheriting it must set the
       ``feature_span`` attribute in ``__init__``.
    """

    def fit(self, X, y, **kwargs):
        """Fit

        Fit learner(s) and transformer(s).

        Parameters
        ----------
        X: array
            input data
        y: array
            targets

        **kwargs: optional
            optional arguments to :attr:`ParallelProcessing.process`
        """
        return self._run('fit', X, y, **kwargs)

    def predict(self, X, **kwargs):
        """Predict

        Use final learner(s) to predict.

        Parameters
        ----------
        X: array,
            input data

        **kwargs: optional
            optional arguments to :attr:`ParallelProcessing.process`
        """
        return self._run('predict', X, **kwargs)

    def transform(self, X, **kwargs):
        """Transform

        Use cross-validation to predict.

        Parameters
        ----------
        X: array,
            input data

        **kwargs: optional
            optional arguments to :attr:`ParallelProcessing.process`
        """
        return self._run('transform', X, **kwargs)

    def _run(self, job, X, y=None, **kwargs):
        """Run job"""
        # Force __no_output__ to False for the run
        _np = self.__no_output__
        self.__no_output__ = False

        r = kwargs.pop('return_preds', False if job == 'fit' else True)
        verbose = max(getattr(self, 'verbose', 0) - 4, 0)
        backend = getattr(self, 'backend', config.BACKEND)
        n_jobs = getattr(self, 'n_jobs', -1)
        with ParallelProcessing(backend, n_jobs, verbose) as mgr:
            out = mgr.process(self, job, X, y, return_preds=r, **kwargs)

        self.__no_output__ = _np
        return out

    @abstractmethod
    def set_output_columns(self, X, y, n_left_concats=0):
        """Set output columns for prediction array"""
        pass

    def shape(self, job):
        """Prediction array shape"""
        if not self.feature_span:
            raise ValueError("Columns not set. Call set_output_columns.")
        return self.size(job), self.feature_span[1]

    def size(self, attr):
        """Get size of dim 0"""
        if attr not in ['n_test_samples', 'n_samples']:
            attr = 'n_test_samples' if attr != 'predict' else 'n_samples'

        indexers = self._get_indexers()
        sizes = list()
        for indexer in indexers:
            sizes.append(getattr(indexer, attr))

        sizes = np.unique(sizes)
        if not sizes.shape[0] == 1:
            raise ValueError(
                "Inconsistent output sizes generated by indexers.\n"
                 "All indexers need to generate same output size.\n"
                 "Got sizes %r from indexers %r" % (sizes.tolist(), indexers))

        return sizes[0]

    def _setup(self, X, y, job):
        """Setup instance for estimation"""
        self._fit_indexers(X, y, job)
        if not getattr(self, '__no_output__', False):
            self.set_output_columns(X, y)


class ProbaMixin(object):

    """"Probability Mixin

    Mixin for probability features on objects
    interfacing with :class:`ParallelProcessing`

    .. Note::
       To use this mixin the instance inheriting it must set the ``proba``
       attribute in ``__init__``.
    """

    def _check_proba_multiplier(self, y, alt=1):
        if self.proba:
            if y is not None:
                self.classes_ = y
            multiplier = self.classes_
        else:
            multiplier = alt
        return multiplier

    @property
    def _predict_attr(self):
        return 'predict' if not self.proba else 'predict_proba'

    @property
    def classes_(self):
        """Prediction classes during proba"""
        return self._classes

    @classes_.setter
    def classes_(self, y):
        """Set classes given input y"""
        if self.proba:
            self._classes = np.unique(y).shape[0]


class BaseBackend(object):

    """Base class for parallel backend

    Implements default backend settings.
    """

    def __init__(self, backend=None, n_jobs=-1, dtype=None,
                 raise_on_exception=True):
        self.n_jobs = n_jobs
        self.dtype = dtype if dtype is not None else config.DTYPE
        self.backend = backend if backend is not None else config.BACKEND
        self.raise_on_exception = raise_on_exception


class BaseParallel(BaseBackend, IndexMixin, BaseEstimator):

    """Base class for parallel objects

    Parameters
    ----------
    name : str
        name of instance. Should be unique.

    backend : str or object (default = 'threading')
        backend infrastructure to use during call to
        :class:`mlens.externals.joblib.Parallel`. See Joblib for further
        documentation. To set global backend, set ``mlens.config.BACKEND``.

    raise_on_exception : bool (default = True)
        whether to issue warnings on soft exceptions or raise error.
        Examples include lack of layers, bad inputs, and failed fit of an
        estimator in a layer. If set to ``False``, warnings are issued instead
        but estimation continues unless exception is fatal. Note that this
        can result in unexpected behavior unless the exception is anticipated.

    verbose : int or bool (default = False)
        level of verbosity.

    n_jobs : int (default = -1)
        Degree of concurrency in estimation. Set to -1 to maximize
        paralellization, while 1 runs on a single process (or thread
        equivalent). Cannot be overriden in the :attr:`add` method.

    dtype : obj (default = np.float32)
        data type to use, must be compatible with a numpy array dtype.
    """

    def __init__(self, name, *args, **kwargs):
        super(BaseParallel, self).__init__(*args, **kwargs)
        self.name = name

        # Flags for parallel
        self.__fitted__ = False      # Status for whether is fitted
        self.__no_output__ = False   # Status for whether to not expect outputs

    def __iter__(self):
        """Iterator for process manager"""
        yield self

    def _setup(self, X, y, job):
        """Setup instance for estimation"""
        self._fit_indexers(X, y, job)

    def get_params(self, deep=True):
        """Get learner parameters

        Parameters
        ----------
        deep : bool
            whether to return nested parameters
        """
        out = super(BaseParallel, self).get_params(deep=deep)
        for par_name in self._get_param_names():
            par = getattr(self, '_%s' % par_name, None)
            if par is None:
                par = getattr(self, par_name, None)
            if deep and hasattr(par, 'get_params'):
                for key, value in par.get_params(deep=True).items():
                    out['%s__%s' % (par_name, key)] = value
            out[par_name] = par

        for name in BaseBackend.__init__.__code__.co_varnames:
            if name not in ['self']:
                out[name] = getattr(self, name)
        return out
