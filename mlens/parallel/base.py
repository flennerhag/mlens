"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Base classes for parallel estimation
"""
import numpy as np

from .. import config
from ..externals.sklearn.base import BaseEstimator


class BaseParallel(BaseEstimator):

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

    def __init__(self, name, backend=None, n_jobs=-1, dtype=None,
                 raise_on_exception=True):
        self.name = name
        self.n_jobs = n_jobs
        self.raise_on_exception = raise_on_exception
        self.dtype = dtype if dtype is not None else config.DTYPE
        self.backend = backend if backend is not None else config.BACKEND

        # Flags for parallel
        self._feature_size = None    # Feature size of instance
        self.__fitted__ = False      # Status for whether is fitted
        self.__no_output__ = False   # Status for whether to not expect outputs

    def __iter__(self):
        """Iterator for process manager"""
        yield self

    def get_params(self, deep=True):
        """Get learner parameters

        Parameters
        ----------
        deep : bool
            whether to return nested parameters
        """
        out = dict()
        for par_name in self._get_param_names():
            par = getattr(self, '_%s' % par_name, None)
            if par is None:
                par = getattr(self, par_name, None)
            if deep and hasattr(par, 'get_params'):
                for key, value in par.get_params(deep=True).items():
                    out['%s__%s' % (par_name, key)] = value
            out[par_name] = par

        return out


class ProbaMixin(object):

    """"Probability Mixin

    Mixin for probability features on objects
    interfacing with :class:`ParallelProcessing`

    Child class needs to set the ``proba`` attribute.
    """

    def __init__(self):
        self.proba = None
        self._classes = None

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
