"""ML-Ensemble

:author: Sebastian Flennerhag
:licence: MIT
:copyright: 2017

Layer module.
"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=len-as-condition

from __future__ import division, print_function

from .base import BaseParallel, OutputMixin
from ..metrics import Data
from ..utils import print_time, safe_print
from ..utils.exceptions import NotFittedError
from ..externals.joblib import delayed
try:
    # Try get performance counter
    from time import perf_counter as time
except ImportError:
    # Fall back on wall clock
    from time import time


class Layer(OutputMixin, BaseParallel):

    r"""Layer of preprocessing pipes and estimators.

    Layer is an internal class that holds a layer and its associated data
    including an estimation procedure. It behaves as an estimator from an
    Scikit-learn API point of view.

    Parameters
    ----------
    proba : bool (default = False)
        whether to call `predict_proba` on the estimators in the layer when
        predicting.

    propagate_features : list, range, optional
        Features to propagate from the input array to the output array.
        Carries input features to the output of the layer, useful for
        propagating original data through several stacked layers. Propagated
        features are stored in the left-most columns.

    raise_on_exception : bool (default = False)
        whether to raise an error on soft exceptions, else issue warning.

    verbose : int or bool (default = False)
        level of verbosity.

            - ``verbose = 0`` silent (same as ``verbose = False``)

            - ``verbose = 1`` messages at start and finish \
              (same as ``verbose = True``)

            - ``verbose = 2`` messages for preprocessing and estimators

            - ``verbose = 3`` messages for completed job

        If ``verbose >= 10`` prints to ``sys.stderr``, else ``sys.stdout``.

    shuffle : bool (default = False)
        Whether to shuffle data before fitting layer.

    random_state : obj, int, optional
        Random seed number to use for shuffling inputs

    **kwargs : optional
        optional arguments to :class:`BaseParallel`.
    """

    def __init__(self, name, propagate_features=None, shuffle=False,
                 random_state=None, verbose=False, groups=None, **kwargs):
        super(Layer, self).__init__(name=name, **kwargs)
        self.feature_span = None
        self.shuffle = shuffle
        self._verbose = verbose
        self.random_state = random_state
        self.propagate_features = propagate_features

        self.n_feature_prop = 0
        if self.propagate_features:
            self.n_feature_prop = len(self.propagate_features)

        self.groups = list() if not groups else groups
        self.__initialized__ = False if not groups else True

    def __call__(self, parallel, args):
        """Process layer

        Parameters
        ----------
        parallel : obj
            a ``Parallel`` instance.

        args : dict
            dictionary with arguments. Expected to contain

            - ``job`` (str): one of ``fit``, ``predict`` or ``transform``

            - ``dir`` (str): path to cache

            - ``auxiliary`` (dict): kwargs for supporting transformer(s)s

            - ``learner`` (dict): kwargs for learner(s)
        """
        if not self.__initialized__:
            raise ValueError(
                "Layer instance (%s) not initialized. "
                "Add learners before calling" % self.name)

        job = args['job']
        path = args['dir']
        _threading = self.backend == 'threading'

        if job != 'fit' and not self.__fitted__:
            raise NotFittedError(
                "Layer instance (%s) not fitted." % self.name)

        if self.verbose:
            msg = "{:<30}"
            f = "stdout" if self.verbose < 10 else "stderr"
            e1 = ' ' if self.verbose <= 1 else "\n"
            e2 = ' ' if self.verbose <= 2 else "\n"
            safe_print(msg.format('Processing %s' % self.name),
                       file=f, end=e1)
            t0 = time()

        if self.transformers:
            if self.verbose >= 2:
                safe_print(msg.format('Preprocess pipelines ...'),
                           file=f, end=e2)
                t1 = time()

            parallel(delayed(subtransformer, not _threading)(path)
                     for transformer in self.transformers
                     for subtransformer
                     in getattr(transformer, 'gen_%s' % job)(
                         **args['auxiliary'])
                     )

            if self.verbose >= 2:
                print_time(t1, 'done', file=f)

        if self.verbose >= 2:
            safe_print(msg.format('Learners ...'), file=f, end=e2)
            t1 = time()

        parallel(delayed(sublearner, not _threading)(path)
                 for learner in self.learners
                 for sublearner in getattr(learner, 'gen_%s' % job)(
                     **args['estimator'])
                 )

        if self.verbose >= 2:
            print_time(t1, 'done', file=f)

        if job == 'fit':
            self.collect(path)

        if self.verbose:
            msg = "done" if self.verbose == 1 \
                else (msg + " {}").format(self.name, "done")
            print_time(t0, msg, file=f)

    def push(self, *groups):
        """Push an indexer and associated learners and transformers"""
        for group in groups:
            self._check_indexer(group.indexer)
            self.groups.append(group)
        self.__initialized__ = True
        return self

    def pop(self, idx):
        """Pop a previous push with index idx"""
        gr = self.groups.pop(idx)
        if not self.groups:
            self.__initialized__ = False
        return gr

    def collect(self, path):
        """Collect cache estimators"""
        for transformer in self.transformers:
            transformer.collect(path)
        for learner in self.learners:
            learner.collect(path)

    def set_output_columns(self, X=None, y=None, n_left_concats=0):
        """Set output columns for learners"""
        start_index = mi = n_left_concats + self.n_feature_prop
        for learner in self.learners:
            learner.set_output_columns(X, y, start_index)
            start_index = learner.feature_span[1]

        mx = start_index
        self.feature_span = (mi, mx)

    def get_params(self, deep=True):
        """Get learner parameters

        Parameters
        ----------
        deep : bool
            whether to return nested parameters
        """
        out = super(Layer, self).get_params(deep=deep)
        if not deep:
            return out

        for i, idx in enumerate(self.indexers):
            for k, v in idx.get_params(deep=deep).items():
                out['indexer-%i__%s' % (i, k)] = v

        for step in [self.transformers, self.learners]:
            for obj in step:
                obj_name = obj.name
                for key, value in obj.get_params(deep=deep).items():
                    if hasattr(value, 'get_params'):
                        for k, v in obj.get_params(deep=deep).items():
                            out["%s__%s" % (obj_name, k)] = v
                    out["%s__%s" % (obj_name, key)] = value
                out[obj_name] = obj
        return out

    @property
    def indexers(self):
        """Check indexer"""
        return [g.indexer for g in self.groups]

    @property
    def learners(self):
        """Generator for learners in layer"""
        return [lr for g in self.groups for lr in g.learners]

    @property
    def transformers(self):
        """Generator for learners in layer"""
        return [tr for g in self.groups for tr in g.transformers]

    @property
    def __fitted__(self):
        """Fitted status"""
        if not self.groups:
            # all on an empty list yields True
            return False
        return all([g.__fitted__ for g in self.groups])

    @__fitted__.setter
    def __fitted__(self, val):
        """Compatibility"""
        pass

    @property
    def verbose(self):
        """Verbosity"""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Set verbosity"""
        for g in self.groups:
            for obj in g:
                obj.verbose = verbose

    @property
    def data(self):
        """Cross validated scores"""
        return Data(self.raw_data)

    @property
    def raw_data(self):
        """Cross validated scores"""
        data = list()

# TODO: Fix table printing
#        if self._preprocess:
#            for transformer in self.transformers:
#                data.extend(transformer.raw_data)

        for learner in self.learners:
            data.extend(learner.raw_data)
        return data
