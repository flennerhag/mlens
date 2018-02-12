"""ML-Ensemble

:author: Sebastian Flennerhag
:license: MIT
:copyright: 2017-2018

Layer module.
"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=too-many-instance-attributes
# pylint: disable=len-as-condition

from __future__ import division, print_function

from .base import OutputMixin, IndexMixin, BaseStacker
from ..utils import time, print_time, safe_print, format_name
from ..utils.exceptions import NotFittedError
from ..externals.joblib import delayed
from ..metrics import Data


GLOBAL_LAYER_NAMES = list()


class Layer(OutputMixin, IndexMixin, BaseStacker):

    r"""Layer of preprocessing pipes and estimators.

    Layer is an internal class that holds a layer and its associated data
    including an estimation procedure. It behaves as an estimator from an
    Scikit-learn API point of view.

    Parameters
    ----------
    propagate_features : list, range, optional
        Features to propagate from the input array to the output array.
        Carries input features to the output of the layer, useful for
        propagating original data through several stacked layers. Propagated
        features are stored in the left-most columns.

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

    def __init__(self, name=None, propagate_features=None, shuffle=False,
                 random_state=None, verbose=False, stack=None, **kwargs):
        if stack and not isinstance(stack, list):
            if stack.__class__.__name__.lower() == 'group':
                stack = [stack]
            else:
                raise ValueError(
                    "Expected stack to be a Group or a list of Groups. "
                    "Got %r" % type(stack))

        name = format_name(name, 'layer', GLOBAL_LAYER_NAMES)
        super(Layer, self).__init__(
            name=name, stack=stack, verbose=verbose, **kwargs)

        self.feature_span = None
        self.shuffle = shuffle
        self.random_state = random_state
        self.propagate_features = propagate_features

        self.n_feature_prop = 0
        if self.propagate_features:
            self.n_feature_prop = len(self.propagate_features)

        # Protect stack against changes
        self.__static__.append('stack')

    def __iter__(self):
        yield self

    def __call__(self, args, parallel):
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
        if not self.__stack__:
            raise ValueError("Layer instance (%s) not initialized. "
                             "Add learners before calling" % self.name)

        job = args['job']
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

            parallel(delayed(subtransformer, not _threading)()
                     for transformer in self.transformers
                     for subtransformer in transformer(args, 'auxiliary'))

            if self.verbose >= 2:
                print_time(t1, 'done', file=f)

        if self.verbose >= 2:
            safe_print(msg.format('Learners ...'), file=f, end=e2)
            t1 = time()

        parallel(delayed(sublearner, not _threading)()
                 for learner in self.learners
                 for sublearner in learner(args, 'main'))

        if self.verbose >= 2:
            print_time(t1, 'done', file=f)

        if job == 'fit':
            self.collect()

        if self.verbose:
            msg = "done" if self.verbose == 1 \
                else (msg + " {}").format(self.name, "done")
            print_time(t0, msg, file=f)

    def collect(self, path=None):
        """Collect cache estimators"""
        for transformer in self.transformers:
            transformer.collect(path)
        for learner in self.learners:
            learner.collect(path)

    def set_output_columns(self, X, y, job, n_left_concats=0):
        """Compatibility method for setting learner output columns"""
        start_index = mi = self.n_feature_prop
        for lr in self.learners:
            lr.set_output_columns(X, y, job, n_left_concats=start_index)
            start_index = lr.feature_span[1]
        mx = start_index
        self.feature_span = (mi, mx)

    def _setup_1_global(self, X, y, job, **kwargs):
        """Run setup on all dependencies"""
        if self.__no_output__:
            for gr in self.stack:
                for g in gr:
                    g.setup(X, y, job, skip=['0_index'], **kwargs)
        else:
            for tr in self.transformers:
                tr.setup(X, y, job, skip=['0_index'], **kwargs)

            start_index = mi = self.n_feature_prop
            for lr in self.learners:
                lr.setup(X, y, job, n_left_concats=start_index,
                         skip=['0_index'], **kwargs)
                start_index = lr.feature_span[1]

            mx = start_index
            self.feature_span = (mi, mx)

    @property
    def indexers(self):
        """Check indexer"""
        return [g.indexer for g in self.stack]

    @property
    def learners(self):
        """Generator for learners in layer"""
        return [lr for g in self.stack for lr in g.learners]

    @property
    def transformers(self):
        """Generator for learners in layer"""
        return [tr for g in self.stack for tr in g.transformers]

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
