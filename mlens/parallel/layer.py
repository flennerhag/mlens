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


import numpy as np


from .learner import Learner, Transformer
from .. import config
from ..metrics import Data
from ..externals.sklearn.base import BaseEstimator
from ..externals.joblib import delayed
from ..utils import (assert_correct_format,
                     check_instances,
                     clone_attribute,
                     print_time,
                     safe_print)
try:
    # Try get performance counter
    from time import perf_counter as time
except ImportError:
    # Fall back on wall clock
    from time import time


class Layer(BaseEstimator):

    r"""Layer of preprocessing pipes and estimators.

    Layer is an internal class that holds a layer and its associated data
    including an estimation procedure. It behaves as an estimator from an
    Scikit-learn API point of view.

    Parameters
    ----------
    estimators: dict, list
        estimators constituting the layer. If ``preprocessing`` is
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

            preprocessing_cases = {"case-1": [trans_1, trans_2].
                                   "case-2": [alt_trans_1, alt_trans_2]}

            estimators = {"case-1": [est_a, est_b].
                          "case-2": [est_c, est_d]}

        The lists for each dictionary entry can be any of ``option_1``,
        ``option_2`` and ``option_3``.

    indexer : instance
        Indexer instance to use. Defaults to the layer class indexer
        instantiated with default settings. Required arguments depend on the
        indexer. See :mod:`mlens.index` for details.

    preprocessing: dict, list, optional
        preprocessing pipelines for given layer. If
        the same preprocessing applies to all estimators, ``preprocessing``
        should be a list of transformer instances. The list can contain the
        instances directly, named tuples of transformers,
        or a combination of both. ::

            option_1 = [transformer_1, transformer_2]
            option_2 = [("trans-1", transformer_1),
                        ("trans-2", transformer_2)]
            option_3 = [transformer_1, ("trans-2", transformer_2)]

        In this case, a ``pr`` preprocessing case will be created for
        both the preprocessing pipeline and the estimators. If different
        preprocessing pipelines are desired, a dictionary
        that maps preprocessing pipelines must be passed. The names of the
        preprocessing dictionary must correspond to the names of the
        estimator dictionary. ::

            preprocessing_cases = {"case-1": [trans_1, trans_2].
                                   "case-2": [alt_trans_1, alt_trans_2]}

            estimators = {"case-1": [est_a, est_b].
                          "case-2": [est_c, est_d]}

        The lists for each dictionary entry can be any of ``option_1``,
        ``option_2`` and ``option_3``.

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

    dtype : numpy dtype class, default = :class:`numpy.float32`
        dtype format of prediction array.

    backend : str, optional
        backend to use when fitting layer. One of
        ``['multiprocessing', 'threading', 'sequential']``

    n_jobs : int (default = -1)
        degree of concurrency. Set to ``-1`` for maximum parallellism and
        ``1`` for sequential processing.
    """

    def __init__(self,
                 estimators,
                 indexer,
                 name,
                 preprocessing=None,
                 proba=False,
                 propagate_features=None,
                 scorer=None,
                 raise_on_exception=False,
                 shuffle=False,
                 random_state=None,
                 dtype=None,
                 n_jobs=-1,
                 backend=None,
                 verbose=False):
        self.name = name
        self.shuffle = shuffle
        self.indexer = indexer
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.propagate_features = propagate_features
        self.raise_on_exception = raise_on_exception

        # Careful with these
        self._verbose = verbose
        self._proba = proba
        self._predict_attr = 'predict' if not proba else 'predict_proba'

        self._classes = None
        self.n_pred = None
        self.n_prep = None
        self.n_est = None
        self.cases = None
        self.n_feature_prop = None
        self.cls = indexer.__class__.__name__.lower()[:-5]
        self.dtype = dtype if dtype is not None else config.DTYPE
        self.backend = backend if backend is not None else config.BACKEND
        self._partitions = getattr(self.indexer, 'partitions', 1)
        self._set_learners(estimators, preprocessing)

    def __iter__(self):
        """Provide jobs for ParallelProcess manager"""
        yield self

    def __call__(self, parallel, args):
        """Process layer

        Parameters
        ----------
        parallel : obj
            a ``mlens.externals.joblib.parallel.Parallel`` instance.

        args : dict
            dictionary with arguments. Expected to contain

            - ``job`` (str): one of ``fit``, ``predict`` or ``transform``

            - ``dir`` (str): path to cache

            - ``transformer`` (dict): kwargs for fitting transformers

            - ``learner`` (dict): kwargs for fitting learner(s)
        """
        job = args['job']
        path = args['dir']

        if self.verbose:
            msg = "{:<30}"
            f = "stdout" if self.verbose < 10 else "stderr"
            e1 = ' ' if self.verbose <= 1 else "\n"
            e2 = ' ' if self.verbose <= 2 else "\n"
            safe_print(msg.format('Processing %s' % self.name),
                       file=f, end=e1)
            t0 = time()

        if self._preprocess:
            if self.verbose >= 2:
                safe_print(msg.format('Preprocess pipelines ...'),
                           file=f, end=e2)
                t1 = time()

            parallel(delayed(subtransformer)(job, path)
                     for transformer in self.transformers
                     for subtransformer
                     in transformer(job, **args['transformer']))

            if self.verbose >= 2:
                print_time(t1, 'done', file=f)

        if self.verbose >= 2:
            safe_print(msg.format('Learners ...'), file=f, end=e2)
            t1 = time()

        parallel(delayed(sublearner)(job, path)
                 for learner in self.learners
                 for sublearner in learner(job, **args['learner']))

        if self.verbose >= 2:
            print_time(t1, 'done', file=f)

        if args['job'] == 'fit':
            self.collect(args['dir'])

        if self.verbose:
            msg = "done" if self.verbose == 1 \
                else (msg + " {}").format(self.name, "done")
            print_time(t0, msg, file=f)

    def collect(self, path):
        """Collect cache estimators"""
        for transformer in self.transformers:
            transformer.collect(path)
        for learner in self.learners:
            learner.collect(path)

    def _set_learners(self, estimators, preprocessing):
        """Set learners and preprocessing pipelines in layer"""
        # TODO: Refactor the formatting into a single function
        assert_correct_format(estimators, preprocessing)
        self._estimators, _flattened_estimators = check_instances(
            estimators, include_flattened=True)
        self._preprocessing = _preprocessing = check_instances(preprocessing)
        self._preprocess = len(self._preprocessing) != 0
        # XXX: This is a little bit of a hack, force create a case
        if isinstance(self._preprocessing, list):
            _preprocessing = {'pr': self._preprocessing}
            if self._preprocess:
                _flattened_estimators = [('pr', n, k)
                                         for _, n, k in _flattened_estimators]
        self._learners = [
            Learner(estimator=est,
                    preprocess=preprocess_name,
                    indexer=self.indexer,
                    name=learner_name,
                    attr=self._predict_attr,
                    output_columns=None,
                    scorer=self.scorer,
                    verbose=max(self.verbose - 2, 0),
                    raise_on_exception=self.raise_on_exception)
            for preprocess_name, learner_name, est in _flattened_estimators
        ]

        self._transformers = [
            Transformer(pipeline=transformers,
                        name=preprocess_name,
                        indexer=self.indexer,
                        verbose=max(self.verbose - 2, 0),
                        raise_on_exception=self.raise_on_exception)
            for preprocess_name, transformers
            in sorted(_preprocessing.items())
        ]

        self._store_layer_data(self._estimators, self._preprocessing)

    def set_output_columns(self, y=None):
        """Set output columns for learners"""
        # First make dummy allocation to check that it works out
        if self.proba:
            if y is not None:
                self.classes_ = y
            multiplier = self.classes_
        else:
            multiplier = 1
        n_prediction_features = self.n_pred * multiplier

        col_index = self.n_feature_prop
        col_map = list()
        sorted_learners = {lr.name:
                           lr for lr in self.learners}
        for _, lr in sorted(sorted_learners.items()):
            col_dict = dict()

            for partition_index in range(self._partitions):
                col_dict[partition_index] = col_index

                col_index += multiplier

            col_map.append([lr, col_dict])

        if col_index != n_prediction_features + self.n_feature_prop:
            # Note that since col_index is incremented at the end,
            # the largest index_value we have col_index - 1
            raise ValueError(
                "Mismatch feature size in prediction array (%i) "
                "and max column index implied by learner "
                "predictions sizes (%i)" %
                (n_prediction_features, col_index - 1))

        # Good to go
        for lr, col_dict in col_map:
            lr.output_columns = col_dict

    def _store_layer_data(self, ests, prep):
        """Utility for storing aggregate attributes about the layer."""
        # Store feature propagation data
        if self.propagate_features:
            if not isinstance(self.propagate_features, (list, range)):
                raise ValueError("propagate features expected list or range,"
                                 "got %s" % self.propagate_features.__class__)
            self.n_feature_prop = len(self.propagate_features)
        else:
            self.n_feature_prop = 0

        # Store layer estimator data
        if isinstance(ests, list):
            # No preprocessing cases. Check if there is one uniform pipeline.
            n_pred = len(ests)
            n_prep = 0 if not prep else 1
            self.cases = [None]
        else:
            # Get the number of predictions by moving through each
            # case and count estimators.
            n_prep = len(prep)
            self.cases = sorted(prep)

            n_pred = 0
            for case in self.cases:
                n_est = len(ests[case])
                setattr(self, '%s_n_est' % case, n_est)
                n_pred += n_est

        self.n_est = n_pred
        self.n_pred = n_pred * self.indexer.partitions
        self.n_prep = n_prep * self.indexer.partitions

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

        if not deep:
            return out

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
    def verbose(self):
        """Verbosity"""
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """Set verbosity"""
        if self._preprocess:
            for tr in self._transformers:
                tr.verbose = verbose
        for lr in self._learners:
            lr.verbose = verbose

    @property
    def estimators_(self):
        """Return copy of estimators"""
        return clone_attribute(self.learners, 'learner_')

    @property
    def estimators(self):
        """Return copy of estimators"""
        return self._estimators

    @estimators.setter
    def estimators(self, estimators, preprocessing=None):
        """Update learners in layer"""
        if preprocessing is None:
            preprocessing = self._preprocessing
        self._set_learners(estimators, preprocessing)

    @property
    def preprocessing_(self):
        """Return copy of preprocessing"""
        return clone_attribute(self.transformers, 'learner_')

    @property
    def preprocessing(self):
        """Return copy of preprocessing"""
        return self._preprocessing

    @preprocessing.setter
    def preprocessing(self, preprocessing, estimators=None):
        """Update learners in layer"""
        if estimators is None:
            estimators = self._estimators
        self._set_learners(estimators, preprocessing)

    @property
    def learners(self):
        """Generator for learners in layer"""
        for learner in self._learners:
            yield learner

    @property
    def transformers(self):
        """Generator for learners in layer"""
        for transformer in self._transformers:
            yield transformer

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

    @property
    def classes_(self):
        """Prediction classes during proba"""
        return self._classes

    @classes_.setter
    def classes_(self, y):
        """Set classes given input y"""
        if self.proba:
            self._classes = np.unique(y).shape[0]

    @property
    def proba(self):
        """Predict proba state"""
        return self._proba

    @proba.setter
    def proba(self, proba):
        """Update proba state"""
        self._proba = proba
        self._predict_attr = 'predict' if not proba else 'predict_proba'
