"""ML-Ensemble

:author: Sebastian Flennerhag
:licence: MIT
:copyright: 2017

Layer module.
"""
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes
# pylint: disable=len-as-condition

from __future__ import division, print_function


import numpy as np

from .. import config
from ..metrics import build_scores
from ..parallel import Learner, Transformer
from ..externals.sklearn.base import BaseEstimator
from ..externals.joblib import delayed
from ..utils import (assert_correct_format,
                     check_instances,
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
    estimators: dict of lists or list
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

    cls : str
        type of layers. Should be the name of an accepted estimator class.

    meta : bool (default = False)
        flag for whether given layer is the final meta layer

    indexer : instance, optional
        Indexer instance to use. Defaults to the layer class indexer
        instantiated with default settings. Required arguments depend on the
        indexer. See :mod:`mlens.base` for details.

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

            preprocessing_cases = {"case-1": [trans_1, trans_2].
                                   "case-2": [alt_trans_1, alt_trans_2]}

            estimators = {"case-1": [est_a, est_b].
                          "case-2": [est_c, est_d]}

        The lists for each dictionary entry can be any of ``option_1``,
        ``option_2`` and ``option_3``.

    proba : bool (default = False)
        whether to call `predict_proba` on the estimators in the layer when
        predicting.

    partitions : int (default = 1)
        Number of subset-specific fits to generate from the learner library.

    propagate_features : list, optional
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

            - ``verbose = 2`` messages for each layer

        If ``verbose >= 50`` prints to ``sys.stdout``, else ``sys.stderr``.
        For verbosity in the layers themselves, use ``fit_params``.

    shuffle : bool (default = False)
        Whether to shuffle data before fitting layer.

    random_state : int, optional
        Random seed number to use for shuffling inputs

    dtype : numpy dtype class, default = :class:`numpy.float32`
        dtype format of prediction array.

    cls_kwargs : dict or None
        optional arguments to pass to the layer type class.

    Attributes
    ----------
    estimators\_ : OrderedDict, list
        container for fitted estimators, possibly mapped to preprocessing
        cases and / or folds.

    preprocessing\_ : OrderedDict, list
        container for fitted preprocessing pipelines, possibly mapped to
        preprocessing cases and / or folds.
    """

    def __init__(self,
                 estimators,
                 indexer,
                 preprocessing=None,
                 meta=False,
                 proba=False,
                 propagate_features=None,
                 scorer=None,
                 raise_on_exception=False,
                 name=None,
                 shuffle=False,
                 random_state=None,
                 dtype=None,
                 verbose=False,
                 cls_kwargs=None):
        self.meta = meta
        self.shuffle = shuffle
        self.verbose = verbose
        self.indexer = indexer
        self.scorer = scorer
        self.cls_kwargs = cls_kwargs
        self.random_state = random_state
        self.propagate_features = propagate_features
        self.raise_on_exception = raise_on_exception

        # Careful with these
        self._proba = proba
        self._predict_attr = 'predict' if not proba else 'predict_proba'

        self._classes = None
        self.n_pred = None
        self.n_prep = None
        self.n_est = None
        self.cases = None
        self.n_feature_prop = None
        self._name = name if name else 'layer'
        self.dtype = dtype if dtype is not None else config.DTYPE
        self._partitions = getattr(self.indexer, 'n_partitions', 1)
        self._set_learners(estimators, preprocessing)

    def __call__(self, parallel, args):
        """Process layer"""
        job = args['job']
        path = args['dir']

        if self.verbose >= 2:
            printout = "stderr" if self.verbose < 50 else "stdout"
            safe_print('Processing %s' % self._name, file=printout)
            t0 = time()

        if self._preprocess:
            parallel(delayed(subtransformer)(job, path)
                     for transformer in self.transformers
                     for subtransformer
                     in transformer(job, **args['transformer']))

        parallel(delayed(sublearner)(job, path)
                 for learner in self.learners
                 for sublearner in learner(job, **args['learner']))

        if args['job'] == 'fit':
            self.collect(args['dir'])

        if self.verbose >= 2:
            print_time(t0, '%s Done' % self._name, file=printout)

    def collect(self, path):
        """Collect cache estimators"""
        for transformer in self.transformers:
            transformer.fitted_pipelines = path
        for learner in self.learners:
            learner.fitted_estimators = path

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

    def _set_learners(self, estimators, preprocessing):
        """Set learners and preprocessing pipelines in layer"""
        assert_correct_format(estimators, preprocessing)
        self._estimators = check_instances(estimators, flatten_sort=True)
        self._preprocessing = check_instances(preprocessing)  # Don't flatten
        self._preprocess = self._preprocessing is not None

        self._learners = [
            Learner(estimator=est,
                    preprocess=learner_name.split('__')[0],
                    indexer=self.indexer,
                    name=learner_name,
                    attr=self._predict_attr,
                    output_columns=None,
                    scorer=self.scorer,
                    raise_on_exception=self.raise_on_exception)
            for learner_name, est in self._estimators
        ]

        self._transformers = [
            Transformer(pipeline=transformers,
                        name=prep_name,
                        indexer=self.indexer,
                        raise_on_exception=self.raise_on_exception)
            for prep_name, transformers in sorted(self._preprocessing.items())
        ]

        self._store_layer_data(
            check_instances(estimators, flatten_sort=False), preprocessing)

    @property
    def learners(self):
        """Generator for learners in layer"""
        for learner in self._learners:
            yield learner

    @learners.setter
    def learners(self, estimators, preprocessing=None):
        """Update learners in layer"""
        if preprocessing is None:
            preprocessing = self._preprocessing
        self._set_learners(estimators, preprocessing)

    @property
    def transformers(self):
        """Generator for learners in layer"""
        for transformer in self._transformers:
            yield transformer

    @transformers.setter
    def transformers(self, preprocessing, estimators=None):
        """Update learners in layer"""
        if estimators is None:
            estimators = self._estimators
        self._set_learners(estimators, preprocessing)

    @property
    def scores(self):
        """Cross validated scores"""
        if self.scorer is not None:
            scores = list()
            for learner in self.learners:
                scores.extend(learner.raw_scores)
            return build_scores(scores, self._partitions)

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

        col_index = 0
        col_map = list()
        for lr in self.learners:
            col_dict = dict()

            for partition_index in range(self._partitions):
                col_dict[partition_index] = col_index

                col_index += multiplier

            col_map.append([lr, col_dict])

        if col_index != n_prediction_features:
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
            if not isinstance(self.propagate_features, list):
                raise ValueError("propagate features expected list, got %s" %
                                 self.propagate_features.__class__)
            self.n_feature_prop = len(self.propagate_features)
        else:
            self.n_feature_prop = 0

        # Store layer estimator data
        if isinstance(ests, list):
            # No preprocessing cases. Check if there is one uniform pipeline.
            self.n_prep = 0 if prep is None or len(prep) == 0 else 1
            self.n_pred = len(ests)
            self.n_est = len(ests)
            self.cases = [None]
        else:
            # Get the number of predictions by moving through each
            # case and count estimators.
            self.n_prep = len(prep)
            self.cases = sorted(prep)

            n_pred = 0
            for case in self.cases:
                n_est = len(ests[case])
                setattr(self, '%s_n_est' % case, n_est)
                n_pred += n_est

            self.n_pred = self.n_est = n_pred

        if 'subset' in self.indexer.__class__.__name__.lower():
            self.n_pred *= self.indexer.n_partitions
            self.n_prep *= self.indexer.n_partitions

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean (default = True)
            If ``True``, will return the layers separately as individual
            parameters. If ``False``, will return the collapsed dictionary.

        Returns
        -------
        params : dict
            mapping of parameter names mapped to their values.
        """
        prefix = '%s__' % self._name if self._name else ''
        out = dict()
        for key in self._get_param_names():
            if key in ['estimators', 'preprocessing']:
                continue

            value = getattr(self, key, None)

            if deep and hasattr(value, 'get_params'):
                for k, val in value.get_params().items():
                    out['%s%s__%s' % (prefix, key, k)] = val
            out['%s%s' % (prefix, key)] = value

        if not deep:
            return out

        for step in [self.transformers, self.learners]:
            for obj in step:
                for key, value in obj.get_params().items():
                    if hasattr(value, 'get_params'):
                        for k, v in obj.get_params().items():
                            out["%s%s__%s" % (prefix, key, k)] = v
                    out["%s%s" % (prefix, key)] = value
        return out
