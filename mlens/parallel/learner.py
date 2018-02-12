"""ML-Ensemble

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Computational graph nodes. Job generator classes spawning jobs and executing
estimation on cross-validation sub-graphs.
"""
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from __future__ import print_function, division

import warnings
from copy import deepcopy
from abc import ABCMeta, abstractmethod

from ._base_functions import (
    slice_array, set_output_columns, assign_predictions, score_predictions,
    replace, save, load, prune_files, check_params)
from .base import OutputMixin, ProbaMixin, IndexMixin, BaseEstimator

from ..metrics import Data
from ..utils import safe_print, print_time, format_name, assert_valid_pipeline
from ..utils.exceptions import (NotFittedError, FitFailedWarning,
                                ParallelProcessingError, NotInitializedError)

from ..externals.sklearn.base import clone
from ..externals.joblib.parallel import delayed
try:
    from time import perf_counter as time
except ImportError:
    from time import time


# Types of indexers that require fits only on subsets or only on the full data
ONLY_SUB = []
ONLY_ALL = ['fullindex', 'nonetype']
GLOBAL_LEARNER_NAMES = list()
GLOBAL_TRANSFORMER_NAMES = list()


###############################################################################
class IndexedEstimator(object):
    """Indexed Estimator

    Lightweight wrapper around estimator dumps during fitting.

    """
    __slots__ = [
        '_estimator', 'name', 'index', 'in_index', 'out_index', 'data']

    def __init__(self, estimator, name, index, in_index, out_index, data):
        self._estimator = estimator
        self.name = name
        self.index = index
        self.in_index = in_index
        self.out_index = out_index
        self.data = data

    @property
    def estimator(self):
        """Deep copy of estimator"""
        return deepcopy(self._estimator)

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator

    def __getstate__(self):
        """Return pickable object"""
        return (self._estimator, self.name, self.index, self.in_index,
                self.out_index, self.data)

    def __setstate__(self, state):
        """Load tuple into instance"""
        (self._estimator, self.name, self.index, self.in_index,
         self.out_index, self.data) = state


class SubLearner(object):
    """Estimation task

    Wrapper around a sub_learner job.
    """
    def __init__(self, job, parent, estimator, in_index, out_index,
                 in_array, targets, out_array, index):
        self.job = job
        self.estimator = estimator
        self.in_index = in_index
        self.out_index = out_index
        self.in_array = in_array
        self.targets = targets
        self.out_array = out_array
        self.score_ = None
        self.index = tuple(index)

        self.path = parent._path
        self.attr = parent.attr
        self.preprocess = parent.preprocess
        self.scorer = parent.scorer
        self.raise_on_exception = parent.raise_on_exception
        self.verbose = parent.verbose

        if not parent.__no_output__:
            self.output_columns = parent.output_columns[index[0]]

        self.score_ = None
        self.fit_time_ = None
        self.pred_time_ = None

        self.name = parent.cache_name
        self.name_index = '.'.join([self.name] + [str(i) for i in index])

        if self.preprocess is not None:
            self.preprocess_index = '.'.join(
                [self.preprocess] + [str(i) for i in index])
        else:
            self.processing_index = ''

    def __call__(self):
        """Launch job"""
        return getattr(self, self.job)()

    def fit(self, path=None):
        """Fit sub-learner"""
        if path is None:
            path = self.path
        t0 = time()
        transformers = self._load_preprocess(path)

        self._fit(transformers)

        if self.out_array is not None:
            self._predict(transformers, self.scorer is not None)

        o = IndexedEstimator(estimator=self.estimator,
                             name=self.name_index,
                             index=self.index,
                             in_index=self.in_index,
                             out_index=self.out_index,
                             data=self.data)

        save(path, self.name_index, o)

        if self.verbose:
            msg = "{:<30} {}".format(self.name_index, "done")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            print_time(t0, msg, file=f)

    def predict(self, path=None):
        """Predict with sublearner"""
        if path is None:
            path = self.path
        t0 = time()
        transformers = self._load_preprocess(path)

        self._predict(transformers, False)
        if self.verbose:
            msg = "{:<30} {}".format(self.name_index, "done")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            print_time(t0, msg, file=f)

    def transform(self, path=None):
        """Predict with sublearner"""
        return self.predict(path)

    def _fit(self, transformers):
        """Sub-routine to fit sub-learner"""
        xtemp, ytemp = slice_array(self.in_array, self.targets, self.in_index)

        # Transform input (triggers copying)
        t0 = time()
        if transformers:
            xtemp, ytemp = transformers.transform(xtemp, ytemp)

        # Fit estimator
        self.estimator.fit(xtemp, ytemp)
        self.fit_time_ = time() - t0

    def _load_preprocess(self, path):
        """Load preprocessing pipeline"""
        if self.preprocess is not None:
            obj = load(path, self.preprocess_index, self.raise_on_exception)
            return obj.estimator
        return

    def _predict(self, transformers, score_preds):
        """Sub-routine to with sublearner"""
        n = self.in_array.shape[0]
        # For training, use ytemp to score predictions
        # During test time, ytemp is None
        xtemp, ytemp = slice_array(self.in_array, self.targets, self.out_index)
        t0 = time()

        if transformers:
            xtemp, ytemp = transformers.transform(xtemp, ytemp)
        predictions = getattr(self.estimator, self.attr)(xtemp)

        self.pred_time_ = time() - t0

        # Assign predictions to matrix
        assign_predictions(self.out_array, predictions,
                           self.out_index, self.output_columns, n)

        # Score predictions if applicable
        if score_preds:
            self.score_ = score_predictions(
                ytemp, predictions, self.scorer, self.name_index, self.name)

    @property
    def data(self):
        """fit data"""
        out = {'score': self.score_,
               'ft': self.fit_time_,
               'pt': self.pred_time_}
        return out


class SubTransformer(object):

    """Sub-routine for fitting a pipeline
    """

    def __init__(self, job, parent, estimator, in_index, in_array,
                 targets, index, out_index=None, out_array=None):
        self.job = job
        self.estimator = estimator
        self.in_index = in_index
        self.out_index = out_index
        self.in_array = in_array
        self.out_array = out_array
        self.targets = targets
        self.index = index

        self.transform_time_ = None

        self.path = parent._path
        self.verbose = parent.verbose
        self.name = parent.cache_name
        self.name_index = '.'.join(
            [self.name] + [str(i) for i in index])

        if not parent.__no_output__:
            self.output_columns = parent.output_columns[index[0]]

    def __call__(self):
        """Launch job"""
        return getattr(self, self.job)()

    def predict(self):
        """Dump transformers for prediction"""
        self._transform()

    def transform(self):
        """Dump transformers for prediction"""
        self._transform()

    def _transform(self):
        """Run a transformation"""
        t0 = time()
        n = self.in_array.shape[0]
        xtemp, ytemp = slice_array(
            self.in_array, self.targets, self.out_index)

        xtemp, ytemp = self.estimator.transform(xtemp, ytemp)

        assign_predictions(
            self.out_array, xtemp, self.out_index, self.output_columns, n)

        if self.verbose:
            msg = "{:<30} {}".format(self.name_index, "done")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            print_time(t0, msg, file=f)

    def fit(self, path=None):
        """Fit transformers"""
        path = path if path else self.path
        t0 = time()
        xtemp, ytemp = slice_array(
            self.in_array, self.targets, self.in_index)

        t0_f = time()
        self.estimator.fit(xtemp, ytemp)
        self.transform_time_ = time() - t0_f

        if self.out_array is not None:
            self._transform()

        o = IndexedEstimator(estimator=self.estimator,
                             name=self.name_index,
                             index=self.index,
                             in_index=self.in_index,
                             out_index=self.out_index,
                             data=self.data)
        save(path, self.name_index, o)
        if self.verbose:
            f = "stdout" if self.verbose < 10 else "stderr"
            msg = "{:<30} {}".format(self.name_index, "done")
            print_time(t0, msg, file=f)

    @property
    def data(self):
        """fit data"""
        return {'ft': self.transform_time_}


class EvalSubLearner(SubLearner):

    """EvalSubLearner

    sub-routine for cross-validated evaluation.
    """
    def __init__(self, job, parent, estimator, in_index, out_index,
                 in_array, targets, index):

        super(EvalSubLearner, self).__init__(
            job=job, parent=parent, estimator=estimator,
            in_index=in_index, out_index=out_index,
            in_array=in_array, out_array=None,
            targets=targets, index=index)
        self.error_score = parent.error_score
        self.train_score_ = None
        self.test_score_ = None
        self.train_pred_time_ = None
        self.test_pred_time_ = None

    def fit(self, path=None):
        """Evaluate sub-learner"""
        path = path if path else self.path
        if self.scorer is None:
            raise ValueError("Cannot generate CV-scores without a scorer")
        t0 = time()
        transformers = self._load_preprocess(path)
        self._fit(transformers)
        self._predict(transformers)

        o = IndexedEstimator(estimator=self.estimator,
                             name=self.name_index,
                             index=self.index,
                             in_index=self.in_index,
                             out_index=self.out_index,
                             data=self.data)
        save(path, self.name_index, o)

        if self.verbose:
            f = "stdout" if self.verbose else "stderr"
            msg = "{:<30} {}".format(self.name_index, "done")
            print_time(t0, msg, file=f)

    def _predict(self, transformers, score_preds=None):
        """Sub-routine to with sublearner"""
        # Train set
        self.train_score_, self.train_pred_time_ = self._score_preds(
            transformers, self.in_index)

        # Validation set
        self.test_score_, self.test_pred_time_ = self._score_preds(
            transformers, self.out_index)

    def _score_preds(self, transformers, index):
        # Train scores
        xtemp, ytemp = slice_array(self.in_array, self.targets, index)
        if transformers:
            xtemp, ytemp = transformers.transform(xtemp, ytemp)

        t0 = time()

        if self.error_score is not None:
            try:
                scores = self.scorer(self.estimator, xtemp, ytemp)
            except Exception as exc:  # pylint: disable=broad-except
                warnings.warn(
                    "Scoring failed. Setting error score %r."
                    "Details:\n%r" % (self.error_score, exc),
                    FitFailedWarning)
                scores = self.error_score
        else:
            scores = self.scorer(self.estimator, xtemp, ytemp)
        pred_time = time() - t0

        return scores, pred_time

    @property
    def data(self):
        """Score data"""
        out = {'test_score': self.test_score_,
               'train_score': self.train_score_,
               'fit_time': self.fit_time_,
               'pred_time': self.train_pred_time_,
               # 'test_pred_time': self.train_pred_time_,
               }
        return out


class Cache(object):

    """Cache wrapper for IndexedEstimator
    """

    def __init__(self, obj, path, verbose):
        self.obj = obj
        self.path = path
        self.name = obj.name
        self.verbose = verbose

    def __call__(self, path=None):
        """Cache estimator to path"""
        path = path if path else self.path
        save(path, self.name, self.obj)
        if self.verbose:
            msg = "{:<30} {}".format(self.name, "cached")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            safe_print(msg, file=f)


###############################################################################
class BaseNode(OutputMixin, IndexMixin, BaseEstimator):

    """Base computational node inherited by job generators.

    Common API for job generators. A class that inherits the base
    need to set a ``__subtype__`` in the constructor. The sub-type should be
    the class that runs estimations and must implement a ``__call__``,
    ``fit``, ``transform`` and ``predict`` method.
    """

    __meta_class__ = ABCMeta

    # Reset subtype class attribute in any class that inherits the base
    __subtype__ = None

    def __init__(self, name, estimator, indexer=None, verbose=False, **kwargs):
        super(BaseNode, self).__init__(name, **kwargs)

        # Variables
        self._path = None
        self._data_ = None
        self._times_ = None
        self._learner_ = None
        self._sublearners_ = None
        self.__collect__ = False
        self._partitions = None
        self.__only_all__ = None
        self.__only_sub__ = None

        # Parameters
        self.indexer = indexer
        if self.indexer:
            self.set_indexer(self.indexer)

        self.estimator = estimator
        self.verbose = verbose
        self.cache_name = None
        self.output_columns = None
        self.feature_span = None

        self.__static__.extend(['estimator', 'name', 'indexer'])

    def __iter__(self):
        yield self

    def __call__(self, args, arg_type='main', parallel=None):
        """Caller for producing jobs"""
        job = args['job']
        self._path = args['dir']
        _threading = self.backend == 'threading'

        if not self.__indexer__:
            raise NotInitializedError(
                "Instance has no indexer attached. Call set_indexer first.")

        if job != 'fit' and not self.__fitted__:
            raise NotFittedError(
                "Instance not fitted with current params. Call 'fit' first.")

        if job == 'fit':
            if self.__fitted__ and args.pop('refit', False):
                # Check refit
                if self.__no_output__:
                    return
                args['job'] = 'transform'
                return self(args, arg_type, parallel)

            # Record static params
            self._store_static_params()

        generator = getattr(self, 'gen_%s' % job)(**args[arg_type])

        if not parallel:
            return generator

        parallel(delayed(subtask, not _threading)()
                 for subtask in generator)

        if self.__collect__:
            self.collect()

    def _gen_pred(self, job, X, P, generator):
        """Generator for predicting with fitted learner

        Parameters
        ----------
        job: str
            type of job

        X : array-like of shape [n_samples, n_features]
            input array

        P : array-like of shape [n_samples, n_prediction_features]
            output array to populate. Must be writeable.

        generator : iterable
            iterator of learners of sub-learners to predict with.
            One of ``self.learner_`` and ``self.sublearners_``.
        """
        for estimator in generator:
            yield self.__subtype__(
                job=job,
                parent=self,
                estimator=estimator.estimator,
                in_index=estimator.in_index,
                out_index=estimator.out_index,
                in_array=X,
                out_array=P,
                index=estimator.index,
                targets=None,
                )

    def gen_fit(self, X, y, P=None):
        """Routine for generating fit jobs conditional on refit

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            input array

        y: array-like of shape [n_samples,]
            targets

        P: array-like of shape [n_samples, n_prediction_features], optional
            output array to populate. Must be writeable. Only pass if
            predictions are desired.
        """
        # We use a derived cache_name during estimation: if the name of the
        # instance or the name of the preprocessing dependency changes, this
        # allows us to pick up on that.
        if hasattr(self, 'preprocess'):
            self.cache_name = '%s.%s' % (
                self.preprocess, self.name) if self.preprocess else self.name
        else:
            self.cache_name = self.name

        if self.__subtype__ is None:
            raise ParallelProcessingError(
                "Class incorrectly constructed. Need to set class attribute "
                "__subtype__")

        self.__collect__ = True

        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        i = 0
        if not self.__only_sub__:
            out = P if self.__only_all__ else None
            for partition_index in self.indexer.partition():
                yield self.__subtype__(
                    job='fit',
                    parent=self,
                    estimator=self.cloned_estimator,
                    in_index=partition_index,
                    out_index=None,
                    in_array=X,
                    targets=y,
                    out_array=out,
                    index=(i, 0),
                )
                i += 1

        if not self.__only_all__:
            # Fit sub-learners on cv folds
            for i, (train_index, test_index) in enumerate(
                    self.indexer.generate()):
                # Note that we bump index[1] by 1 to have index[1] start at 1
                if self._partitions == 1:
                    index = (0, i + 1)
                else:
                    splits = self.indexer.folds
                    index = (i // splits, i % splits + 1)

                yield self.__subtype__(
                    job='fit',
                    parent=self,
                    estimator=self.cloned_estimator,
                    in_index=train_index,
                    out_index=test_index,
                    in_array=X,
                    targets=y,
                    out_array=P,
                    index=index,
                )

    def gen_transform(self, X, P=None):
        """Generate cross-validated predict jobs

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            input array

        y: array-like of shape [n_samples,]
            targets

        P: array-like of shape [n_samples, n_prediction_features], optional
            output array to populate. Must be writeable. Only pass if
            predictions are desired.
        """
        return self._gen_pred('transform', X, P, self.sublearners)

    def gen_predict(self, X, P=None):
        """Generate predicting jobs

        Parameters
        ----------
        X: array-like of shape [n_samples, n_features]
            input array

        y: array-like of shape [n_samples,]
            targets

        P: array-like of shape [n_samples, n_prediction_features], optional
            output array to populate. Must be writeable. Only pass if
            predictions are desired.
        """
        return self._gen_pred('predict', X, P, self.learner)

    def collect(self, path=None):
        """Load fitted estimator from cache

        Parameters
        ----------
        path: str, list, optional
            path to cache.
        """
        if path is None:
            path = self._path
        if self.__collect__:
            (learner_files,
             learner_data,
             sublearner_files,
             sublearner_data) = self._collect(path)

            self.clear()
            self._learner_ = learner_files
            self._sublearners_ = sublearner_files
            self._data_ = sublearner_data
            self._times_ = learner_data

            # Collection complete, turn off
            self.__collect__ = False

    def clear(self):
        """Clear load"""
        self._sublearners_ = None
        self._learner_ = None
        self._data_ = None
        self._times_ = None
        self._path = None

    def set_indexer(self, indexer):
        """Set indexer and auxiliary attributes

        Parameters
        ----------
        indexer: obj
            indexer to build instance with.
        """
        self.indexer = indexer
        self._partitions = indexer.partitions
        self.__only_all__ = indexer.__class__.__name__.lower() in ONLY_ALL
        self.__only_sub__ = indexer.__class__.__name__.lower() in ONLY_SUB

    def _collect(self, path):
        """Collect files from cache"""
        files = prune_files(path, self.cache_name)
        learner_files = list()
        learner_data = list()
        sublearner_files = list()
        sublearner_data = list()
        while files:
            f = files.pop(0)
            if f in files:
                raise ParallelProcessingError(
                    "Corrupt cache: duplicate cache entry found.\n%r" % f)

            if f.index[1] == 0:
                learner_files.append(f)
                learner_data.append((f.name, f.data))
            else:
                sublearner_files.append(f)
                sublearner_data.append((f.name, f.data))

        if self.__only_sub__:
            # Full learners are the same as the sub-learners
            learner_files, learner_data = replace(sublearner_files)
        if self.__only_all__:
            # Sub learners are the same as the sub-learners
            sublearner_files, sublearner_data = replace(learner_files)

        return learner_files, learner_data, sublearner_files, sublearner_data

    def _return_attr(self, attr):
        if not self.__fitted__:
            raise NotFittedError("Instance not fitted.")
        return getattr(self, attr)

    def set_output_columns(self, X=None, y=None, job=None, n_left_concats=0):
        """Set the output_columns attribute"""
        # pylint: disable=unused-argument
        multiplier = self._get_multiplier(X, y)
        target = self._partitions * multiplier + n_left_concats
        set_output_columns(
            [self], self._partitions, multiplier, n_left_concats, target)

        mi = n_left_concats
        mx = max([i for i in self.output_columns.values()]) + multiplier
        self.feature_span = (mi, mx)

    @abstractmethod
    def _get_multiplier(self, X, y):
        """Get the prediction multiplier given input (X, y)"""
        return 1

    @property
    def __fitted__(self):
        """Fit status"""
        if (not self._learner_ or not self._sublearners_ or
                not self.indexer.__fitted__):
            return False

        # Check estimator param overlap
        fitted = self._learner_ + self._sublearners_
        fitted_params = fitted[0].estimator.get_params(deep=True)
        model_estimator_params = self.estimator.get_params(deep=True)

        # NOTE: Currently we just issue a warning if params don't overlap
        check_params(fitted_params, model_estimator_params)
        self._check_static_params()

        # NOTE: This check would trigger a FitFailedError if param_check fails
        # check_params(fitted_params, model_estimator_params):
        #    self.clear()  # Release obsolete estimators
        #    return False

        # Check that hyper-params hasn't changed
        # if not self._check_static_params():
        #     return False
        # return True
        return True

    @property
    def cloned_estimator(self):
        """Copy of estimator"""
        return clone(self.estimator)

    @property
    def learner(self):
        """Generator for learner fitted on full data"""
        # pylint: disable=not-an-iterable
        out = self._return_attr('_learner_')
        for estimator in out:
            yield deepcopy(estimator)

    @property
    def sublearners(self):
        """Generator for learner fitted on folds"""
        # pylint: disable=not-an-iterable
        out = self._return_attr('_sublearners_')
        for estimator in out:
            yield deepcopy(estimator)

    @property
    def raw_data(self):
        """List of data collected from each sub-learner during fitting."""
        return self._return_attr('_data_')

    @property
    def data(self):
        """Dictionary with aggregated data from fitting sub-learners."""
        out = self._return_attr('_data_')
        return Data(out)

    @property
    def times(self):
        """Fit and predict times for the final learners"""
        out = self._return_attr('_times_')
        return Data(out)


class Learner(ProbaMixin, BaseNode):

    """Learner

    Wrapper for base learners.

    Parameters
    __________
    estimator : obj
        estimator to construct learner from

    preprocess : str, obj
        preprocess transformer. Pass either the string
        cache reference or the transformer instance. If the latter,
        the :attr:`preprocess` will refer to the transformer name.

    name : str
        name of learner. If ``preprocess`` is not ``None``,
        the name will be prepended to ``preprocess__name``.

    attr : str (default='predict')
        predict attribute, typically one of 'predict' and 'predict_proba'

    scorer : func
        function to use for scoring predictions during cross-validated
        fitting.

    output_columns : dict, optional
        mapping of prediction feature columns from learner to columns in
        output array. Normally, this map is ``{0: x}``, but if the ``indexer``
        creates partitions, each partition needs to be mapped:
        ``{0: x, 1: x + 1}``. Note that if ``output_columns`` are not given at
        initialization, the ``set_output_columns`` method must be called before
        running estimations.

    verbose : bool, int (default = False)
        whether to report completed fits.

    **kwargs : bool (default=True)
        Optional ParallelProcessing arguments. See :class:`BaseParallel`.
    """

    __subtype__ = SubLearner

    def __init__(self, estimator, indexer=None, name=None, preprocess=None,
                 attr=None, scorer=None, proba=False, **kwargs):
        super(Learner, self).__init__(
            name=format_name(name, 'learner', GLOBAL_LEARNER_NAMES),
            estimator=estimator, indexer=indexer, **kwargs)

        self._classes = None
        self.proba = proba
        self._scorer = scorer
        self.preprocess = preprocess
        self.n_pred = self._partitions
        self.attr = attr if attr else self._predict_attr

        # Protect preprocess against later changes
        self.__static__.append('preprocess')

    @property
    def scorer(self):
        """Copy of scorer"""
        return deepcopy(self._scorer)

    @scorer.setter
    def scorer(self, scorer):
        """Copy of scorer"""
        self._scorer = scorer


class Transformer(BaseNode):

    """Preprocessing handler.

    Wrapper for transformation pipeline.

    Parameters
    __________
    indexer : obj, None
        indexer to use for generating fits.
        Set to ``None`` to fit only on all data.

    estimator : obj
        transformation pipeline to construct learner from

    name : str
        name of learner. If ``preprocess`` is not ``None``,
        the name will be prepended to ``preprocess__name``.

    output_columns : dict, optional
        If transformer is to be used to output data, need to
        set ``output_columns``. Normally, this map is
        ``{0: x}``, but if the ``indexer``
        creates partitions, each partition needs to be mapped:
        ``{0: x, 1: x + 1}``.

    verbose : bool, int (default = False)
        whether to report completed fits.

    raise_on_exception : bool (default=True)
        whether to warn on non-fatal exceptions or raise an error.
    """

    __subtype__ = SubTransformer

    def __init__(self, estimator, indexer=None, name=None, **kwargs):
        assert_valid_pipeline(estimator)
        name = format_name(name, 'transformer', GLOBAL_TRANSFORMER_NAMES)
        super(Transformer, self).__init__(
            name=name, estimator=estimator, indexer=indexer, **kwargs)
        self.__no_output__ = True

    def _get_multiplier(self, X, y=None, alt=None):
        """Number of cols produced in prediction"""
        return X.shape[1]

    def _gen_pred(self, job, X, P, generator):
        if self.__no_output__:
            def gen():
                for o in generator:
                    yield Cache(o, self._path, self.verbose)

            return gen()
        else:
            return super(Transformer, self)._gen_pred(job, X, P, generator)


class EvalTransformer(Transformer):

    r"""Evaluator version of the Transformer.

    Derived class from Transformer adapted to cross\-validated grid-search.
    See :class:`Transformer` for more details.
    """

    def __init__(self, estimator, indexer=None, name=None, **kwargs):
        super(EvalTransformer, self).__init__(
            estimator, indexer=indexer, name=name, **kwargs)
        self.output_columns = {0: 0}  # For compatibility with SubTransformer
        self.__only_all__ = False
        self.__only_sub__ = True


class EvalLearner(Learner):

    """EvalLearner

    EvalLearner is a derived class from Learner used for cross-validated
    scoring of an estimator.

    Parameters
    __________
    estimator : obj
        estimator to construct learner from

    preprocess : str
        preprocess cache refernce

    indexer : obj, None
        indexer to use for generating fits.
        Set to ``None`` to fit only on all data.

    name : str
        name of learner. If ``preprocess`` is not ``None``,
        the name will be prepended to ``preprocess__name``.

    attr : str (default='predict')
        predict attribute, typically one of 'predict' and 'predict_proba'

    scorer : func
        function to use for scoring predictions during cross-validated
        fitting.

    error_score : int, float, None (default = None)
        score to set if cross-validation fails. Set to ``None`` to raise error.

    verbose : bool, int (default = False)
        whether to report completed fits.

    raise_on_exception : bool (default=True)
        whether to warn on non-fatal exceptions or raise an error.
    """

    __subtype__ = EvalSubLearner

    def __init__(self, estimator, preprocess, name, attr, scorer,
                 error_score=None, verbose=False, **kwargs):
        super(EvalLearner, self).__init__(
            estimator=estimator, preprocess=preprocess,
            name=name, attr=attr, scorer=scorer, verbose=verbose, **kwargs)

        self.__only_sub__ = True
        self.__only_all__ = False
        self.output_columns = {0: 0}     # For compatibility with SubLearner
        self.error_score = error_score

    def gen_fit(self, X, y, P=None, refit=True):
        """Generator for fitting learner on given data"""
        self.cache_name = '%s.%s' % (
            self.preprocess, self.name) if self.preprocess else self.name

        if not refit and self.__fitted__:
            self.gen_transform(X, P)

        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        if self.indexer is None:
            raise ValueError("Cannot run cross-validation without an indexer")

        self.__collect__ = True
        for i, (train_index, test_index) in enumerate(
                self.indexer.generate()):
            # Note that we bump index[1] by 1 to have index[1] start at 1
            if self._partitions == 1:
                index = (0, i + 1)
            else:
                index = (0, i % self._partitions + 1)
            yield EvalSubLearner(
                job='fit',
                parent=self,
                estimator=self.cloned_estimator,
                in_index=train_index,
                out_index=test_index,
                in_array=X,
                targets=y,
                index=index,
            )
