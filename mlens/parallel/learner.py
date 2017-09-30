"""

ML-Ensemble

:author: Sebastian Flennerhag
:license: MIT

Base learner classes
"""
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from __future__ import print_function, division

import os
from copy import deepcopy
import warnings
from abc import ABCMeta, abstractmethod
from ._base_functions import (slice_array,
                              transform,
                              assign_predictions,
                              score_predictions)
from ..metrics import Data
from ..utils import (pickle_save,
                     pickle_load,
                     load,
                     safe_print,
                     print_time)
from ..utils.exceptions import NotFittedError, FitFailedWarning
from ..externals.sklearn.base import clone, BaseEstimator
try:
    from time import perf_counter as time
except ImportError:
    from time import time


# Types of indexers that require fits only on subsets or only on the full data
ONLY_SUB = []
ONLY_ALL = ['fullindex', 'nonetype']


def _replace(source_files):
    """Utility function to replace empty files list"""
    replace_files = [deepcopy(o) for o in source_files]
    for o in replace_files:
        o.name = o.name[:-1] + '0'
        o.index = (o.index[0], 0)
        o.out_index = None
        o.in_index = None

    # Set a vacuous data list
    replace_data = [(o.name, None) for o in replace_files]
    return replace_files, replace_data


class IndexedEstimator(object):
    """Indexed Estimator

    Lightweight wrapper around estimator dumps during fitting.

    """
    __slots__ = ['_estimator', 'name', 'index',
                 'in_index', 'out_index', 'data']

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
        return (self._estimator,
                self.name,
                self.index,
                self.in_index,
                self.out_index,
                self.data)

    def __setstate__(self, state):
        """Load tuple into instance"""
        (self._estimator,
         self.name,
         self.index,
         self.in_index,
         self.out_index,
         self.data) = state


class _BaseEstimator(BaseEstimator):

    """Base estimator class

    Adapts Scikit-learn's base estimator class.
    """

    __meta_class__ = ABCMeta

    def __init__(self, indexer, name):
        self.name = name
        self.__fitted__ = False
        self._data_ = None
        self._times_ = None
        self._learner_ = None
        self._sublearners_ = None

        self._set_indexer(indexer)

    def __call__(self, job, *args, **kwargs):
        """Caller for producing jobs"""
        job = 'gen_%s' % job
        if not hasattr(self, job):
            raise AttributeError("Job [%s] not accepted." % job)
        return getattr(self, job)(*args, **kwargs)

    @abstractmethod
    def _gen_pred(self, X, P, generator):
        """Method for generating predict or transform jobs"""
        return

    @abstractmethod
    def gen_fit(self, X, y, P=None):
        """Method for generating fit jobs"""
        return

    def gen_transform(self, X=None, P=None):
        """Generator for regenerating training predictions"""
        return self._gen_pred(X, P, self.sublearners_)

    def gen_predict(self, X=None, P=None):
        """Generator for predicting test set"""
        return self._gen_pred(X, P, self.learner_)

    def collect(self, path):
        """Load fitted estimator from cache"""
        (learner_files,
         learner_data,
         sublearner_files,
         sublearner_data) = self._collect(path)

        self._learner_ = learner_files
        self._sublearners_ = sublearner_files
        self._data_ = sublearner_data
        self._times_ = learner_data
        self.__fitted__ = True

    def clear(self):
        """Clear load"""
        self._sublearners_ = None
        self._learner_ = None
        self._data_ = None
        self.__fitted__ = False

    def _set_indexer(self, indexer):
        """Set indexer and auxiliary attributes"""
        self._indexer = indexer
        self._partitions = getattr(indexer, 'n_partitions', 1)
        self.__only_all__ = indexer.__class__.__name__.lower() in ONLY_ALL
        self.__only_sub__ = indexer.__class__.__name__.lower() in ONLY_SUB

    def _collect(self, path):
        """Collect files from cache"""
        self.clear()

        files = [os.path.join(path, f)
                 for f in os.listdir(path)
                 if f.startswith(self.name)]

        learner_files = list()
        learner_data = list()
        sublearner_files = list()
        sublearner_data = list()
        for f in sorted(files):
            o = pickle_load(f)
            if o.index[1] == 0:
                learner_files.append(o)
                learner_data.append((o.name, o.data))
            else:
                sublearner_files.append(o)
                sublearner_data.append((o.name, o.data))

        if self.__only_sub__:
            # Full learners are the same as the sub-learners
            learner_files, learner_data = _replace(sublearner_files)
        if self.__only_all__:
            # Sub learners are the same as the sub-learners
            sublearner_files, sublearner_data = _replace(learner_files)

        self.__fitted__ = True

        return learner_files, learner_data, sublearner_files, sublearner_data

    def _return_attr(self, attr):
        if not self.__fitted__:
            raise NotFittedError("Instance not fitted.")
        return getattr(self, attr)

    def get_params(self, deep=True):
        """Get learner parameters

        Parameters
        ----------
        deep : bool (default = True)
            whether to return nested parameters
        """
        out = dict()
        for par_name in self._get_param_names():
            par = getattr(self, '_%s' % par_name, None)
            if par is None:
                getattr(self, par_name, None)
            if deep and hasattr(par, 'get_params'):
                for key, value in par.get_params(deep=True).items():
                    out['%s__%s' % (par_name, key)] = value
            out[par_name] = par
        return out

    @property
    def learner_(self):
        """Generator for learner fitted on full data"""
        # pylint: disable=not-an-iterable
        out = self._return_attr('_learner_')
        for estimator in out:
            yield deepcopy(estimator)

    @property
    def sublearners_(self):
        """Generator for learner fitted on folds"""
        # pylint: disable=not-an-iterable
        out = self._return_attr('_sublearners_')
        for estimator in out:
            yield deepcopy(estimator)

    @property
    def raw_data(self):
        """List of data collected from each sub-learner during fiting."""
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

    @property
    def indexer(self):
        """Copy of indexer"""
        return deepcopy(self._indexer)

    @indexer.setter
    def indexer(self, indexer):
        """Update indexer"""
        self._set_indexer(indexer)


class Learner(_BaseEstimator):

    """Learner

    Wrapper for base learners.

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

    output_columns : dict
        mapping of prediction feature columns from learner to columns in
        output array. Normally, this map is ``{0: x}``, but if the ``indexer``
        creates partitions, each partition needs to be mapped:
        ``{0: x, 1: x + 1}``.

    verbose : bool, int (default = False)
        whether to report completed fits.

    raise_on_exception : bool (default=True)
        whether to warn on non-fatal exceptions or raise an error.
    """

    def __init__(self,
                 estimator,
                 preprocess,
                 indexer,
                 name,
                 attr,
                 scorer,
                 output_columns,
                 verbose=False,
                 raise_on_exception=True):
        super(Learner, self).__init__(indexer, name)
        self.preprocess = preprocess
        self.output_columns = output_columns
        self.attr = attr
        self.raise_on_exception = raise_on_exception
        self.verbose = verbose

        # We protect these to avoid corrupting a fitted learner
        self._estimator = estimator
        self._scorer = scorer

        if preprocess:
            self.name = '%s__%s' % (preprocess, name)

        self._sublearners_ = None        # Fitted sub-learners
        self._learner_ = None            # Fitted learners

    def _gen_pred(self, X, P, generator):
        """Generator for predicting with fitted learner

        Parameters
        ----------
        X : array
            input array

        P : array
            output array to populate. Must be writeable.

        generator : iterable
            iterator of learners of sub-learners to predict with.
            One of ``self.learner_`` and ``self.sublearners_``.
        """
        for estimator in generator:
            yield SubLearner(learner=self,
                             estimator=estimator.estimator,
                             in_index=estimator.in_index,
                             out_index=estimator.out_index,
                             in_array=X,
                             targets=None,
                             out_array=P,
                             index=estimator.index)

    def gen_fit(self, X, y, P=None):
        """Generator for fitting learner on given data

         Parameters
        ----------
        X : array
            input array

        y : array
            training targets

        P : array
            output array to populate. Must be writeable.
        """
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if not self.__only_sub__:
            out = P if self.__only_all__ else None
            for partition_index in self.indexer.partition():
                # This yields None, None for default indexers
                yield SubLearner(learner=self,
                                 estimator=self.estimator,
                                 in_index=partition_index,
                                 out_index=None,
                                 in_array=X,
                                 targets=y,
                                 out_array=out,
                                 index=index)
                index[0] = index[0] + 1

        if not self.__only_all__:
            # Fit sub-learners on cv folds
            for i, (train_index, test_index) in enumerate(
                    self.indexer.generate()):
                # Note that we bump index[1] by 1 to have index[1] start at 1
                if self._partitions == 1:
                    index = (0, i + 1)
                else:
                    splits = self.indexer.n_splits
                    index = (i // splits, i % splits + 1)

                yield SubLearner(learner=self,
                                 estimator=self.estimator,
                                 in_index=train_index,
                                 out_index=test_index,
                                 in_array=X,
                                 targets=y,
                                 out_array=P,
                                 index=index)

    @property
    def estimator(self):
        """Copy of estimator"""
        return clone(self._estimator)

    @estimator.setter
    def estimator(self, estimator):
        """Replace blueprint estimator"""
        self._estimator = estimator

    @property
    def scorer(self):
        """Copy of scorer"""
        return deepcopy(self._scorer)

    @scorer.setter
    def scorer(self, scorer):
        """Replace blueprint scorer"""
        self._scorer = scorer


class SubLearner(object):
    """Estimation task

    Wrapper around a sub_learner job.
    """
    def __init__(self,
                 learner,
                 estimator,
                 in_index,
                 out_index,
                 in_array,
                 targets,
                 out_array,
                 index):
        self.estimator = estimator
        self.in_index = in_index
        self.out_index = out_index
        self.in_array = in_array
        self.targets = targets
        self.out_array = out_array
        self.score_ = None
        self.index = tuple(index)

        self.attr = learner.attr
        self.preprocess = learner.preprocess
        self.scorer = learner.scorer
        self.raise_on_exception = learner.raise_on_exception
        self.verbose = learner.verbose
        self.output_columns = learner.output_columns[index[0]]

        self.score_ = None
        self.fit_time_ = None
        self.pred_time_ = None

        self.name = learner.name
        self.name_index = '__'.join(
            [self.name] + [str(i) for i in index])

        if self.preprocess is not None:
            self.preprocess_index = '__'.join(
                [self.preprocess] + [str(i) for i in index])
        else:
            self.processing_index = ''

    def __call__(self, job, path):
        """Launch job"""
        if not hasattr(self, job):
            raise NotImplementedError(
                "SubLearner does not implement [%s]" % job)
        return getattr(self, job)(path)

    def fit(self, path):
        """Fit sub-learner"""
        t0 = time()
        transformers = self._load_preprocess(path)

        self._fit(transformers)

        if self.out_array is not None:
            self._predict(transformers, self.scorer is not None)

        f = os.path.join(path, self.name_index)
        o = IndexedEstimator(estimator=self.estimator,
                             name=self.name_index,
                             index=self.index,
                             in_index=self.in_index,
                             out_index=self.out_index,
                             data=self.data)
        pickle_save(o, f)

        if self.verbose:
            msg = "{:<30} {}".format(self.name_index, "done")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            print_time(t0, msg, file=f)

    def predict(self, path):
        """Predict with sublearner"""
        t0 = time()
        transformers = self._load_preprocess(path)

        self._predict(transformers, False)
        if self.verbose:
            msg = "{:<30} {}".format(self.name_index, "done")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            print_time(t0, msg, file=f)

    def transform(self, path):
        """Predict with sublearner"""
        return self.predict(path)

    def _fit(self, transformers):
        """Sub-routine to fit sub-learner"""
        xtemp, ytemp = slice_array(self.in_array,
                                   self.targets,
                                   self.in_index)

        # Transform input (triggers copying)
        t0 = time()
        for _, tr in transformers:
            xtemp, ytemp = transform(tr, xtemp, ytemp)

        # Fit estimator
        self.estimator.fit(xtemp, ytemp)
        self.fit_time_ = time() - t0

    def _load_preprocess(self, path):
        """Load preprocessing pipeline"""
        if self.preprocess is not None:
            f = os.path.join(path, self.preprocess_index)
            obj = load(f, self.raise_on_exception)
            tr_list = obj.estimator
        else:
            tr_list = list()
        return tr_list

    def _predict(self, transformers, score_preds):
        """Sub-routine to with sublearner"""
        n = self.in_array.shape[0]
        # For training, use ytemp to score predictions
        # During test time, ytemp is None
        xtemp, ytemp = slice_array(self.in_array,
                                   self.targets,
                                   self.out_index)
        t0 = time()
        for _, tr in transformers:
            xtemp = tr.transform(xtemp)

        predictions = getattr(self.estimator, self.attr)(xtemp)
        self.pred_time_ = time() - t0

        # Assign predictions to matrix
        assign_predictions(self.out_array,
                           predictions,
                           self.out_index,
                           self.output_columns,
                           n)

        # Score predictions if applicable
        if score_preds:
            self.score_ = score_predictions(
                ytemp, predictions, self.scorer,
                self.name_index, self.name)

    @property
    def data(self):
        """fit data"""
        out = {'score': self.score_,
               'ft': self.fit_time_,
               'pt': self.pred_time_,
               }
        return out


class Transformer(_BaseEstimator):

    """Preprocessing handler.

    Wrapper for transformation pipeline.

    Parameters
    __________
    pipeline : obj
        transformation pipeline to construct learner from

    indexer : obj, None
        indexer to use for generating fits.
        Set to ``None`` to fit only on all data.

    name : str
        name of learner. If ``preprocess`` is not ``None``,
        the name will be prepended to ``preprocess__name``.

    verbose : bool, int (default = False)
        whether to report completed fits.

    raise_on_exception : bool (default=True)
        whether to warn on non-fatal exceptions or raise an error.
    """

    def __init__(self,
                 pipeline,
                 indexer,
                 name,
                 verbose=False,
                 raise_on_exception=True):
        super(Transformer, self).__init__(indexer, name)
        self._pipeline = pipeline
        self.verbose = verbose
        self.raise_on_exception = raise_on_exception

    @property
    def pipeline(self):
        """Blueprint pipeline"""
        return [(tr_name, clone(tr))
                for tr_name, tr in self._pipeline]

    @pipeline.setter
    def pipeline(self, pipeline):
        """Update pipeline blueprint"""
        self._pipeline = pipeline

    def _gen_pred(self, X, P, generator):
        """Generator for Cache object"""
        for obj in generator:
            yield Cache(obj, self.verbose)

    def gen_fit(self, X, y, P=None):
        """Generator for fitting pipeline on given data"""
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if not self.__only_sub__:
            # Let index[1] == 0 for all full fits
            # Hence fold_index = 0 -> final base learner
            for partition_index in self.indexer.partition():
                yield SubTransformer(transformer=self,
                                     pipeline=self.pipeline,
                                     in_index=partition_index,
                                     in_array=X,
                                     targets=y,
                                     index=index)
                index[0] = index[0] + 1

        if not self.__only_all__:
            # Fit sub-learners on cv folds
            for i, (train_index, _) in enumerate(
                    self.indexer.generate()):
                # Note that we bump index[1] by 1 to have index[1] start at 1
                if self._partitions == 1:
                    index = (0, i + 1)
                else:
                    splits = self.indexer.n_splits
                    index = (i // splits, i % splits + 1)
                yield SubTransformer(transformer=self,
                                     pipeline=self.pipeline,
                                     in_index=train_index,
                                     in_array=X,
                                     targets=y,
                                     index=index)


class Cache(object):

    """Cache wrapper for IndexedEstimator
    """

    def __init__(self, obj, verbose):
        self.obj = obj
        self.name = obj.name
        self.verbose = verbose

    def __call__(self, job, path):
        """Cache estimator to path"""
        if not hasattr(self, job):
            raise NotImplementedError(
                "Cache does not implement [%s]' % job")
        return getattr(self, job)(path)

    def predict(self, path):
        """Cache transformer for predict"""
        f = os.path.join(path, self.name)
        pickle_save(self.obj, f)
        if self.verbose:
            msg = "{:<30} {}".format(self.name, "cached")
            f = "stdout" if self.verbose < 10 - 3 else "stderr"
            safe_print(msg, file=f)

    def transform(self, path):
        """Cache transformer for transform"""
        self.predict(path)


class SubTransformer(object):

    """Sub-routine for fitting a pipeline
    """

    def __init__(self,
                 transformer,
                 pipeline,
                 in_index,
                 in_array,
                 targets,
                 index):
        self.pipeline = pipeline
        self.in_index = in_index
        self.in_array = in_array
        self.targets = targets
        self.index = index

        self.transform_time_ = None

        self.verbose = transformer.verbose
        self.name = transformer.name
        self.name_index = '__'.join(
            [self.name] + [str(i) for i in index])

    def __call__(self, job, path):
        """Launch job"""
        if not hasattr(self, job):
            raise NotImplementedError(
                "SubTransformer does not implement [%s]' % job")
        return getattr(self, job)(path)

    @staticmethod
    def predict(path=None):
        """Dump transformers for prediction"""
        # pylint: disable=unused-argument
        raise ValueError("SubTransformer only implements a fit method. "
                         "Use the Cache class for predict and transform.")

    def transform(self, path=None):
        """Dump transformers for prediction"""
        self.predict(path)

    def fit(self, path):
        """Fit transformers"""
        t0 = time()
        n = len(self.pipeline)
        xtemp, ytemp = slice_array(self.in_array,
                                   self.targets,
                                   self.in_index)

        t0_f = time()
        fitted_transformers = list()
        for tr_name, tr in self.pipeline:
            tr.fit(xtemp, ytemp)
            fitted_transformers.append((tr_name, tr))

            if n > 1:
                xtemp, ytemp = transform(tr, xtemp, ytemp)

        self.transform_time_ = time() - t0_f

        f = os.path.join(path, self.name_index)
        o = IndexedEstimator(estimator=fitted_transformers,
                             name=self.name_index,
                             index=self.index,
                             in_index=None,
                             out_index=None,
                             data=self.data)
        pickle_save(o, f)
        if self.verbose:
            f = "stdout" if self.verbose < 10 else "stderr"
            msg = "{:<30} {}".format(self.name_index, "done")
            print_time(t0, msg, file=f)

    @property
    def data(self):
        """fit data"""
        return {'ft': self.transform_time_}


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
    def __init__(self,
                 estimator,
                 preprocess,
                 indexer,
                 name,
                 attr,
                 scorer,
                 error_score=None,
                 verbose=False,
                 raise_on_exception=False):
        super(EvalLearner, self).__init__(
            estimator=estimator,
            preprocess=preprocess,
            indexer=indexer,
            name=name,
            attr=attr,
            scorer=scorer,
            output_columns={0: 0},
            verbose=verbose,
            raise_on_exception=raise_on_exception)
        self.error_score = error_score

    def gen_fit(self, X, y, P=None):
        """Generator for fitting learner on given data"""
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if self.indexer is None:
            raise ValueError("Cannot run cross-validation without an indexer")

        for i, (train_index, test_index) in enumerate(
                self.indexer.generate()):
            # Note that we bump index[1] by 1 to have index[1] start at 1
            if self._partitions == 1:
                index = (0, i + 1)
            else:
                index = (0, i % self._partitions + 1)

            yield EvalSubLearner(learner=self,
                                 estimator=self.estimator,
                                 in_index=train_index,
                                 out_index=test_index,
                                 in_array=X,
                                 targets=y,
                                 index=index)


class EvalSubLearner(SubLearner):

    """EvalSubLearner

    sub-routine for cross-validated evaluation.
    """
    def __init__(self,
                 learner,
                 estimator,
                 in_index,
                 out_index,
                 in_array,
                 targets,
                 index):

        super(EvalSubLearner, self).__init__(learner=learner,
                                             estimator=estimator,
                                             in_index=in_index,
                                             out_index=out_index,
                                             in_array=in_array,
                                             out_array=None,
                                             targets=targets,
                                             index=index)
        self.error_score = learner.error_score
        self.train_score_ = None
        self.test_score_ = None
        self.train_pred_time_ = None
        self.test_pred_time_ = None

    def fit(self, path):
        """Evaluate sub-learner"""
        if self.scorer is None:
            raise ValueError("Cannot generate CV-scores without a scorer")
        t0 = time()
        transformers = self._load_preprocess(path)
        self._fit(transformers)
        self._predict(transformers)

        f = os.path.join(path, self.name_index)
        o = IndexedEstimator(estimator=self.estimator,
                             name=self.name_index,
                             index=self.index,
                             in_index=self.in_index,
                             out_index=self.out_index,
                             data=self.data)
        pickle_save(o, f)

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
        xtemp, ytemp = slice_array(self.in_array,
                                   self.targets,
                                   index)
        for _, tr in transformers:
            xtemp = tr.transform(xtemp)

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
#              'test_pred_time': self.train_pred_time_,
               }
        return out
