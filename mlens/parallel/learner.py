"""

ML-Ensemble

:author: Sebastian Flennerhag
:license: MIT

Base learner classes
"""
# pylint: disable=protected-access
# pylint: disable=too-few-public-methods
# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

from __future__ import print_function, division

import os
from copy import deepcopy

from ..metrics import Data
from ..utils import (pickle_save,
                     pickle_load,
                     load,
                     safe_print,
                     print_time)
from ..utils.exceptions import NotFittedError
from ..externals.sklearn.base import clone, BaseEstimator
from ._base_functions import (_slice_array,
                              _transform,
                              _assign_predictions,
                              _score_predictions)
try:
    from time import perf_counter as time
except ImportError:
    from time import time


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


class Learner(BaseEstimator):

    """Base Learner

    Wrapper for base learners.

    Parameters
    ----------
    estimator : obj
        estimator to construct learner from

    input :
        object to collect input from.

    indexer : obj, None
        indexer to use. Set to None for no subset calls.

    name : str
        name of learner.

    attr : str (default='predict')
        predict attribute, typically one of 'predict' and 'predict_proba'

    cols : int
        number of prediction features produced by ``attr``.

    exclude : list, optional
        list of columns indexes to exclude from the prediction array.
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

        self._estimator = estimator
        self.preprocess = preprocess
        self._indexer = indexer
        self.output_columns = output_columns
        self.attr = attr
        self._scorer = scorer
        self.raise_on_exception = raise_on_exception
        self.verbose = verbose

        if preprocess:
            name = '%s__%s' % (preprocess, name)
        self._name = name
        self._partitions = getattr(indexer, 'n_partitions', 1)
        self._fit_on_all = indexer.__class__.__name__.lower() != 'blendindex'

        self._fitted_estimators = None   # All fitted estimators
        self._fitted_learner = None      # Estimators for predict
        self._data = None              # Scores during cv fit
        self._state = None               # Output state
        self.__fitted__ = False          # Status flag

    def __call__(self, job, *args, **kwargs):
        """Caller for producing jobs"""
        job = 'gen_%s' % job

        if not hasattr(self, job):
            raise AttributeError("Job %s not accepted." % job)

        return getattr(self, job)(*args, **kwargs)

    def gen_transform(self, X, P):
        """Generator for regenerating training predictions"""
        return self._gen_pred(X, P, self.fitted_sublearners)

    def gen_predict(self, X, P):
        """Generator for predicting test set"""
        return self._gen_pred(X, P, self.fitted_learner)

    def _gen_pred(self, X, P, estimators):
        """Generator for predicting with fitted learner"""
        for estimator in estimators:
            yield SubLearner(learner=self,
                             estimator=estimator.estimator,
                             in_index=estimator.in_index,
                             out_index=estimator.out_index,
                             in_array=X,
                             targets=None,
                             out_array=P,
                             index=estimator.index)

    def gen_fit(self, X, y, P=None):
        """Generator for fitting learner on given data"""
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if self._fit_on_all:
            # Let index[1] == 0 for all full fits
            # Hence fold_index = 0 -> final base learner
            if self._partitions == 1:
                yield SubLearner(learner=self,
                                 estimator=self.estimator,
                                 in_index=None,
                                 out_index=None,
                                 in_array=X,
                                 targets=y,
                                 out_array=None,
                                 index=index)
            else:
                for partition_index in self.indexer.partition():
                    yield SubLearner(learner=self,
                                     estimator=self.estimator,
                                     in_index=partition_index,
                                     out_index=None,
                                     in_array=X,
                                     targets=y,
                                     out_array=None,
                                     index=index)
                    index[0] = index[0] + 1

        if self.indexer is not None:
            # Fit sub-learners on cv folds
            for i, (train_index, test_index) in enumerate(
                    self.indexer.generate()):
                # Note that we bump index[1] by 1 to have index[1] start at 1
                if self._partitions == 1:
                    index = (0, i + 1)
                else:
                    index = (i // self._partitions, i % self._partitions + 1)

                yield SubLearner(learner=self,
                                 estimator=self.estimator,
                                 in_index=train_index,
                                 out_index=test_index,
                                 in_array=X,
                                 targets=y,
                                 out_array=P,
                                 index=index)

    @property
    def fitted_sublearners(self):
        """Generator of fitted estimators of base learner"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")

        for estimator in self._fitted_estimators:
            yield deepcopy(estimator)

    def collect(self, path):
        """Load fitted estimator from cache"""
        files = [os.path.join(path, f)
                 for f in os.listdir(path)
                 if f.startswith(self._name)]

        learners, estimators, data = list(), list(), list()
        for f in sorted(files):
            o = pickle_load(f)
            if o.index[1] == 0:
                learners.append(o)
            else:
                estimators.append(o)

            if o.data is not None:
                data.append((o.name, o.data))

            del o.data

        self._fitted_learner = learners
        self._fitted_estimators = estimators
        self._data = data
        self.__fitted__ = True

    def clear(self):
        """Clear load"""
        self._fitted_estimators = None
        self._fitted_learner = None
        self._data = None
        self.__fitted__ = False

    @property
    def fitted_learner(self):
        """Fitted learner"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")

        for estimator in self._fitted_learner:
            yield deepcopy(estimator)

    @property
    def estimator(self):
        """Original unfitted estimator"""
        return clone(self._estimator)

    @estimator.setter
    def estimator(self, estimator):
        """Replace blueprint estimator"""
        self._estimator = estimator

    @property
    def raw_data(self):
        """CV scores during prediction"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")
        return self._data

    @property
    def data(self):
        """CV scores during prediction"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")
        return Data(self._data, self._partitions)

    @property
    def scorer(self):
        """Copy of learner scorer"""
        return deepcopy(self._scorer)

    @scorer.setter
    def scorer(self, scorer):
        """Replace blueprint scorer"""
        self._scorer = scorer

    @property
    def indexer(self):
        """(Deep) copy of indexer"""
        return deepcopy(self._indexer)

    @indexer.setter
    def indexer(self, indexer):
        """Replace indexer"""
        self._indexer = indexer


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

        self.name = learner._name
        self.name_index = '__'.join(
            [self.name, *tuple((str(i) for i in index))])

        if self.preprocess is not None:
            self.preprocess_index = '__'.join(
                [self.preprocess, *tuple((str(i) for i in index))])
        else:
            self.processing_index = ''

    def __call__(self, job, path):
        """Launch job"""
        if not hasattr(self, job):
            raise NotImplementedError(
                "SubLearner does not implement [%s]' % job")
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
            print_time(t0, "%s done" % self.name_index, end='')

    def predict(self, path):
        """Predict with sublearner"""
        t0 = time()
        transformers = self._load_preprocess(path)

        self._predict(transformers, False)
        if self.verbose:
            print_time(t0, "%s done" % self.name_index, end='')

    def transform(self, path):
        """Predict with sublearner"""
        return self.predict(path)

    def _fit(self, transformers):
        """Sub-routine to fit sub-learner"""
        xtemp, ytemp = _slice_array(self.in_array,
                                    self.targets,
                                    self.in_index)

        # Transform input (triggers copying)
        t0 = time()
        for _, tr in transformers:
            xtemp, ytemp = _transform(tr, xtemp, ytemp)

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
        xtemp, ytemp = _slice_array(self.in_array,
                                    self.targets,
                                    self.out_index)
        t0 = time()
        for _, tr in transformers:
            xtemp = tr.transform(xtemp)

        predictions = getattr(self.estimator, self.attr)(xtemp)
        self.pred_time_ = time() - t0

        # Assign predictions to matrix
        _assign_predictions(self.out_array,
                            predictions,
                            self.out_index,
                            self.output_columns,
                            n)

        # Score predictions if applicable
        if score_preds:
            self.score_ = _score_predictions(
                ytemp, predictions, self.scorer,
                self.name_index, self.name)

    @property
    def data(self):
        """fit data"""
        out = {'ft': self.fit_time_,
               'pt': self.pred_time_,
               'score': self.score_}
        return out


class Transformer(BaseEstimator):
    """Preprocessing handler.

    """
    def __init__(self,
                 pipeline,
                 indexer,
                 name,
                 verbose=False,
                 raise_on_exception=True):
        self._pipeline = pipeline
        self._indexer = indexer
        self.verbose = verbose
        self.raise_on_exception = raise_on_exception

        self._partitions = getattr(indexer, 'n_partitions', 1)
        self._fit_on_all = indexer.__class__.__name__.lower() != 'blendindex'

        self._name = name
        self._learner_pipeline = None
        self._estimator_pipeline = None
        self.__fitted__ = False
        self._data = None

    def __call__(self, job, *args, **kwargs):
        """Caller for producing jobs"""
        job = 'gen_%s' % job
        if not hasattr(self, job):
            raise AttributeError("Job [%s] not accepted." % job)
        return getattr(self, job)(*args, **kwargs)

    @property
    def fitted_pipelines(self):
        """Copy of fitted pipelines"""
        if not self.__fitted__:
            raise NotFittedError("Transformer instance not fitted.")
        pipes = list()
        ls = self.learner_pipeline
        es = self.sublearner_pipeline
        if ls is not None:
            pipes.extend(ls)
        if es is not None:
            pipes.extend(es)
        return pipes

    def collect(self, path):
        """Load fitted pipelines from cache"""
        files = list()
        for f in os.listdir(path):
            if f.startswith(self._name):
                files.append(os.path.join(path, f))

        learner_pipelines, estimator_pipelines, data = list(), list(), list()
        for f in sorted(files):
            o = pickle_load(f)
            if o.index[1] == 0:
                learner_pipelines.append(o)
            else:
                estimator_pipelines.append(o)

            if o.data is not None:
                data.append(o.data)

        self._learner_pipeline = learner_pipelines
        self._estimator_pipeline = estimator_pipelines
        self._data = data
        self.__fitted__ = True

    def clear(self):
        """Clear load"""
        self._estimator_pipeline = None
        self._learner_pipeline = None
        self.__fitted__ = False
        self._data = None

    @property
    def learner_pipeline(self):
        """Copy of fitted pipeline for base learner"""
        if not self.__fitted__:
            raise NotFittedError("Transformer instance not fitted.")

        return [deepcopy(pipe) for pipe in self._learner_pipeline]

    @property
    def sublearner_pipeline(self):
        """Copy of fitted pipeline for sub-learners"""
        if not self.__fitted__:
            raise NotFittedError("Transformer instance not fitted.")

        return [deepcopy(pipe) for pipe in self._estimator_pipeline]

    @property
    def raw_data(self):
        """CV scores during prediction"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")
        return self._data

    @property
    def data(self):
        """CV scores during prediction"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")
        return Data(self._data, self._partitions)

    @property
    def indexer(self):
        """Blueprint indexer"""
        return deepcopy(self._indexer)

    @indexer.setter
    def indexer(self, indexer):
        """Update indexer"""
        self._indexer = indexer

    @property
    def pipeline(self):
        """Blueprint pipeline"""
        return [(tr_name, clone(tr))
                for tr_name, tr in self._pipeline]

    @pipeline.setter
    def pipeline(self, pipeline):
        """Update pipeline blueprint"""
        self._pipeline = pipeline

    def gen_predict(self, path):
        """Dump learner pipeline to cache."""
        self.dump(path, self.learner_pipeline)

    def gen_transform(self, path):
        """Dump learner pipeline to cache."""
        self.dump(path, self.sublearner_pipeline)

    def dump(self, path, pipeline):
        """Dump pipelines to cache"""
        if pipeline is not None:
            for pipe in pipeline:
                f = os.path.join(path, pipe.name)
                pickle_save(pipe, f)
                if self.verbose:
                    safe_print("%s cached" % pipe.name, end='')

    def gen_fit(self, X, y):
        """Generator for fitting pipeline on given data"""
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if self._fit_on_all:
            # Let index[1] == 0 for all full fits
            # Hence fold_index = 0 -> final base learner
            if self._partitions == 1:
                yield SubTransformer(transformer=self,
                                     pipeline=self.pipeline,
                                     in_index=None,
                                     in_array=X,
                                     targets=y,
                                     index=index)
            else:
                for partition_index in self.indexer.partition():
                    yield SubTransformer(transformer=self,
                                         pipeline=self.pipeline,
                                         in_index=partition_index,
                                         in_array=X,
                                         targets=y,
                                         index=index)
                    index[0] = index[0] + 1

        if self.indexer is not None:
            # Fit sub-learners on cv folds
            for i, (train_index, _) in enumerate(
                    self.indexer.generate()):
                # Note that we bump index[1] by 1 to have index[1] start at 1
                if self._partitions == 1:
                    index = (0, i + 1)
                else:
                    index = (i // self._partitions, i % self._partitions + 1)

                yield SubTransformer(transformer=self,
                                     pipeline=self.pipeline,
                                     in_index=train_index,
                                     in_array=X,
                                     targets=y,
                                     index=index)


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
        self.name = transformer._name
        self.name_index = '__'.join(
            [self.name, *tuple((str(i) for i in index))])

    def __call__(self, job, path):
        """Launch job"""
        if not hasattr(self, job):
            raise NotImplementedError(
                "SubTransformer does not implement [%s]' % job")
        return getattr(self, job)(path)

    def fit(self, path):
        """Fit transformers"""
        t0 = time()
        n = len(self.pipeline)
        xtemp, ytemp = _slice_array(self.in_array,
                                    self.targets,
                                    self.in_index)

        t0_f = time()
        fitted_transformers = list()
        for tr_name, tr in self.pipeline:
            tr.fit(xtemp, ytemp)
            fitted_transformers.append((tr_name, tr))

            if n > 1:
                xtemp, ytemp = _transform(tr, xtemp, ytemp)

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
            print_time(t0, "%s done" % self.name_index, end='')

    @property
    def data(self):
        """fit data"""
        return {'ft': self.transform_time_}
