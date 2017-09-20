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

from ..utils import pickle_save, pickle_load, load
from ..utils.exceptions import NotFittedError
from ..externals.sklearn.base import clone
from ._base_functions import (_slice_array,
                              _transform,
                              _assign_predictions,
                              _score_predictions)


class IndexedEstimator(object):
    """Indexed Estimator

    Lightweight wrapper around estimator dumps during fitting.

    """
    __slots__ = ['_estimator', 'name', 'index',
                 'in_index', 'out_index', 'score']

    def __init__(self, estimator, name, index, in_index, out_index, score):
        self._estimator = estimator
        self.name = name
        self.index = index
        self.in_index = in_index
        self.out_index = out_index
        self.score = score

    @property
    def estimator(self):
        """Deep copy of estimator"""
        return deepcopy(self._estimator)

    @estimator.setter
    def estimator(self, estimator):
        self._estimator = estimator


class Learner(object):

    """Base Learner

    Wrapper for base learners.

    Parameters
    ----------
    estimator : obj
        estimator to construct learner from

    preprocess : str, None
        Preprocessing pipeline. String if fetching from cache,
        None if no preprocessing.

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
                 partitions,
                 n_prediction_features,
                 output_columns,
                 scorer,
                 fit_on_all,
                 raise_on_exception=True):

        self._estimator = estimator
        self.preprocess = preprocess
        self._indexer = indexer
        self.name = name
        self.attr = attr
        self.partitions = partitions
        self.n_prediction_features = n_prediction_features
        self.output_columns = output_columns
        self._scorer = scorer
        self.fit_on_all = fit_on_all
        self.raise_on_exception = raise_on_exception

        self._fitted_estimators = None   # All fitted estimators
        self._fitted_learner = None      # Estimators for predict
        self._scores = None              # Scores during cv fit
        self.__fitted__ = False          # Status flag

    def gen_trans(self, X, P):
        """Generator for regenerating training predictions"""
        return self._gen_pred(X, P, self.fitted_estimators)

    def gen_pred(self, X, P):
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

    def gen_fit(self, X, y, P):
        """Generator for fitting learner on given data"""
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if self.fit_on_all:
            # Let index[1] == 0 for all full fits
            # Hence fold_index = 0 -> final base learner
            if self.partitions == 1:
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
                                     out_array=P,
                                     index=index)
                    index[0] = index[0] + 1

        if self.indexer is not None:
            # Fit sub-learners on cv folds
            for i, (train_index, test_index) in enumerate(
                    self.indexer.generate()):
                # Note that we bump index[1] by 1 to have index[1] start at 1
                if self.partitions == 1:
                    index = (0, i + 1)
                else:
                    index = (i // self.partitions, i % self.partitions + 1)

                yield SubLearner(learner=self,
                                 estimator=self.estimator,
                                 in_index=train_index,
                                 out_index=test_index,
                                 in_array=X,
                                 targets=y,
                                 out_array=P,
                                 index=index)

    @property
    def fitted_estimators(self):
        """Generator of fitted estimators of base learner"""
        if not self.__fitted__:
            raise NotFittedError("Learner instance not fitted.")

        for estimator in self._fitted_estimators:
            yield deepcopy(estimator)

    @fitted_estimators.setter
    def fitted_estimators(self, path):
        """Load fitted estimator from cache"""
        files = [os.path.join(path, f)
                 for f in os.listdir(path)
                 if f.startswith(self.name)]

        learners, estimators, scores = list(), list(), list()
        for f in sorted(files):
            o = pickle_load(f)
            if o.index[1] == 0:
                learners.append(o)
            else:
                estimators.append(o)

            if o.score is not None:
                scores.append(o.name, o.score)
                del o.score

        self._fitted_learner = learners
        self._fitted_estimators = estimators
        self._scores = scores
        self.__fitted__ = True

    @fitted_estimators.deleter
    def fitted_estimators(self):
        """Clear load"""
        self._fitted_estimators = None
        self._fitted_learner = None
        self._scores = None
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
        self.attr = learner.attr
        self.preprocess = learner.preprocess
        self.in_index = in_index
        self.out_index = out_index
        self.in_array = in_array
        self.targets = targets
        self.out_array = out_array
        self.name = learner.name
        self.scorer = learner.scorer
        self.raise_on_exception = learner.raise_on_exception
        self.score_ = None
        self.output_columns = learner.output_columns[index[0]]
        self.index = tuple(index)
        self.name_index = '__'.join(
            [self.name, *tuple((str(i) for i in index))])

        if self.preprocess is not None:
            self.preprocess_index = '__'.join(
                [self.preprocess, *tuple((str(i) for i in index))])
        else:
            self.processing_index = ''

    def fit(self, path):
        """Fit sub-learner"""
        transformers = self._load_preprocess(path)
        self._fit(transformers)

        if self.out_array is not None:
            self._predict(transformers, self.scorer is not None)

        f = os.path.join(path, self.name_index)
        o = IndexedEstimator(self.estimator,
                             self.name_index,
                             self.index,
                             self.in_index,
                             self.out_index,
                             self.score_)
        pickle_save(o, f)

    def predict(self, path):
        """Predict with sublearner"""
        transformers = self._load_preprocess(path)

        self._predict(transformers, False)

    def _fit(self, transformers):
        """Sub-routine to fit sub-learner"""
        xtemp, ytemp = _slice_array(self.in_array,
                                    self.targets,
                                    self.in_index)

        # Transform input (triggers copying)
        for _, tr in transformers:
            xtemp, ytemp = _transform(tr, xtemp, ytemp)

        # Fit estimator
        self.estimator.fit(xtemp, ytemp)

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

        for _, tr in transformers:
            xtemp = tr.transform(xtemp)

        predictions = getattr(self.estimator, self.attr)(xtemp)

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


class Transformer(object):
    """Preprocessing handler.

    """
    def __init__(self,
                 pipeline,
                 indexer,
                 name,
                 partitions,
                 fit_on_all,
                 raise_on_exception):
        self._pipeline = pipeline
        self._indexer = indexer
        self.name = name
        self.partitions = partitions
        self.fit_on_all = fit_on_all
        self.raise_on_exception = raise_on_exception

        self._learner_pipeline = None
        self._estimator_pipeline = None
        self.__fitted__ = False

    @property
    def fitted_pipelines(self):
        """Copy of fitted pipelines"""
        if not self.__fitted__:
            raise NotFittedError("Transformer instance not fitted.")
        pipes = list()
        ls = self.learner_pipeline
        es = self.estimator_pipeline
        if ls is not None:
            pipes.extend(ls)
        if es is not None:
            pipes.extend(es)
        return pipes

    @fitted_pipelines.setter
    def fitted_pipelines(self, path):
        """Load fitted pipelines from cache"""
        files = list()
        for f in os.listdir(path):
            if f.startswith(self.name):
                files.append(os.path.join(path, f))

        learner_pipelines, estimator_pipelines = list(), list()
        for f in sorted(files):
            o = pickle_load(f)
            if o.index[1] == 0:
                learner_pipelines.append(o)
            else:
                estimator_pipelines.append(o)

        self._learner_pipeline = learner_pipelines
        self._estimator_pipeline = estimator_pipelines
        self.__fitted__ = True

    @fitted_pipelines.deleter
    def fitted_pipelines(self):
        """Clear load"""
        self._estimator_pipeline = None
        self._learner_pipeline = None
        self.__fitted__ = False

    @property
    def learner_pipeline(self):
        """Copy of fitted pipeline for base learner"""
        if not self.__fitted__:
            raise NotFittedError("Transformer instance not fitted.")

        return [deepcopy(pipe) for pipe in self._learner_pipeline]

    @property
    def estimator_pipeline(self):
        """Copy of fitted pipeline for sub-learners"""
        if not self.__fitted__:
            raise NotFittedError("Transformer instance not fitted.")

        return [deepcopy(pipe) for pipe in self._estimator_pipeline]

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

    def gen_pred(self, path):
        """Dump learner pipeline to cache."""
        self.dump(path, self.learner_pipeline)

    def gen_trans(self, path):
        """Dump learner pipeline to cache."""
        self.dump(path, self.estimator_pipeline)

    @staticmethod
    def dump(path, pipeline):
        """Dump pipelines to cache"""
        if pipeline is not None:
            for pipe in pipeline:
                f = os.path.join(path, pipe.name)
                pickle_save(pipe, f)

    def gen_fit(self, X, y):
        """Generator for fitting pipeline on given data"""
        # We use an index to keep track of partition and fold
        # For single-partition estimations, index[0] is constant
        index = [0, 0]
        if self.fit_on_all:
            # Let index[1] == 0 for all full fits
            # Hence fold_index = 0 -> final base learner
            if self.partitions == 1:
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
                if self.partitions == 1:
                    index = (0, i + 1)
                else:
                    index = (i // self.partitions, i % self.partitions + 1)

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
        self.name = transformer.name
        self.name_index = '__'.join(
            [self.name, *tuple((str(i) for i in index))])

    def fit(self, path):
        """Fit transformers"""
        n = len(self.pipeline)
        xtemp, ytemp = _slice_array(self.in_array,
                                    self.targets,
                                    self.in_index)

        fitted_transformers = list()
        for tr_name, tr in self.pipeline:
            tr.fit(xtemp, ytemp)
            fitted_transformers.append((tr_name, tr))

            if n > 1:
                xtemp, ytemp = _transform(tr, xtemp, ytemp)

        f = os.path.join(path, self.name_index)
        o = IndexedEstimator(estimator=fitted_transformers,
                             name=self.name_index,
                             index=self.index,
                             in_index=None,
                             out_index=None,
                             score=None)
        pickle_save(o, f)
