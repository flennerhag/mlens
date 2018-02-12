"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Helpers to generate test cases
"""

from __future__ import division, print_function, with_statement

import os
import shutil
import warnings

from abc import abstractmethod
import numpy as np

from ..utils.utils import pickle_save, pickle_load
from ..utils.dummy import OLS, LogisticRegression, Scale
from ..externals.sklearn.base import clone
from ..index import INDEXERS
from ..ensemble.base import Sequential
from ..estimators import LayerEnsemble
from ..parallel import (
    ParallelProcessing, Learner, Transformer, Layer, make_group, Pipeline)

##############################################################################
PREPROCESSING = {'no': [], 'sc': [('scale', Scale())]}

ESTIMATORS = {'sc': [('offs', OLS(offset=2)), ('null', OLS())],
              'no': [('offs', OLS(offset=2)), ('null', OLS())]}

ESTIMATORS_PROBA = {'sc': [('offs', LogisticRegression(offset=2)),
                           ('null', LogisticRegression())],
                    'no': [('offs', LogisticRegression(offset=2)),
                           ('null', LogisticRegression())]}


ECM = [('ols-%i' % i, OLS(offset=i)) for i in range(4)]
ECM_PROBA = [('lr-%i' % i, LogisticRegression(offset=i)) for i in range(4)]


def mae(y, p): return np.mean((y - p) ** 2)


##############################################################################
# pylint: disable=too-few-public-methods
class InitMixin(object):

    """Mixin to make a mlens ensemble behave as Scikit-learn estimator.

    Scikit-learn expects an estimator to be fully initialized when
    instantiated, but an ML-Ensemble estimator requires layers to be
    initialized before calling ``fit`` or ``predict`` makes sense.

    ``InitMixin`` is intended to be used to create temporary test classes
    of proper mlens ensemble classes that are identical to the parent class
    except that ``__init__`` will also initialize one layer with one
    estimator, and if applicable one meta estimator.

    The layer estimator and the meta estimator are both the dummy
    ``AverageRegressor`` class to minimize complexity and avoids raising
    errors due to the estimators in the layers.

    To create a testing class, modify the ``__init__`` of the test class
    to call ``super().__init__`` as in the example below.

    Examples
    --------

    Assert the :class:`SuperLearner` passes the Scikit-learn estimator test

    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from mlens.ensemble import SuperLearner
    >>> from mlens.testing.dummy import InitMixin
    >>>
    >>> class TestSuperLearner(InitMixin, SuperLearner):
    ...
    ...     def __init__(self):
    ...         super(TestSuperLearner, self).__init__()
    >>>
    >>> check_estimator(TestSuperLearner)
    """

    @abstractmethod
    def __init__(self):

        # Instantiate class
        super(InitMixin, self).__init__()

        # The test is parallelized and Scikit-learn estimators default to
        # n_jobs = 1, so need to coerce ensembles to the same behavior
        self.n_jobs = 1

        # Build an ensemble consisting of two OLS estimators in the first
        # layer, and a single on top.
        getattr(self, 'add')([OLS(offset=1), OLS(offset=2)])
        getattr(self, 'add_meta')(OLS())


###############################################################################
class EstimatorContainer(object):

    """Class for generating architectures of various types."""

    def __init__(self):
        pass

    def get_learner(self, kls, proba, preprocessing, *args, **kwargs):
        """Generate learner and transformer

        Parameters
        ----------
        kls : str
            class type

        proba : bool
            whether to set ``proba`` to ``True``

        preprocessing : bool
            layer with preprocessing cases
        """
        indexer, kwargs = self.load_indexer(kls, args, kwargs)
        est = OLS() if not proba else LogisticRegression()

        if preprocessing:
            transformer = Transformer(
                estimator=Pipeline(Scale(), return_y=True),
                indexer=indexer, name='sc')
        else:
            transformer = None

        learner = Learner(estimator=est, indexer=indexer,
                          preprocess='sc' if transformer else None,
                          scorer=mae if not proba else None,
                          name=kls, proba=proba)

        return learner, transformer

    def get_layer(self, kls, proba, preprocessing, *args, **kwargs):
        """Generate a layer instance.

        Parameters
        ----------
        kls : str
            class type

        proba : bool
            whether to set ``proba`` to ``True``

        preprocessing : bool
            layer with preprocessing cases
        """
        indexer, kwargs = self.load_indexer(kls, args, kwargs)

        learner_kwargs = {'proba': kwargs.pop('proba', proba),
                          'scorer': kwargs.pop('scorer', None)}

        layer = Layer(name='layer', dtype=np.float64, **kwargs)

        if preprocessing:
            ests = ESTIMATORS_PROBA if proba else ESTIMATORS
            prep = PREPROCESSING
        else:
            ests = ECM_PROBA if proba else ECM
            prep = []

        group = make_group(indexer, ests, prep,
                           learner_kwargs=learner_kwargs, name='group')
        layer.push(group)
        return layer

    def get_layer_estimator(self, kls, proba, preprocessing, *args, **kwargs):
        """Generate a layer estimator instance.

        Parameters
        ----------
        kls : str
            class type

        proba : bool
            whether to set ``proba`` to ``True``

        preprocessing : bool
            layer with preprocessing cases
        """
        indexer, kwargs = self.load_indexer(kls, args, kwargs)

        learner_kwargs = {'proba': kwargs.pop('proba', proba),
                          'scorer': kwargs.pop('scorer', None)}

        if preprocessing:
            ests = ESTIMATORS_PROBA if proba else ESTIMATORS
            prep = PREPROCESSING
        else:
            ests = ECM_PROBA if proba else ECM
            prep = []

        group = make_group(indexer, ests, prep,
                           learner_kwargs=learner_kwargs)
        layer = LayerEnsemble([group], dtype=np.float64, **kwargs)
        return layer

    def get_sequential(self, kls, proba, preprocessing, *args, **kwargs):
        """Generate a layer container instance.

        Parameters
        ----------
        kls : str
            class type

        proba : bool
            whether to set ``proba`` to ``True``

        preprocessing : bool
            layer with preprocessing cases
        """
        lyr = self.get_layer(kls, proba, preprocessing, *args, **kwargs)
        lyr.name += '-1'
        seq = Sequential()
        return seq.push(lyr)

    @staticmethod
    def load_indexer(kls, args, kwargs):
        """Load indexer and return remaining kwargs"""
        indexer = INDEXERS[kls]
        idx_kwargs = dict()
        for var in indexer.__init__.__code__.co_varnames:
            if var in kwargs:
                idx_kwargs[var] = kwargs.pop(var)
        indexer = indexer(*args, **idx_kwargs)
        return indexer, kwargs


class Data(object):

    """Class for getting data."""

    def __init__(self, cls, proba, preprocessing, is_learner=False,
                 *args, **kwargs):
        self.lr = is_learner
        self.proba = proba
        self.preprocessing = preprocessing
        self.cls = cls
        self.indexer = INDEXERS[cls](*args, **kwargs)

    def get_data(self, shape, m):
        """Generate X and y data with X.
        Parameters
        ----------
        shape : tuple
            shape of data to be generated
        m : int
            length of step function for y
        Returns
        -------
        train : ndarray
            generated as a sequence of  reshaped to (LEN, WIDTH)
        labels : ndarray
            generated as a step-function with a step every ``m``. As such,
            each prediction fold during cross-validation have
            a unique level value.
        """
        s = shape[0]
        w = shape[1]

        train = np.array(range(int(s * w)), dtype='float').reshape((s, w))
        train += 1

        labels = np.zeros(train.shape[0])

        if not self.proba:
            increment = 10
            for i in range(0, s, m):
                labels[i:i + m] += increment

                increment += 10

        else:
            labels = np.arange(train.shape[0]) % 2

        return train, labels

    def ground_truth(self, X, y, subsets=1, feature_prop=None):
        """Set up an experiment ground truth.
        Returns
        -------
        F : ndarray
            Full prediction array (train errors)
        P : ndarray
            Folded prediction array (test errors)
        Raises
        ------
        AssertionError :
            Raises assertion error if any weight vectors overlap or any
            predictions (as measured by columns in F and P) are judged to be
            equal.
        """
        P, weights_p = self._full_ests(X, y, subsets)
        if feature_prop:
            P = np.hstack([X[:, :feature_prop], P])

        if self.indexer.__class__.__name__.lower() == 'fullindex':
            return (P, weights_p), (P, weights_p)

        F, weights_f = self._folded_ests(X, y, subsets)
        if feature_prop:
            r = X.shape[0] - F.shape[0]
            F = np.hstack([X[r:][:, :feature_prop], F])

        return (F, weights_f), (P, weights_p)

    def _set_up_est(self, y):
        """Get estimators, preprocessing, num_ests, predict attr to use."""
        attr = 'predict_proba' if self.proba else 'predict'
        labels = len(np.unique(y)) if self.proba else 1

        if self.preprocessing:
            if self.lr:
                est = OLS() if not self.proba else LogisticRegression()
                ests = {'sc': [('est', est)]}
                prep = {'sc': [('scale', Scale())]}
            else:
                ests = ESTIMATORS_PROBA if self.proba else ESTIMATORS
                prep = PREPROCESSING

            n_ests = 0
            for case in ests:
                for _ in ests[case]:
                    n_ests += 1

        else:
            if self.lr:
                est = OLS() if not self.proba else LogisticRegression()
                ests = {'no-case': [('ols', est)]}
            else:
                ests = {'no-case': ECM_PROBA if self.proba else ECM}
            prep = {'no-case': []}

            n_ests = len(ests['no-case'])

        self.classes_ = labels
        self.n_pred = n_ests
        if self.cls == 'subsemble':
            self.n_pred *= self.indexer.partitions

        return ests, prep, n_ests, attr, labels

    def _folded_ests(self, X, y, subsets=1):
        """Build ground truth for each fold."""
        ests, prep, n_ests, attr, labels = self._set_up_est(y)

        t = [t for _, t in self.indexer.generate(X, True)]
        t = np.unique(np.hstack(t))
        t.sort()

        weights = []
        F = np.zeros((len(t), n_ests * subsets * labels), dtype=np.float)

        col_id = {}
        col_ass = 0

        # Sort at every occasion
        vps = [(c, e[0])
               for c in prep.keys()
               for e in ests[c]]

        for meta_key in sorted(vps):
            key, est_name = meta_key
            est = dict(ests[key])[est_name]
            for i, (tri, tei) in enumerate(self.indexer.generate(X, True)):

                if subsets > 1:
                    i = i // self.indexer.folds
                else:
                    i = 0

                if '%s-%s-%s' % (i, key, est_name) not in col_id:
                    col_id['%s-%s-%s' % (i, key, est_name)] = col_ass
                    col_ass += labels

                xtrain = X[tri]
                xtest = X[tei]

                # Transform inputs
                for _, tr in prep[key]:
                    t = clone(tr)
                    xtrain = t.fit_transform(xtrain)
                    xtest = t.transform(xtest)

                # Fit estimator
                e = clone(est).fit(xtrain, y[tri])
                w = e.coef_
                weights.append(w)

                # Get out-of-sample predictions
                p = getattr(e, attr)(xtest)

                rebase = X.shape[0] - F.shape[0]
                fix = tei - rebase

                if labels == 1:
                    F[fix, col_id['%s-%s-%s' % (i, key, est_name)]] = p
                else:
                    c = col_id['%s-%s-%s' % (i, key, est_name)]
                    F[np.ix_(fix, np.arange(c, c + labels))] = p

        return F, weights

    def _full_ests(self, X, y, subsets=1):
        """Get ground truth for train and predict on full data."""
        ests, prep, n_ests, attr, labels = self._set_up_est(y)
        if subsets == 1:
            indexer = DummyPartition(np.arange(X.shape[0]))
        else:
            indexer = self.indexer
            indexer.fit(X)

        P = np.zeros((X.shape[0], n_ests * subsets * labels), dtype=np.float)

        weights = list()
        col_id = {}
        col_ass = 0

        vps = ['%s__%s' % (c, e[0])
               for c in prep.keys()
               for e in ests[c]]
        for meta_key in sorted(vps):
            key, est_name = meta_key.split('__')
            est = dict(ests[key])[est_name]
            for i, tri in enumerate(indexer.partition(as_array=True)):
                if '%s-%s-%s' % (i, key, est_name) not in col_id:
                    col_id['%s-%s-%s' % (i, key, est_name)] = col_ass
                    col_ass += labels

                # Transform input
                xtrain = X[tri]
                ytrain = y[tri]

                xtest = X

                for _, tr in prep[key]:
                    t = clone(tr)
                    xtrain = t.fit_transform(xtrain)
                    xtest = t.transform(xtest)

                # Fit est
                e = clone(est).fit(xtrain, ytrain)
                w = e.coef_
                weights.append(w.tolist())

                # Predict
                p = getattr(e, attr)(xtest)
                c = col_id['%s-%s-%s' % (i, key, est_name)]
                if labels == 1:
                    P[:, c] = p
                else:
                    P[:, c:c + labels] = p

        return P, weights


class DummyPartition(object):

    """Dummy class to generate tri."""

    def __init__(self, tri):
        self.tri = tri

    def partition(self, as_array=True):
        """Return the tri index."""
        yield self.tri


###############################################################################
def _get_path(backend, is_learner):
    """Helper to get a path dir"""
    if backend in ['manual', 'multiprocessing']:
        path = os.path.join(os.getcwd(), '.mlens_testing_tmp')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    else:
        path = list() if is_learner else dict()
    return path


def get_learner(case, *args):
    """Generator function for test"""
    data = Data(*args, is_learner=True)
    p = getattr(data.indexer, 'partitions', 1)
    X, y = data.get_data((12, 4), 2)
    (F, wf), (H, wh) = data.ground_truth(X, y, p)
    data.indexer.fit(X)

    learner, transformer = EstimatorContainer().get_learner(
        *args, classes=data.classes_)

    return {'fit': ('fit', learner, transformer, X, y, F, wf),
            'predict': ('predict', learner, transformer, X, y, H, wh),
            'transform': ('transform', learner, transformer, X, y, F, wf)}[case]


def run_learner(job, learner, transformer, X, y, F, wf=None):
    """Sub-routine for running learner job"""
    if job != 'fit':
        run_learner('fit', learner, transformer, X, y, F)

    P = np.zeros(F.shape)

    path = _get_path('manual', is_learner=True)
    args = {
        'dir': path,
        'job': job,
        'auxiliary': {'X': X, 'y': y} if job == 'fit' else {'X': X},
        'main': {'X': X, 'y': y, 'P': P} if job == 'fit' else {'X': X, 'P': P}
        }

    if transformer:
        transformer.setup(X, y, job)
        for obj in transformer(args, 'auxiliary'):
            obj()

    learner.setup(X, y, job)
    for obj in learner(args, 'main'):
        obj()

    if job == 'fit':
        learner.collect()
        if transformer:
            transformer.collect()

    if isinstance(path, str):
        try:
            shutil.rmtree(path)
        except OSError:
            warnings.warn("Failed to destroy temporary test cache at %s" % path)

    if wf is not None:
        if job in ['fit', 'transform']:
            lrs = learner.sublearners
        else:
            lrs = learner.learner
        np.testing.assert_array_equal(P, F)
        w = [obj.estimator.coef_ for obj in lrs]
        np.testing.assert_array_equal(w, wf)


def get_layer(job, backend, case, proba, preprocess, feature_prop=None):
    """Generator function for test"""
    data = Data(case, proba, preprocess)
    p = getattr(data.indexer, 'partitions', 1)
    X, y = data.get_data(shape=(12, 4), m=2)
    (F, wf), (H, wh) = data.ground_truth(X, y, p, feature_prop=feature_prop)
    data.indexer.fit(X)

    prop = range(feature_prop) if feature_prop else None
    layer = EstimatorContainer().get_layer(
        kls=case, proba=proba, preprocessing=preprocess,
        backend=backend, propagate_features=prop)

    return {'fit': ('fit', layer, X, y, F, wf),
            'predict': ('predict', layer, X, y, H, wh),
            'transform': ('transform', layer, X, y, F, wf)}[job]


def run_layer(job, layer, X, y, F, wf=None):
    """Sub-routine for running learner job"""
    if job != 'fit':
        run_layer('fit', layer, X, y, F)

    P = np.zeros(F.shape)

    path = _get_path(layer.backend, is_learner=False)
    args = {
        'dir': path,
        'job': job,
        'auxiliary': {'X': X, 'y': y} if job == 'fit' else {'X': X},
        'main': {'X': X, 'y': y, 'P': P} if job == 'fit' else {'X': X, 'P': P}
        }

    if layer.backend == 'manual':
        # Run manually
        layer.setup(X, y, job)
        if layer.transformers:
            for transformer in layer.transformers:
                for subtransformer in transformer(args, 'auxiliary'):
                    subtransformer()
        for learner in layer.learners:
            for sublearner in learner(args, 'main'):
                sublearner()

        if job == 'fit':
            layer.collect()

    else:
        args = (X, y) if job == 'fit' else (X,)
        with ParallelProcessing(layer.backend, layer.n_jobs) as manager:
            P = manager.map(layer, job, *args, path=path, return_preds=True)

    if isinstance(path, str) and layer.backend == 'manual':
        try:
            shutil.rmtree(path)
        except OSError:
            warnings.warn(
                "Failed to destroy temporary test cache at %s" % path)

    if wf is not None:
        if job in ['fit', 'transform']:
            w = [obj.estimator.coef_
                 for lr in layer.learners for obj in lr.sublearners]
        else:
            w = [obj.estimator.coef_
                 for lr in layer.learners for obj in lr.learner]
        np.testing.assert_array_equal(P, F)
        np.testing.assert_array_equal(w, wf)

        assert P.__class__.__name__ == 'ndarray'
        assert w[0].__class__.__name__ == 'ndarray'


def return_pickled(model):
    """Pickle a model and return the loaded model"""
    loc = str(np.random.randint(0, 1000000))
    pickle_save(model, loc)
    model = pickle_load(loc)
    os.remove(loc + '.pkl')
    return model
