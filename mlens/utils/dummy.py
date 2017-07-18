"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Collection of dummy estimator classes, Mixins to build transparent layers for
unit testing.

Also contains pre-made Layer, LayerContainers and data generation functions
for unit testing.
"""

from __future__ import division, print_function

import gc
import os
import shutil
from abc import abstractmethod

import numpy as np
import warnings
from ..externals.joblib import Parallel, dump, load

from ..parallel.manager import  Job
from .exceptions import NotFittedError
from ..externals.sklearn.base import BaseEstimator, TransformerMixin, clone
from ..externals.sklearn.validation import check_X_y, check_array
from ..base import INDEXERS
from ..ensemble.base import Layer, LayerContainer
from ..parallel.manager import ENGINES


class OLS(BaseEstimator):

    """No frills vanilla OLS estimator implemented through the normal equation.

    MWE of a Scikit-learn estimator.

    OLS is a simple estimator designed to allow for total control over
    predictions in unit testing. It implements OLS through the Normal
    Equation, no learning takes place. The ``offset`` option allows
    the user to offset weights by a scalar value, if different instances
    should be differentiated in their predictions.

    Parameters
    ----------
    offset : float (default = 0)
        scalar value to add to the coefficient vector after fitting.

    Examples
    --------

    Asserting the OLS passes the Scikit-learn estimator test

    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from mlens.utils.dummy import OLS
    >>> check_estimator(OLS)

    OLS comparison with Scikit-learn's LinearRegression

    >>> from numpy.testing import assert_array_equal
    >>> from mlens.utils.dummy import OLS
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_boston
    >>> X, y = load_boston(True)
    >>>
    >>> lr = LinearRegression(False)
    >>> lr.fit(X, y)
    >>>
    >>> ols = OLS()
    >>> ols.fit(X, y)
    >>>
    >>> assert_array_equal(lr.coef_, ols.coef_)
    """

    def __init__(self, offset=0):
        self.offset = offset

    def fit(self, X, y):
        """Fit coefficient vector."""
        X, y = check_X_y(X, y, accept_sparse=False)

        O = np.linalg.lstsq(X, y)

        self.coef_ = O[0] + self.offset
        self.resid_ = O[1]

        return self

    def predict(self, X):
        """Predict with fitted weights."""
        if not hasattr(self, 'coef_'):
            raise NotFittedError("Estimator not fitted. Call 'fit' first.")

        X = check_array(X, accept_sparse=False)

        return np.dot(X, self.coef_.T)


# FIXME: Needs a quality check!
class LogisticRegression(OLS):

    """No frill Logistic Regressor w. one-vs-rest estimation of P(label).

    MWE of a Scikit-learn classifier.

    LogisticRegression is a simple classifier estimator designed for
    transparency in unit testing. It implements a Logistic
    Regression with one-vs-rest strategy of classification.

    The estimator is a wrapper around the :class:`OLS`. The OLS
    prediction is squashed using the Sigmoid function, and classification
    is done by picking the label with the highest probability.

    The ``offset`` option allows the user to offset weights in the OLS by a
    scalar value, if different instances should be differentiated in their
    predictions.

    Examples
    --------

    Asserting the LogisticRegression passes the Scikit-learn estimator test

    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from mlens.utils.dummy import LogisticRegression
    >>> check_estimator(LogisticRegression)

    Comparison with Scikit-learn's LogisticRegression

    >>> from mlens.utils.dummy import LogisticRegression as mlensL
    >>> from sklearn.linear_model import LogisticRegression as sklearnL
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification()
    >>>
    >>> slr = sklearnL()
    >>> slr.fit(X, y)
    >>>
    >>> mlr = mlensL()
    >>> mlr.fit(X, y)
    >>>
    >>> (mlr.predict(X) == slr.predict(X)).sum() / y.shape
    array([ 0.98])
    """

    def fit(self, X, y):
        """Fit one model per label."""
        X, y = check_X_y(X, y, accept_sparse=False)

        self.labels_ = np.unique(y)

        models = []
        for label in self.labels_:
            labels = y == label
            models.append(super(LogisticRegression,
                                clone(self)).fit(X, labels))

        self._models_ = models
        self.coef_ = np.vstack([l.coef_ for l in self._models_])

        return self

    def predict_proba(self, X):
        """Get probability predictions."""
        if not hasattr(self, '_models_'):
            raise NotFittedError("Estimator not fitted. Call 'fit' first.")

        X = check_array(X, accept_sparse=False)

        preds = []
        for m in self._models_:

            p = 1 / (1 + np.exp(- m._predict(X)))

            preds.append(p)

        return np.vstack(preds).T

    def _predict(self, X):
        """Original OLS prediction."""
        return super(LogisticRegression, self).predict(X)

    def predict(self, X):
        """Get label predictions."""
        if not hasattr(self, '_models_'):
            raise NotFittedError("Estimator not fitted. Call 'fit' first.")

        X = check_array(X, accept_sparse=False)

        preds = self.predict_proba(X)

        labels = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            labels[i] = self.labels_[preds[i].argmax()]

        return labels


class Scale(BaseEstimator, TransformerMixin):

    """Removes the a learnt mean in a column-wise manner in an array.

    MWE of a Scikit-learn transformer, to be used for unit-tests of ensemble
    classes.

    Parameters
    ----------
    copy : bool (default = True)
        Whether to copy X before transforming.

    Examples
    --------
    Asserting :class:`Scale` passes the Scikit-learn estimator test

    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from mlens.utils.dummy import Scale
    >>> check_estimator(Scale)

    Scaling elements

    >>> from numpy import arange
    >>> from mlens.utils.dummy import Scale
    >>> X = arange(6).reshape(3, 2)
    >>> X[:, 1] *= 2
    >>> print('X:')
    >>> print('%r' % X)
    >>> print('Scaled:')
    >>> S = Scale().fit_transform(X)
    >>> print('%r' % S)
    X:
    array([[ 0,  2],
           [ 2,  6],
           [ 4, 10]])
    Scaled:
    array([[-2., -4.],
           [ 0.,  0.],
           [ 2.,  4.]])
    """

    def __init__(self, copy=True):
        self.copy = copy
        self.__is_fitted__ = False

    def fit(self, X, y=None):
        """Estimate mean.

        Parameters
        ----------
        X : array-like
            training data to fit transformer on.

        y : array-like or None
            pass through for pipeline.
        """
        X = check_array(X, accept_sparse='csr')
        self.__is_fitted__ = True
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        """Transform array by adjusting all elements with scale.

        Parameters
        ----------
        X : ndarray
            matrix to transform.
        """
        if not self.__is_fitted__:
            raise NotFittedError("Estimator not fitted.")
        X = check_array(X, accept_sparse='csr')
        Xt = X.copy() if self.copy else X
        return Xt - self.mean_


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
    >>> from mlens.utils.dummy import InitMixin
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
        if getattr(self, 'layers', None) is None:
            getattr(self, 'add')([OLS(offset=1), OLS(offset=2)])
            getattr(self, 'add_meta')(OLS())


###############################################################################
# Pre-made Layer and LayerContainer classes

PREPROCESSING = {'no': [], 'sc': [('scale', Scale())]}

ESTIMATORS = {'sc': [('offs', OLS(offset=2))],
              'no': [('offs', OLS(offset=2)), ('null', OLS())]}

ESTIMATORS_PROBA = {'sc': [('offs', LogisticRegression(offset=2))],
                    'no': [('offs', LogisticRegression(offset=2)),
                           ('null', LogisticRegression())]}


ECM = [('ols-%i' % i, OLS(offset=i)) for i in range(4)]
ECM_PROBA = [('lr-%i' % i, LogisticRegression(offset=i)) for i in range(4)]


###############################################################################
# Data generation functions and Layer estimation wrappers
class LayerGenerator(object):

    """Class for generating architectures of various types."""

    def __init__(self):
        pass

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

        if preprocessing:
            ests = ESTIMATORS_PROBA if proba else ESTIMATORS
            return Layer(estimators=ests,
                         cls=kls,
                         proba=proba,
                         indexer=indexer,
                         dtype=np.float,
                         partitions=1 if kls != 'subset' else
                         indexer.n_partitions,
                         preprocessing=PREPROCESSING)
        else:
            ests = ECM_PROBA if proba else ECM
            return Layer(estimators=ests,
                         cls=kls,
                         proba=proba,
                         indexer=indexer,
                         dtype=np.float,
                         partitions=1 if kls != 'subset' else
                         indexer.n_partitions)

    def get_layer_container(self, kls, proba, preprocessing, *args, **kwargs):
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
        indexer, kwargs = self.load_indexer(kls, args, kwargs)

        if preprocessing:
            ests = ESTIMATORS_PROBA if proba else ESTIMATORS
            return LayerContainer().add(estimators=ests,
                                        cls=kls,
                                        proba=proba,
                                        indexer=indexer,
                                        preprocessing=PREPROCESSING,
                                        dtype=np.float64,
                                        **kwargs)
        else:
            ests = ECM_PROBA if proba else ECM
            return LayerContainer().add(estimators=ests,
                                        cls=kls,
                                        proba=proba,
                                        indexer=indexer,
                                        dtype=np.float64,
                                        **kwargs)

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


class Cache(object):

    """Object for controlling caching."""

    def __init__(self, X, y, data):
        path = os.path.join(os.getcwd(), 'tmp')
        try:
            shutil.rmtree(path)
        except Exception:
            pass
        os.mkdir(path)
        self.path = path

        paths = {}
        for name, arr in zip(('X', 'y'), (X, y)):
            f = os.path.join(path, '%s.mmap' % name)
            paths[name] = f
            if os.path.exists(f):
                os.unlink(f)
            dump(arr, f)

        X = load(paths['X'], mmap_mode='r')
        y = load(paths['y'], mmap_mode='r')

        # Prediction for fit
        f = os.path.join(path, 'Pf.mmap')
        if os.path.exists(f):
            os.unlink(f)

        n = data.indexer.n_test_samples

        s = data.n_pred

        if data.proba:
            self.classes_ = data.classes_
            s *= data.classes_

        P_f = np.memmap(f, dtype=np.float64, shape=(n, s), mode='w+')

        # Prediction for predict
        f = os.path.join(path, 'Pp.mmap')
        if os.path.exists(f):
            os.unlink(f)

        n = data.indexer.n_samples

        s = data.n_pred
        if data.proba:
            s *= data.classes_

        P_p = np.memmap(f, dtype=np.float64, shape=(n, s), mode='w+')

        # Prediction for transform
        f = os.path.join(path, 'Pt.mmap')
        if os.path.exists(f):
            os.unlink(f)

        n = data.indexer.n_test_samples

        s = data.n_pred
        if data.proba:
            s *= data.classes_

        P_t = np.memmap(f, dtype=np.float64, shape=(n, s), mode='w+')

        self.job = {'X': X,
                    'y': y,
                    'P_fit': P_f,
                    'P_predict': P_p,
                    'P_transform': P_t,
                    'dir': self.path}

    def store_X_y(self, X, y, as_csv=False):
        """Save X and y to file in temporary directory."""
        if not as_csv:
            xf, yf = (os.path.join(self.path, 'X_mapped.npy'),
                      os.path.join(self.path, 'y_mapped.npy'))

            np.save(xf, X)
            np.save(yf, y)

        else:
            xf, yf = (os.path.join(self.path, 'X_mapped.csv'),
                      os.path.join(self.path, 'y_mapped.csv'))

            np.savetxt(xf, X)
            np.savetxt(yf, y)

        return xf, yf

    def layer_est(self, layer, attr):
        """Test the estimation routine for a layer."""
        est = ENGINES[layer.cls]

        # Wrap in try-except to always close the tmp if asked to
        with Parallel(temp_folder=self.job['dir'],
                      mmap_mode='r+',
                      max_nbytes=None) as parallel:

            # Run test
            job = Job(attr)
            job.y = self.job['y']
            job.dir = self.job['dir']
            if attr == 'fit':
                job.P = [self.job['X'], self.job['P_fit']]
            elif attr == 'transform':
                job.P = [self.job['X'], self.job['P_transform']]
            else:
                job.P = [self.job['X'], self.job['P_predict']]

            if hasattr(self, 'classes_'):
                layer.classes_ = self.classes_

            e = est(layer=layer, job=job, n=0)
            e(parallel)

        # Get prediction output
        P = self.job['P_%s' % attr.split('_')[0]]
#        P.flush()
        preds = np.asarray(P)

        return preds

    def terminate(self):
        """Remove temporary items in directory during tests."""
        del self.job
        gc.collect()

        try:
            shutil.rmtree(self.path)
        except OSError:
            warnings.warn("Failed to destroy temporary test cache at %s" % dir)

        os.mkdir(self.path)


class Data(object):

    """Class for getting data."""

    def __init__(self, cls, proba, preprocessing, *args, **kwargs):
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

    def _set_up_est(self, y):
        """Get estimators, preprocessing, num_ests, predict attr to use."""
        attr = 'predict_proba' if self.proba else 'predict'
        labels = len(np.unique(y)) if self.proba else 1

        if self.preprocessing:
            ests = ESTIMATORS_PROBA if self.proba else ESTIMATORS
            prep = PREPROCESSING

            n_ests = 0
            for case in ests:
                for _ in ests[case]:
                    n_ests += 1

        else:
            ests = {'no-case': ECM_PROBA if self.proba else ECM}
            prep = {'no-case': []}

            n_ests = len(ests['no-case'])

        self.classes_ = labels
        self.n_pred = n_ests
        if self.cls == 'subset':
            self.n_pred *= self.indexer.n_partitions

        return ests, prep, n_ests, attr, labels

    def _folded_ests(self, X, y, subsets=1, verbose=True):
        """Build ground truth for each fold."""
        if verbose:
            print('                                    FOLD OUTPUT')
            print('-' * 100)
            print('  EST   |'
                  '    TRI    |'
                  '   TEI     |'
                  '     TEST LABELS    |'
                  '     TRAIN LABELS   |'
                  '      COEF     |'
                  '        PRED')

        ests, prep, n_ests, attr, labels = self._set_up_est(y)

        t = [t for _, t in self.indexer.generate(X, True)]
        t = np.unique(np.hstack(t))
        t.sort()

        weights = []
        F = np.zeros((len(t), n_ests * subsets * labels), dtype=np.float)

        col_id = {}
        col_ass = 0

        # Sort at every occasion
        for key in sorted(prep):
            for i, (tri, tei) in enumerate(self.indexer.generate(X, True)):

                if subsets > 1:
                    i = i // self.indexer.n_splits
                else:
                    i = 0

                for est_name, est in ests[key]:

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
                    weights.append(w.tolist())

                    # Get out-of-sample predictions
                    p = getattr(e, attr)(xtest)

                    rebase = X.shape[0] - F.shape[0]
                    fix = tei - rebase

                    if labels == 1:
                        F[fix, col_id['%s-%s-%s' % (i, key, est_name)]] = p
                    else:
                        c = col_id['%s-%s-%s' % (i, key, est_name)]
                        F[np.ix_(fix, np.arange(c, c + labels))] = p

                    try:
                        if verbose:
                            print('%s | %r | %r | %r | %r | %13r | %r' % (
                                '%s-%s' % (key, est_name),
                                list(tri),
                                list(tei),
                                [float('%.1f' % i) for i in y[tei]],
                                [float('%.1f' % i) for i in y[tri]],
                                [float('%.1f' % i) for i in w],
                                [float('%.1f' % i) for i in p]))
                    except Exception:
                        pass

        return F, weights

    def _full_ests(self, X, y, subsets=1, verbose=True):
        """Get ground truth for train and predict on full data."""
        if verbose:
            print('\n                        FULL PREDICTION OUTPUT')
            print('-' * 100)
            print('  EST   |'
                  '             GROUND TRUTH             |'
                  '    COEF     |'
                  '           PRED')

        ests, prep, n_ests, attr, labels = self._set_up_est(y)

        if subsets == 1:
            tri = [t for t, _ in self.indexer.generate(X, True)]
            tri = np.unique(np.hstack(tri))
            indexer = DummyPartition(tri)
        else:
            indexer = self.indexer

        P = np.zeros((X.shape[0], n_ests * subsets * labels), dtype=np.float)
        weights = list()
        col_id = {}
        col_ass = 0

        for key in sorted(prep):
            for i, tri in enumerate(indexer.partition(as_array=True)):
                for est_name, est in ests[key]:

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

                    try:
                        if verbose:
                            print('%s | %r | %11r | %r' % (
                                '%s-%s' % (key, est_name),
                                [float('%.1f' % i) for i in y],
                                [float('%.1f' % i) for i in w],
                                [float('%.1f' % i) for i in p]))
                    except Exception:
                        pass

        return P, weights

    def ground_truth(self, X, y, subsets=1, verbose=False):
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
        if verbose:
            print('                            CONSTRUCTING GROUND TRUTH\n')

        # Build predictions matrices.
        N = 0
        for case in ESTIMATORS:
            N += len(ESTIMATORS[case])

        F, weights_f = self._folded_ests(X, y, subsets, verbose)
        P, weights_p = self._full_ests(X, y, subsets, verbose)

        if verbose:
            print('\n                 SUMMARY')
            print('-' * 42)

        col = 0
        for case in sorted(ESTIMATORS):
            for est_name, _ in ESTIMATORS[case]:

                if verbose:
                    print('%s | %6s: %20r' % (
                        '%s-%s' % (case, est_name), 'FULL',
                        [float('%.1f' % i) for i in P[:, col]]))
                    print('%s | %6s: %20r' % (
                        '%s-%s' % (case, est_name), 'FOLDS',
                        [float('%.1f' % i) for i in F[:, col]]))

                col += 1

        if verbose:
            print('GT              : %r' % [float('%.1f' % i) for i in y])
            print('\nCHECKING UNIQUENESS...', end=' ')

        for i in range(N):
            for j in range(N):
                if j > i:
                    if P.shape[0] == F.shape[0]:
                        assert not np.equal(P[:, i], P[:, j]).all()
                        assert not np.equal(F[:, i], F[:, j]).all()
                        assert not np.equal(P[:, i], F[:, j]).all()
                        assert not np.equal(F[:, i], P[:, j]).all()

        if verbose:
            print('OK.')

        return (F, weights_f), (P, weights_p)


class DummyPartition(object):

    """Dummy class to generate tri."""

    def __init__(self, tri):
        self.tri = tri

    def partition(self, as_array=True):
        """Return the tri index."""
        if as_array:
            pass
        yield self.tri


###############################################################################
def layer_fit(layer, cache, F, wf):
    """Test the layer's fit method."""
    # Check predictions against ground truth
    preds = cache.layer_est(layer, 'fit')
    np.testing.assert_array_equal(preds, F)

    # Check coefficients
    d = layer.estimators_
    if layer.cls != 'blend':
        d = d[layer.n_pred:]

    ests = [(c, tup) for c, tup in d]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf

    assert preds.__class__.__name__ == 'ndarray'

    for i in layer.estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def layer_predict(layer, cache, P, wp):
    """Test the layer's predict method."""
    preds = cache.layer_est(layer, 'predict')
    np.testing.assert_array_equal(preds, P)

    # Check weights
    d = layer.estimators_
    ests = [(c, tup) for c, tup in d[:layer.n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wp


def layer_transform(layer, cache, F):
    """Test the layer's transform method."""
    # Check predictions against ground truth
    preds = cache.layer_est(layer, 'transform')

    # Check predictions against GT
    np.testing.assert_array_equal(preds, F)


def lc_fit(lc, X, y, F, wf):
    """Test the layer containers fit method."""
    out = lc.fit(X, y, return_preds=True)

    # Test preds
    np.testing.assert_array_equal(F, out[-1])

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    if lc.layers['layer-1'].cls != 'blend':
        d = d[lc.layers['layer-1'].n_pred:]

    ests = [(c, tup) for c, tup in d]
    w = [tup[1][1].coef_.tolist() for tup in ests]
    assert w == wf

    assert out[-1].__class__.__name__ == 'ndarray'

    for i in lc.layers['layer-1'].estimators_:
        assert i[1][1].coef_.__class__.__name__ == 'ndarray'


def lc_predict(lc, X, P, wp):
    """Test the layer containers predict method."""
    pred = lc.predict(X)

    # Test preds
    np.testing.assert_array_equal(P, pred)

    # Test coefs
    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d[:lc.layers['layer-1'].n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wp


def lc_transform(lc, X, F):
    """Test the layer containers transform method."""
    pred = lc.transform(X)
    np.testing.assert_array_equal(pred, F)


def lc_feature_prop(lc, X, y, F):
    """Test input feature propagation."""

    feature_prop = lc.layers["layer-1"].propagate_features
    n = lc.layers["layer-1"].n_feature_prop

    r = X.shape[0] - F.shape[0]

    preds = lc.fit(X, y, return_preds=True)[1]

    np.testing.assert_array_equal(X[r:, feature_prop],  preds[:, :n])
    np.testing.assert_array_equal(F, preds[:, n:])


def lc_from_file(lc, cache, X, y, F, wf, P, wp):
    """Fit a layer container from file path to numpy array."""
    X_path, y_path = cache.store_X_y(X, y)

    # TEST FIT
    out = lc.fit(X_path, y_path, return_preds=True)

    np.testing.assert_array_equal(F, out[-1])

    d = lc.layers['layer-1'].estimators_
    if lc.layers['layer-1'].cls != 'blend':
        d = d[lc.layers['layer-1'].n_pred:]

    ests = [(c, tup) for c, tup in d]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wf

    # TEST MMAP
    assert out[-1].__class__.__name__ == 'ndarray'
    for e in lc.layers['layer-1'].estimators_:
        assert e[1][1].coef_.__class__.__name__ == 'ndarray'

    # TEST PREDICT
    out = lc.predict(X_path)

    np.testing.assert_array_equal(P, out)

    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d[:lc.layers['layer-1'].n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wp


def lc_from_csv(lc, cache, X, y, F, wf, P, wp):
    """Fit a layer container from file path to csv."""
    X_path, y_path = cache.store_X_y(X, y, as_csv=True)

    # TEST FIT
    out = lc.fit(X_path, y_path, return_preds=True)

    np.testing.assert_array_equal(F, out[-1])

    d = lc.layers['layer-1'].estimators_
    if lc.layers['layer-1'].cls != 'blend':
        d = d[lc.layers['layer-1'].n_pred:]

    ests = [(c, tup) for c, tup in d]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wf

    # TEST MMAP
    assert out[-1].__class__.__name__ == 'ndarray'
    for e in lc.layers['layer-1'].estimators_:
        assert e[1][1].coef_.__class__.__name__ == 'ndarray'

    # TEST PREDICT
    out = lc.predict(X_path)

    np.testing.assert_array_equal(P, out)

    d = lc.layers['layer-1'].estimators_
    ests = [(c, tup) for c, tup in d[:lc.layers['layer-1'].n_pred]]
    w = [tup[1][1].coef_.tolist() for tup in ests]

    assert w == wp
