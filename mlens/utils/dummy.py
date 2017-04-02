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
import subprocess
from abc import abstractmethod

import numpy as np
import warnings
from joblib import Parallel, dump, load

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

    Parameters
    ----------
    offset : float (default = 0)
        scalar value to add to the coefficient vector after fitting.

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
        """Transform array by adjusting all elements with scale."""
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


ECM = [('ols-%i' % i, OLS(offset=i)) for i in range(16)]
ECM_PROBA = [('lr-%i' % i, LogisticRegression(offset=i)) for i in range(16)]


###############################################################################
# Data generation functions and Layer estimation wrappers

def get_layers(cls, proba, *args, **kwargs):
    """Standardized setup for unit testing of Layer and LayerContainer."""
    if not proba:
        layer = Layer(estimators=ESTIMATORS,
                      cls=cls,
                      indexer=INDEXERS[cls](*args, **kwargs),
                      preprocessing=PREPROCESSING)

        lc = LayerContainer().add(estimators=ESTIMATORS,
                                  cls=cls,
                                  indexer=INDEXERS[cls](*args, **kwargs),
                                  preprocessing=PREPROCESSING)

        lcm = LayerContainer().add(estimators=ECM, cls=cls)
    else:
        layer = Layer(estimators=ESTIMATORS_PROBA,
                      cls=cls,
                      proba=True,
                      indexer=INDEXERS[cls](*args, **kwargs),
                      preprocessing=PREPROCESSING)

        lc = LayerContainer().add(estimators=ESTIMATORS_PROBA,
                                  cls=cls,
                                  proba=True,
                                  indexer=INDEXERS[cls](*args, **kwargs),
                                  preprocessing=PREPROCESSING)

        lcm = LayerContainer().add(estimators=ECM_PROBA, proba=True, cls=cls)
    return layer, lc, lcm


def get_path():
    """Set up a temprorary folder in the current working directory."""
    path = os.path.join(os.getcwd(), 'tmp')
    try:
        shutil.rmtree(path)
    except:
        pass
    os.mkdir(path)
    return path


def data(shape, m):
    """Generate X and y data with X.

    Returns
    -------
    train : ndarray
        generated as a sequence of  reshaped to (LEN, WIDTH)

    labels : ndarray
        generated as a step-function with a step every MOD. As such,
        each prediction fold during cross-validation have a unique level value.
    """
    s = shape[0]
    w = shape[1]

    train = np.array(range(s * w), dtype='float').reshape((s, w))
    train += 1

    labels = np.zeros(train.shape[0])

    increment = 10
    for i in range(0, s, m):
        labels[i:i + m] += increment

        increment += 10

    return train, labels


def _store_X_y(dir, X, y):
    """Save X and y to file in temporary directory."""

    xf, yf = os.path.join(dir, 'X.npy'), os.path.join(dir, 'y.npy')
    np.save(xf, X)
    np.save(yf, y)

    return xf, yf


def destroy_temp_dir(dir):
    """Remove temporary directories created during tests."""
    try:
        shutil.rmtree(dir)
    except OSError:
        warnings.warn("Failed to destroy temporary test cache at %s" % dir)


def _folded_ests(X, y, n_ests, indexer, attr, labels=1, subsets=1,
                 verbose=True):
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

    ests = ESTIMATORS if labels == 1 else ESTIMATORS_PROBA
    prep = PREPROCESSING

    t = [t for _, t in indexer.generate(X, True)]
    t = np.unique(np.hstack(t))
    t.sort()

    weights = []
    F = np.zeros((len(t), n_ests * subsets * labels), dtype=np.float)

    col_id = {}
    col_ass = 0

    # Sort at every occasion
    for key in sorted(prep):
        for tri, tei in indexer.generate(X, True):
            # Sort again
            for est_name, est in ests[key]:

                if '%s-%s' % (key, est_name) not in col_id:
                    col_id['%s-%s' % (key, est_name)] = col_ass
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
                    F[fix, col_id['%s-%s' % (key, est_name)]] = p
                else:
                    c = col_id['%s-%s' % (key, est_name)]
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
                except:
                    pass

    return F, weights


def _full_ests(X, y, n_ests, indexer, attr, labels=1, subsets=1, verbose=True):
    """Get ground truth for train and predict on full data."""
    if verbose:
        print('\n                        FULL PREDICTION OUTPUT')
        print('-' * 100)
        print('  EST   |'
              '             GROUND TRUTH             |'
              '    COEF     |'
              '           PRED')

    ests = ESTIMATORS if labels == 1 else ESTIMATORS_PROBA
    prep = PREPROCESSING

    tri = [t for t, _ in indexer.generate(X, True)]
    tri = np.unique(np.hstack(tri))

    P = np.zeros((X.shape[0], n_ests * subsets * labels), dtype=np.float)
    weights = list()
    col_id = {}
    col_ass = 0

    # Sort at every occasion
    for key in sorted(prep):
        for est_name, est in ests[key]:

            if '%s-%s' % (key, est_name) not in col_id:
                col_id['%s-%s' % (key, est_name)] = col_ass
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
            c = col_id['%s-%s' % (key, est_name)]
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
            except:
                pass

    return P, weights


def ground_truth(X, y, indexer, attr, labels, subsets=1, verbose=True):
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
        Raises assertion error if any weight vectors overlap or any predictions
        (as measured by columns in F and P) are judged to be equal.

    Examples
    --------
    >>> from mlens.utils.dummy import ground_truth, data
    >>> from mlens.base.indexer import FoldIndex
    >>> X, y = data((4, 1), 2)
    >>> indexer = FoldIndex(X=X)
    >>> (F, wf), (P, wp) = ground_truth(X, y, indexer, 'predict', 1)
                                CONSTRUCTING GROUND TRUTH
                                        FOLD OUTPUT
    --------------------------------------------------------------------------------
      EST   |  TRI   |  TEI   | TEST LABELS  | TRAIN LABELS | COEF   | PRED
    no-offs | [2, 3] | [0, 1] | [10.0, 10.0] | [20.0, 20.0] |  [7.6] | [7.6,
    15.2]
    no-null | [2, 3] | [0, 1] | [10.0, 10.0] | [20.0, 20.0] |  [5.6] | [5.6,
    11.2]
    no-offs | [0, 1] | [2, 3] | [20.0, 20.0] | [10.0, 10.0] |  [8.0] | [
    24.0, 32.0]
    no-null | [0, 1] | [2, 3] | [20.0, 20.0] | [10.0, 10.0] |  [6.0] | [
    18.0, 24.0]
    sc-offs | [2, 3] | [0, 1] | [10.0, 10.0] | [20.0, 20.0] |  [2.0] | [
    -5.0, -3.0]
    sc-offs | [0, 1] | [2, 3] | [20.0, 20.0] | [10.0, 10.0] |  [2.0] | [3.0,
    5.0]
                            FULL PREDICTION OUTPUT
    --------------------------------------------------------------------
      EST   |       GROUND TRUTH       | COEF  |           PRED
    no-offs | [10.0, 10.0, 20.0, 20.0] | [7.7] | [7.7, 15.3, 23.0, 30.7]
    no-null | [10.0, 10.0, 20.0, 20.0] | [5.7] | [5.7, 11.3, 17.0, 22.7]
    sc-offs | [10.0, 10.0, 20.0, 20.0] | [6.0] | [-9.0, -3.0, 3.0, 9.0]
                     SUMMARY
    ------------------------------------------
    no-null |   FULL: [7.7, 15.3, 23.0, 30.7]
    no-null |  FOLDS: [7.6, 15.2, 24.0, 32.0]
    no-offs |   FULL: [7.7, 15.3, 23.0, 30.7]
    no-offs |  FOLDS: [7.6, 15.2, 24.0, 32.0]
    sc-offs |   FULL: [5.7, 11.3, 17.0, 22.7]
    sc-offs |  FOLDS: [5.6, 11.2, 18.0, 24.0]
    GT              : [10.0, 10.0, 20.0, 20.0]
    CHECKING UNIQUENESS... OK.
    """
    if verbose:
        print('                            CONSTRUCTING GROUND TRUTH\n')

    # Build predictions matrices.
    N = 0
    for case in ESTIMATORS:
        N += len(ESTIMATORS[case])

    F, weights_f = _folded_ests(X, y, N, indexer, attr, labels, subsets, verbose)
    P, weights_p = _full_ests(X, y, N, indexer, attr, labels, subsets, verbose)

    if verbose:
        print('\n                 SUMMARY')
        print('-' * 42)
    col = 0
    for key in sorted(ESTIMATORS):
        for est_name, est in ESTIMATORS[key]:
            if verbose:
                print('%s | %6s: %20r' % ('%s-%s' % (key, est_name), 'FULL',
                                          [float('%.1f' % i) for i in P[:, col]]))
                print('%s | %6s: %20r' % ('%s-%s' % (key, est_name), 'FOLDS',
                                          [float('%.1f' % i) for i in F[:, col]]))
            col += 1

    if verbose:
        print('GT              : %r' % [float('%.1f' % i) for i in y])

        print('\nCHECKING UNIQUENESS...', end=' ')

    # First, assert folded preds differ from full preds:
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


def _init(train, label, shape):
    """Simple temp folder initialization for testing estimation functions."""

    dir = os.path.join(os.getcwd(), 'tmp')

    if not os.path.exists(dir):
        os.mkdir(dir)

    paths = {}
    for name, arr in zip(('X', 'y'), (train, label)):
        f = os.path.join(dir, '%s.mmap' % name)
        paths[name] = f
        if os.path.exists(f):
            os.unlink(f)
        dump(arr, f)

    X = load(paths['X'], mmap_mode='r')
    y = load(paths['y'], mmap_mode='r')

    p = os.path.join(dir, 'P.mmap')
    if os.path.exists(p):
        os.unlink(p)

    paths['P'] = p
    P = np.memmap(paths['P'], dtype=np.float, shape=shape, mode='w+')

    return {'X': X, 'y': y, 'P': P, 'dir': dir}


def _layer_est(layer, attr, train, label, n_jobs, rem=True, args=None):
    """Test the estimation routine for a layer."""

    est = ENGINES[layer.cls]


    # Create a cache
    if 'fit' in attr:
        n = layer.indexer.n_test_samples
    else:
        n = layer.indexer.n_samples

    s1 = layer.n_pred

    if layer.proba:
        if 'fit' in attr:
            layer.classes_ = np.unique(label).shape[0]

        s1 *= layer.classes_

    job = _init(train, label, (n, s1))

    try:
        # Wrap in try-except to always close the tmp if asked to
        with Parallel(n_jobs=n_jobs,
                      temp_folder=job['dir'],
                      mmap_mode='r+',
                      max_nbytes=None) as parallel:

            # Run test
            if args is None:
                kwargs = job
            else:
                kwargs = {arg: job[arg] for arg in args}

            kwargs['parallel'] = parallel

            e = est(layer=layer)
            getattr(e, attr)(**kwargs)

        # Check prediction output
        P = job['P']
        P.flush()
        preds = np.asarray(P)

    except Exception as e:
        raise RuntimeError("Could not estimate layer:\n%r" % e)

    finally:
        # Always remove tmp if asked
        if rem:
            f = job['dir']
            job.clear()
            gc.collect()
            try:
                shutil.rmtree(f)
            except OSError:
                try:
                    dlc = subprocess.Popen('rmdir /S /Q %s' % f,
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                except OSError:
                    warnings.warn("Could not close temp dir %s." % f)

    return preds
