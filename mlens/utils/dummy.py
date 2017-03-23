"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Collection of dummy estimator classes, Mixins to build transparent layers for
unit testing.

Also contains pre-made Layer, LayerContainers and data generation functions
for unit testing.
"""

import numpy as np
import itertools

from .exceptions import NotFittedError
from ..ensemble.base import LayerContainer, Layer
from ..externals.base import BaseEstimator, TransformerMixin, clone
from ..externals.validation import check_array, check_X_y
from ..parallel.manager import ENGINES

from joblib import Parallel, load, dump
import shutil
import os
import gc

from abc import abstractmethod


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

    def predict(self, X, y=None):
        """Predict with fitted weights."""
        X = check_array(X, accept_sparse=False)

        return np.dot(X, self.coef_.T)


class Scale(BaseEstimator, TransformerMixin):
    """Removes the learnt mean column-wise in an array.

    MWE of a Scikit-learn transformer, to be used for unit-tests of ensemble
    classes.

    Parameters
    ----------
    copy : bool (default = True)
        Whether to copy X before transforming.

    Examples
    --------

    Scaling elements

    >>> from numpy import arange
    >>> from mlens.utils.dummy import Scale
    >>> X = arange(6).reshape(3, 2)
    >>> X[:, 1] *= 2
    >>> print('X:')
    >>> print('%r' % X)
    >>> print()
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

    Asserting :class:`Scale` passes the Scikit-learn estimator test

    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from mlens.utils.dummy import Scale
    >>> check_estimator(Scale)
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
            getattr(self, 'add')(OLS())

###############################################################################
# Pre-made Layer and LayerContainer classes

PREPROCESSING = {'no': [], 'sc': [Scale()]}

ESTIMATORS = {'sc': [('offs', OLS(offset=2))],
              'no': [('offs', OLS(offset=2)), ('null', OLS())]}

ECM = [OLS(offset=i) for i in range(16)]

LAYERS = {('layer', 'stack'):
          Layer(estimators=ESTIMATORS, cls='stack',
                preprocessing=PREPROCESSING),
          ('layer', 'blend'):
          Layer(estimators=ESTIMATORS, cls='blend',
                preprocessing=PREPROCESSING),
          ('lc', 'stack'):
          LayerContainer().add(estimators=ESTIMATORS, cls='stack',
                               preprocessing=PREPROCESSING),
          ('lc', 'blend'):
              LayerContainer().add(estimators=ESTIMATORS, cls='blend',
                                   preprocessing=PREPROCESSING),
          ('lcm', 'stack'):
          LayerContainer().add(estimators=ECM, cls='stack'),
          ('lcm', 'blend'):
          LayerContainer().add(estimators=ECM, cls='blend')
          }

###############################################################################
# Data generation functions and Layer estimation wrappers


def get_path():
    """Set up a temprorary folder in the current working directory."""
    path = os.path.join(os.getcwd(), 'tmp')
    try:
        shutil.rmtree(path)
    except:
        pass
    os.mkdir(path)
    return path


def data(shape, mod):
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
    for i in range(0, s, mod):
        labels[i:i + mod] += increment

        increment += 10

    return train, labels


def _store_X_y(dir, X, y):
    """Save X and y to file in temporary directory."""

    xf, yf = os.path.join(dir, 'X.npy'), os.path.join(dir, 'y.npy')
    np.save(xf, X)
    np.save(yf, y)

    return xf, yf


def _destroy_temp_dir(dir):
    """Remove temporary directories created during tests."""
    shutil.rmtree(dir)


def _folded_ests(X, y, n_ests, indexer):
    """Build ground truth for each fold."""
    print('                                    FOLD OUTPUT')
    print('-' * 80)
    print('  EST   |'
          '  TRI   |'
          '  TEI   |'
          ' TEST LABELS  |'
          ' TRAIN LABELS |'
          ' COEF   |'
          ' PRED')

    t = [t for _, t in indexer.generate(X, True)]
    t = np.hstack(t)
    t.sort()

    weights = []
    F = np.zeros((len(t), n_ests))

    col_id = {}
    col_ass = 0

    # Sort at every occasion
    for key in sorted(PREPROCESSING):
        for tri, tei in indexer.generate(X, True):
            # Sort again
            for est_name, est in ESTIMATORS[key]:

                if '%s-%s' % (key, est_name) not in col_id:
                    col_id['%s-%s' % (key, est_name)] = col_ass
                    col_ass += 1

                xtrain = X[tri]
                xtest = X[tei]

                # Transform inputs
                for tr in PREPROCESSING[key]:
                    t = clone(tr)
                    xtrain = t.fit_transform(xtrain)
                    xtest = t.transform(xtest)

                # Fit estimator
                e = clone(est).fit(xtrain, y[tri])
                w = e.coef_
                weights.append(w.tolist())

                # Get out-of-sample predictions
                p = e.predict(xtest)
                F[tei, col_id['%s-%s' % (key, est_name)]] = p

                print('%s | %r | %r | %r | %r | %6r | %r' % (
                    '%s-%s' % (key, est_name),
                    list(tri),
                    list(tei),
                    [float('%.1f' % i) for i in y[tei]],
                    [float('%.1f' % i) for i in y[tri]],
                    [float('%.1f' % i) for i in w],
                    [float('%.1f' % i) for i in p]))
    return F, weights


def _full_ests(X, y, n_ests, indexer):
    """Get ground truth for train and predict on full data."""
    print('\n                        FULL PREDICTION OUTPUT')
    print('-' * 68)
    print('  EST   |'
          '       GROUND TRUTH       |'
          ' COEF  |'
          '           PRED')

    t = [t for _, t in indexer.generate(X, True)]
    t = np.hstack(t)
    t.sort()

    P = np.zeros((X.shape[0], n_ests))
    weights = list()
    col_id = {}
    col_ass = 0

    # Sort at every occasion
    for key in sorted(PREPROCESSING):
        for est_name, est in ESTIMATORS[key]:

            if '%s-%s' % (key, est_name) not in col_id:
                col_id['%s-%s' % (key, est_name)] = col_ass
                col_ass += 1

            # Transform input
            xtrain = X[t]
            ytrain = y[t]
            xtest = X
            for tr in PREPROCESSING[key]:
                t = clone(tr)
                xtrain = t.fit_transform(xtrain)
                xtest = t.transform(xtest)

            # Fit est
            e = clone(est).fit(xtrain, ytrain)
            w = e.coef_
            weights.append(w.tolist())

            # Predict
            p = e.predict(xtest)
            P[:, col_id['%s-%s' % (key, est_name)]] = p

            print('%s | %r | %r | %r' % (
                '%s-%s' % (key, est_name),
                [float('%.1f' % i) for i in y],
                [float('%.1f' % i) for i in w],
                [float('%.1f' % i) for i in p]))

    return P, weights


def ground_truth(X, y, n_ests, indexer):
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
    >>>ground_truth()
                                CONSTRUCTING GROUND TRUTH

                                        FOLD OUTPUT
    -------------------------------------------------------------------------..
      EST   |  TRI   |  TEI   | TEST LABELS  | TRAIN LABELS | COEF   | PRED
    no-null | [2, 3] | [0, 1] | [10.0, 10.0] | [20.0, 20.0] |  [5.6] | [5.6, ..
    no-null | [0, 1] | [2, 3] | [20.0, 20.0] | [10.0, 10.0] |  [6.0] | [18.0,..
    sc-offs | [2, 3] | [0, 1] | [10.0, 10.0] | [20.0, 20.0] |  [2.0] | [-5.0,..
    sc-offs | [0, 1] | [2, 3] | [20.0, 20.0] | [10.0, 10.0] |  [2.0] | [3.0, ..

                            FULL PREDICTION OUTPUT
    --------------------------------------------------------------------
      EST   |       GROUND TRUTH       | COEF  |           PRED
    no-null | [10.0, 10.0, 20.0, 20.0] | [5.7] | [5.7, 11.3, 17.0, 22.7]
    sc-offs | [10.0, 10.0, 20.0, 20.0] | [6.0] | [-9.0, -3.0, 3.0, 9.0]

                     SUMMARY
    ------------------------------------------
    no-null |   FULL: [5.7, 11.3, 17.0, 22.7]
    no-null |  FOLDS: [5.6, 11.2, 18.0, 24.0]
    sc-offs |   FULL: [-9.0, -3.0, 3.0, 9.0]
    sc-offs |  FOLDS: [-5.0, -3.0, 3.0, 5.0]
    GT              : [10.0, 10.0, 20.0, 20.0]

    CHECKING UNIQUENESS... OK.
    """
    print('                            CONSTRUCTING GROUND TRUTH\n')

    # First build folded estimations.
    F, weights_f = _folded_ests(X, y, n_ests, indexer)
    P, weights_p = _full_ests(X, y, n_ests, indexer)

    print('\n                 SUMMARY')
    print('-' * 42)
    col = 0
    for key in sorted(ESTIMATORS):
        for est_name, est in sorted(ESTIMATORS[key]):
            print('%s | %6s: %20r' % ('%s-%s' % (key, est_name), 'FULL',
                                      [float('%.1f' % i) for i in P[:, col]]))
            print('%s | %6s: %20r' % ('%s-%s' % (key, est_name), 'FOLDS',
                                      [float('%.1f' % i) for i in F[:, col]]))
        col += 1

    print('GT              : %r' % [float('%.1f' % i) for i in y])

    print('\nCHECKING UNIQUENESS...', end=' ')

    try:
        # First, assert folded preds differ from full preds:
        for i in range(n_ests):
            for j in range(n_ests):
                if j > i:
                    assert not np.equal(P[:, i], P[:, j]).any()
                    assert not np.equal(P[:, i], F[:, j]).any()
                    assert not np.equal(F[:, i], P[:, j]).any()
                    for _, t in indexer.generate(X, as_array=True):
                        assert not np.equal(F[t, i], F[t, j]).any()

        # Second, assert all combinations of weights are not equal
        for weights in [weights_f, weights_p]:
            for a, b in itertools.combinations(weights, 2):
                assert not np.equal(a, b).any()
        print('OK.')

    except AssertionError as exc:
        print('ERROR:')
        raise AssertionError(exc)
    finally:
        return (F, weights_f), (P, weights_p)


def _init(train, label, shape):
    """Simple temp folder initialization for testing estimation functions."""

    dir = os.path.join(os.getcwd(), 'tmp/tmp')
    try:
        shutil.rmtree(dir)
    except:
        pass
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

    paths['P'] = os.path.join(dir, 'P.mmap')
    P = np.memmap(paths['P'], dtype=y.dtype, shape=shape, mode='w+')

    return {'X': X, 'y': y, 'P': P, 'dir': dir}


def _layer_est(layer, attr, train, label, n_jobs, rem=True, args=None):
    """Test the estimation routine for a layer."""

    est = ENGINES[layer.cls]

    job = None
    try:
        # Wrap in try-except to always close the tmp if asked to

        # Create a cache
        if attr == 'fit':
            n = layer.indexer.n_test_samples
        else:
            n = layer.indexer.n_samples

        job = _init(train, label,
                    (n, layer.n_pred))

        # Get a parallel jobs up
        with Parallel(n_jobs=n_jobs,
                      temp_folder=job['dir'],
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
        preds = np.asarray(job['P'])

    finally:
        # Always remove tmp if asked
        if rem and job is not None:
            f = job['dir']
            job.clear()
            shutil.rmtree(f)
            gc.collect()

    return preds
