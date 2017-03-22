"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Dummy estimators and Mixins to facilitate unit testing and some benchmarking.
"""

import numpy as np

from .exceptions import NotFittedError

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y

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
        """
        X = check_array(X, accept_sparse='csr')
        self.__is_fitted__ = True
        self.mean_ = X.mean(axis = 0)
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
    ...         super().__init__()
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
