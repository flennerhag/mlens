"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Collection of dummy estimator classes, Mixins to build transparent layers for
unit testing.

Also contains pre-made Layer, LayerContainers and data generation functions
for unit testing.
"""

from __future__ import division, print_function


import numpy as np

from .exceptions import NotFittedError
from ..externals.sklearn.base import BaseEstimator, TransformerMixin, clone
from ..externals.sklearn.validation import check_X_y, check_array


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
            models.append(OLS().fit(X, labels))

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

            p = 1 / (1 + np.exp(- m.predict(X)))

            preds.append(p)

        return np.vstack(preds).T

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
        X = check_array(X, accept_sparse=False)
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
        X = check_array(X, accept_sparse=False)
        Xt = X.copy() if self.copy else X
        return Xt - self.mean_
