"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Checks to deploy on estimators to assert proper behavior.
"""

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_X_y
from mlens.ensemble import StackingEnsemble
from sklearn.utils.estimator_checks import check_estimator

from abc import abstractmethod

CLASS = StackingEnsemble


class AverageRegressor(BaseEstimator):
    """Predicts the average of training labels.

    MWP of a Scikit-learn estimator, to be used for unit-tests of ensemble
    classes.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """Find average."""
        X, y = check_X_y(X, y, accept_sparse='csr')
        self.average_ = np.mean(y)
        return self

    def predict(self, X, y=None):
        """Predict the average."""
        X = check_array(X, accept_sparse='csr')

        return np.ones(X.shape[0]) * self.average_


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
    >>> from sklearn.utils.estimator_checks import check_estimator
    >>> from mlens.ensemble import StackingEnsemble
    >>> from mlens.utils.estimator_checks import InitMixin
    >>>
    >>> class TestStackingEnsemble(InitMixin, StackingEnsemble):
    ...
    ...     def __init__(self):
    ...         super().__init__()
    >>>
    >>> check_estimator(TestStackingEnsemble)
    [Some warning messages from mlens ONLY (not sklearn warnings)]
    """
    @abstractmethod
    def __init__(self):

        # Instantiate class
        super(InitMixin, self).__init__()

        self.n_jobs = 1

        if getattr(self, 'layers', None) is None:
            self.add([AverageRegressor()])

        if hasattr(self, 'meta_estimator') and self.meta_estimator is None:
            self.add_meta(AverageRegressor())
