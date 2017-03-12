"""ML-ENSEMBLE

author: Sebastian Flennerhag
licence: MIT
Quick checks that an estimator is built as expected.
"""

import warnings
from .exceptions import (NotFittedError, LayerSpecificationWarning,
                         LayerSpecificationError, FitFailedError,
                         FitFailedWarning)


def _non_null_return(*args):
    """Utility for returning non-null values."""
    return tuple([arg for arg in args if arg is not None])


def check_layer_output(layer, layer_name, raise_on_exception):
    """Quick check to determine if no estimators where fitted."""
    if not hasattr(layer, 'estimators_'):
        # If the attribute was not created during fit, the instance will not
        # function. Raise error.
        raise FitFailedError("[%s] Fit failed. The 'fit_function' did "
                             "not return expected output: the layer [%s] is "
                             "missing the 'estimators_' attribute with "
                             "fitted estimators." % (layer_name, layer_name))

    ests = layer.estimators_
    if ests is None or len(ests) == 0:
        msg = "[%s] No estimators in layer was fitted."
        if raise_on_exception:
            raise FitFailedError(msg % layer_name)
        warnings.warn(msg % layer_name, FitFailedWarning)


def check_is_fitted(estimator, attr):
    """Check that ensemble has been fitted.

    Parameters
    ----------
    estimator : estimator instance
        ensemble instance to check.

    attr : str
        attribute to assert existence of. Default is the 'layer_' attribute
        that holds fitted layers.
    """
    msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
           "appropriate arguments before using this method.")
    if not hasattr(estimator, attr):
        raise NotFittedError(msg % {"name": type(estimator).__name__})


def check_ensemble_build(inst, attr='layers'):
    """Check that layers have been instantiated."""

    if not hasattr(inst, attr):
        # No layer container. This should not happen!

        msg = ("No layer class attached to instance (%s). (Cannot find a "
               "'LayerContainer' class instance as attribute [%s].)")

        raise AttributeError(msg % (inst.__class__.__name__, attr))

    if getattr(inst, attr) is None:
        # No layers instantiated.

        if not getattr(inst, 'raise_on_exception', True):

            msg = "No Layers in instance (%s). Nothing to fit / predict."
            warnings.warn(msg % inst.__class__.__name__,
                          LayerSpecificationWarning)
            # For PASSED flag on soft fail
            return False

        else:

            msg = ("No Layers in instance (%s). Add layers before calling "
                   "'fit' and 'predict'.")
            raise LayerSpecificationError(msg % inst.__class__.__name__)

    # For PASSED flag
    return True


def assert_correct_layer_format(estimators, preprocessing):
    """Initial check to assert layer can be constructed."""
    if (preprocessing is None) or (isinstance(preprocessing, list)):
        # Either no preprocessing or uniform preprocessing
        if not isinstance(estimators, list):
            msg = ("Preprocessing is either 'None' or 'list': 'estimators' "
                   "must be of type 'list' (%s type passed).")
            raise LayerSpecificationError(msg % type(estimators))

    else:
        # Check that both estimators and preprocessing are dicts
        if not isinstance(preprocessing, dict):
            msg = ("Unexpected format of 'preprocessing'. Needs to be "
                   " of type 'None' or 'list' or 'dict'  (%s type passed).")
            raise LayerSpecificationError(msg % type(preprocessing))

        if not isinstance(estimators, dict):
            msg = ("Unexpected format of 'estimators'. Needs to be "
                   "'dict' when preprocessing is 'dict' (%s type passed).")
            raise LayerSpecificationError(msg % type(estimators))

        # Check that keys overlap
        prep_check = [key in preprocessing for key in estimators]
        est_check = [key in estimators for key in preprocessing]

        if not all(est_check):
            msg = ("Too few estimator cases to match preprocessing cases:\n"
                   "estimator cases:     %r\npreprocessing cases: %r")
            raise LayerSpecificationError(msg % (list(estimators),
                                                 list(preprocessing)))
        if not all(prep_check):
            msg = ("Too few preprocessing cases to match estimators cases:\n"
                   "preprocessing cases: %r\nestimator cases:     %r")
            raise LayerSpecificationError(msg % (list(preprocessing),
                                                 list(estimators)))
