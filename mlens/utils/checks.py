"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:licence: MIT

Controls that an estimator was built as expected.
"""

import warnings
from .exceptions import (NotFittedError, LayerSpecificationWarning,
                         LayerSpecificationError, ParallelProcessingError,
                         ParallelProcessingWarning)


def check_ensemble_build(inst, attr='stack'):
    """Check that layers have been instantiated."""
    if not hasattr(inst, attr):
        # No layer container. This should not happen!

        msg = ("No layer class attached to instance (%s). (Cannot find a "
               "'Sequential' class instance as attribute [%s].)")

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


def assert_valid_estimator(instance):
    """Assert that an instance has a ``get_params`` and ``fit`` method."""
    has_get_params = hasattr(instance, 'get_params')
    has_fit = hasattr(instance, 'fit')

    if not has_get_params:
        raise TypeError("'%s' does not appear to be a valid"
                        " estimator as it does not implement a "
                        "'get_params' method. Type: "
                        "%s" % (instance, type(instance)))

    if not has_fit:
        raise TypeError("'%s' does not appear to be a valid"
                        " estimator as it does not implement a "
                        "'fit' method. Type: "
                        "%s" % (instance, type(instance)))

    try:
        instance.get_params()
    except TypeError:
        raise TypeError("'%s' does not appear to be an instance of an "
                        "estimator class, but the class itself." % instance)


def assert_valid_pipeline(pipeline):
    """Quick check to ensure the pipeline is an mlens Pipeline"""
    cls = str(pipeline.__class__).split("'")[1].lower()
    if not cls.endswith('pipeline') and not cls.startswith('mlens'):
        raise ValueError("Expect mlens Pipeline instance. Got %r" % pipeline)


def assert_correct_format(estimators, preprocessing):
    """Initial check to assert layer can be constructed."""
    if (preprocessing is None) or (not isinstance(preprocessing, dict)):
        if isinstance(estimators, dict):
            # Either no or uniform preprocessing, estimators should be list
            msg = ("Preprocessing is either 'None' or 'list': 'estimators' "
                   "must be of type 'list'. Got %s.")
            raise LayerSpecificationError(msg % type(estimators))
    else:
        # Check that both estimators and preprocessing are dicts
        if not isinstance(preprocessing, dict):
            msg = ("Unexpected format of 'preprocessing'. Needs to be "
                   " of type 'None' or 'list' or 'dict'. Got %s .")
            raise LayerSpecificationError(msg % type(preprocessing))

        if not isinstance(estimators, dict):
            msg = ("Unexpected format of 'estimators'. Needs to be "
                   "'dict' when preprocessing is 'dict'. Got %s.")
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


def check_initialized(inst):
    """Check if a ParallelProcessing instance is initialized properly."""
    if not inst.__initialized__:
        msg = "ParallelProcessing is not initialized. Call " \
              "'initialize' before calling 'fit'."
        raise ParallelProcessingError(msg)

    if getattr(inst, '__fitted__', None):
        if inst.layers.raise_on_exception:
            raise ParallelProcessingError("This instance is already "
                                          "fitted and its parallel "
                                          "processing jobs has not been "
                                          "terminated. To refit "
                                          "instance, call 'terminate' "
                                          "before calling 'fit'.")
        else:
            warnings.warn("This instance is already "
                          "fitted and its parallel "
                          "processing job has not been "
                          "terminated. Will refit using previous job's cache.",
                          ParallelProcessingWarning)
