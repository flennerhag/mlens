"""ML-ENSEMBLE

Exception handling classes.
"""


class NotFittedError(ValueError, AttributeError):

    """Error class for an ensemble or estimator that is not fitted yet

    Raised when some method has been called that expects the instance to be
    fitted.
    """


class NotInitializedError(ValueError, AttributeError):

    """Error class for an instance that has not been properly initialized.

    Raised when required methods and attributes have not been initialized.
    """


class FitFailedWarning(RuntimeWarning):

    """Warning for a failed estimator 'fit' call."""


class ParameterChangeWarning(UserWarning):

    """Warning for different params in blueprint estimator and fitted copy.

    .. versionadded:: 0.2.2
    """


class LayerSpecificationError(TypeError, ValueError):

    """Error class for incorrectly specified layers."""


class LayerSpecificationWarning(UserWarning):

    """Warning class if layer has been specified in a dubious form.

    This warning is raised when the input does not look like expected, but
    is not fatal and a best guess of how to fix it will be made.
    """


class ParallelProcessingError(AttributeError, RuntimeError):

    """Error class for fatal errors related to :class:`ParallelProcessing`.

    Can be subclassed for more specific error classes.
    """


class ParallelProcessingWarning(UserWarning):

    """Warnings related to methods on :class:`ParallelProcessing`.

    Can be subclassed for more specific warning classes.
    """


class InputDataWarning(UserWarning):

    """Warning used to notify that an array does not behave as expected.

    Raised if data looks suspicious, but not outright fatal. Used sparingly,
    as it is often better to raise an error if input does not look like
    expected. Debugging corrupt data during parallel estimation is difficult
    and requires knowledge of backend operations.
    """


class MetricWarning(UserWarning):

    """Warning to notify that scoring do not behave as expected.

    Raised if scoring fails or if aggregating scores fails.
    """


###############################################################################
class DeprecationWarning(UserWarning):

    """Warning to notify the user of a deprecated feature"""


class EfficiencyWarning(UserWarning):

    """Warning used to notify the user of inefficient computation.

    This warning notifies the user that the efficiency may not be optimal due
    to some reason which may be included as a part of the warning message.
    This may be subclassed into a more specific Warning class.

    .. versionadded:: 0.18

    .. note:: imported from Scikit-learn for validation compatibility
    """


class NonBLASDotWarning(EfficiencyWarning):

    """Warning used when the dot operation does not use BLAS.

    FROM SCIKIT-LEARN

    This warning is used to notify the user that BLAS was not used for dot
    operation and hence the efficiency may be affected.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation, extends EfficiencyWarning.

    .. note:: imported from Scikit-learn for validation compatibility
    """


class DataConversionWarning(UserWarning):

    """Warning used to notify implicit data conversions happening in the code.

    This warning occurs when some input data needs to be converted or
    interpreted in a way that may not match the user's expectations.

    For example, this warning may occur when the user
        - passes an integer array to a function which expects float input and
          will convert the input
        - requests a non-copying operation, but a copy is required to meet the
          implementation's data-type expectations;
        - passes an input whose shape can be interpreted ambiguously.

    .. versionchanged:: 0.18
       Moved from sklearn.utils.validation.

    .. note:: imported from Scikit-learn for validation compatibility.
    """
