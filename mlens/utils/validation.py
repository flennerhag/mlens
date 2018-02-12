"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Input validation module. Builds on Scikit-learns ``validation`` module, but
extends it to a soft check that issues warnings but don't force changes to the
inputs.
"""

import warnings

import numpy as np
import scipy.sparse as sp

from ..externals import six
from mlens.externals.sklearn.validation import check_X_y, _num_samples, \
    _shape_repr, check_array, check_consistent_length
from ..utils.exceptions import InputDataWarning, NonBLASDotWarning

FLOAT_DTYPES = (np.float64, np.float32, np.float16)

# Silenced by default to reduce verbosity. Turn on at runtime for
# performance profiling.
warnings.simplefilter('ignore', NonBLASDotWarning)


def _get_context(estimator=None):
    """Get context name for warning messages."""
    if estimator is not None:
        if isinstance(estimator, six.string_types):
            estimator_name = estimator.lower()
        else:
            estimator_name = estimator.__class__.__name__.lower()

        estimator_name = "[%s] " % estimator_name
    else:
        estimator_name = ""

    return estimator_name


def soft_check_array(array, accept_sparse=True, dtype=None,
                     ensure_2d=True, force_all_finite=True, allow_nd=True,
                     ensure_min_samples=1, ensure_min_features=1,
                     estimator=None):
    """Input validation on an array, list, sparse matrix or similar.

    Like Scikit-learn's ``check_array`` , but issues warnings on failed tests
    and do no forced array conversion.

    Parameters
    ----------
    array : array-like
        Input object, expected to be array-like, to check / convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", warning is raised if array.dtype is object.
        If dtype is a list of types, warning is raised if array.dtype is not
        a member of the list.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    ensure_2d : boolean (default=True)
        Whether to warn if X is not at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    ensure_min_samples : int (default=1)
        Make sure that the array has a minimum number of samples in its first
        axis (rows for a 2D array). Setting to 0 disables this check.

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when the input data has effectively 2
        dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
        disables this check.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    CHANGE : bool
        Whether X should be changed.
    """
    # Set initial change flag to False. Will be set to True if any test fails.
    CHANGE = False

    context = _get_context(estimator)

    # ---- Check dtype -----

    # store whether originally we wanted numeric dtype
    dtype_numeric = dtype == "numeric"

    # Get input array's dtype
    dtype_orig = getattr(array, "dtype", None)

    if not hasattr(dtype_orig, 'kind'):
        # not a data type (e.g. a column named dtype in a pandas DataFrame)
        dtype_orig = None

    if dtype_numeric:
        # We want to check that the dtype is numeric.
        if dtype_orig is not None and dtype_orig.kind == "O":
            dtype = np.float64
        else:
            dtype = None

    wrong_dtype = False
    if dtype is not None:
        if isinstance(dtype, (list, tuple)):
            wrong_dtype = dtype_orig is not None and dtype_orig not in dtype
        else:
            wrong_dtype = dtype_orig is not None and dtype_orig != dtype

    if wrong_dtype:
        CHANGE = True
        msg = ("%sDtype of input array not the expected type [dtype: %s]. "
               "Consider changing to %r")
        warnings.warn(msg % (context, dtype_orig, dtype), InputDataWarning)

    # ----- check array shape ------
    if isinstance(accept_sparse, str):
        accept_sparse = [accept_sparse]

    if sp.issparse(array):
        CHANGE = _check_sparse_format(array, accept_sparse, dtype,
                                      force_all_finite, context)
    else:
        # Check if X is 2d
        if ensure_2d:
            if array.ndim == 1:
                if (ensure_min_samples >= 2) and (len(array) == 1):
                    # Raise error if we want X to be 2d, but only have one obs
                    raise ValueError("%sexpected at least 2 samples provided "
                                     "in a 2 dimensional array-like input"
                                     % context)
                # Else,flag for bad formatting
                CHANGE = True
                msg = ("%sX is one-dimensional. Reshape your data either "
                       "using X.reshape(-1, 1) if your data has a single"
                       "feature or X.reshape(1, -1) if it contains a single "
                       "sample.")
                warnings.warn(msg % context, InputDataWarning)

        # Check for number of dimensions
        if not allow_nd and array.ndim >= 3:
            warnings.warn("%sFound array with dim %d. %s expected <= 2." % (
                          context, array.ndim, context), InputDataWarning)

        # Check for finite inputs
        if force_all_finite:
            ALL_FINITE = _check_all_finite(array)

            if not ALL_FINITE:
                CHANGE = True
                msg = ("%sNot all elements in array are finite. This may "
                       "cause estimation problems. Consider nan conversion "
                       "and replacing infinite values.")
                warnings.warn(msg % context, InputDataWarning)

    # Check shape
    try:
        shape_repr = _shape_repr(array.shape)
    except Exception as e:
        CHANGE = True
        warnings.warn("%sCannot infer shape of input data: may not be "
                      "a suitable data type for estimation. Will proceed "
                      "without checking dimensionality. "
                      "Details:\n%r" % (context, e), InputDataWarning)
        shape_repr = 'NaN'

    if ensure_min_samples > 0:
        try:
            n_samples = _num_samples(array)
        except Exception as e:
            CHANGE = True
            warnings.warn("%sCannot infer samples size of input data: may not "
                          "be a suitable data type for estimation."
                          "Will proceed without checking sample size. "
                          "Details:\n%r" % (context, e), InputDataWarning)
            n_samples = np.inf

        if n_samples < ensure_min_samples:
            CHANGE = True
            msg = ("%sFound array with %d sample(s) (shape=%s) "
                   "while a minimum of %d is required.")
            warnings.warn(msg % (context, n_samples, shape_repr,
                                 ensure_min_samples), InputDataWarning)

    if ensure_min_features > 0 and array.ndim == 2:
        try:
            n_features = array.shape[1]
        except Exception as e:
            CHANGE = True
            warnings.warn("%sCannot infer feature size of input data: may not "
                          "be a suitable data type for estimation."
                          "Will proceed without checking feature size. "
                          "Details:\n%r" % (context, e), InputDataWarning)
            n_features = np.inf

        if n_features < ensure_min_features:
            CHANGE = True
            msg = ("%sFound array with %d feature(s) (shape=%s) while "
                   " a minimum of %d is required.")
            warnings.warn(msg % (context, n_features, shape_repr,
                                 ensure_min_features), InputDataWarning)

    if CHANGE:
        warnings.warn("%sInput data failed initial test. Estimation may fail. "
                      "Consider converting input data to a numpy array with "
                      "finite elements and no missing values." % context,
                      InputDataWarning)

    return CHANGE


def _check_all_finite(X):
    """General check for all finite values in X."""
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    try:
        if (X.dtype.char in np.typecodes['AllFloat'] and not
                np.isfinite(X.sum()) and not np.isfinite(X).all()):
            return False
        else:
            return True

    except Exception as e:
        warnings.warn('Could not check array for all finite. Ensure X is an'
                      'array type, and consider converting to an ndarray or'
                      'scipy sparse array. Details:\n%r' % e, InputDataWarning)


def check_all_finite(X):
    """Return False if X contains NaN or infinity."""
    return _check_all_finite(X.data if sp.issparse(X) else X)


def _check_sparse_format(spmatrix, accept_sparse=True, dtype=None,
                         force_all_finite=True, context=""):
    """Check if a sparse array needs format changes.

    Checks the sparse format of spmatrix and alerts if changes are
    recommended. Like Scikit-learn's ``_assert_sparse_format`` but without
    forced conversion.

    Parameters
    ----------
    spmatrix : scipy sparse matrix
        Input to validate and convert.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats ('csc',
        'csr', 'coo', 'dok', 'bsr', 'lil', 'dia'). None means that sparse
        matrix input will raise an error.  If the input is sparse but not in
        the allowed format, it will be converted to the first listed format.

    dtype : string, type or None (default=none)
        Data type of result. If None, the dtype of the input is preserved.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X.

    context: str
        contextual message to begin warnings with.

    Returns
    -------
    CHANGE : bool
        False if no change is required, True if change is required
    """
    if accept_sparse in [None, False]:
        raise TypeError('%sA sparse matrix was passed, but dense '
                        'data is required. Use X.toarray() to '
                        'convert to a dense numpy array.' % context)
    if dtype is None:
        dtype = spmatrix.dtype

    CHANGE_FORMAT = False
    if (isinstance(accept_sparse, (list, tuple)) and spmatrix.format not in
            accept_sparse):
        CHANGE_FORMAT = True

    if CHANGE_FORMAT:
        msg = ("%sSparse format not one of recommended [format: %s]. "
               "Consider changing one of %r")
        warnings.warn(msg % (context, spmatrix.format, accept_sparse),
                      InputDataWarning)

    CHANGE_DTYPE = False
    if dtype != spmatrix.dtype:
        # convert dtype
        CHANGE_DTYPE = True

    if CHANGE_DTYPE:
        msg = ("%sDtype of sparse array not the expected type [dtype: %s]. "
               "Consider changing to %r")
        warnings.warn(msg % (context, spmatrix.dtype, dtype), InputDataWarning)

    ALL_FINITE = True
    if force_all_finite:
        if not hasattr(spmatrix, "data"):
            msg = "%sCan't check %s sparse matrix for nan or inf."
            warnings.warn(msg % (context, spmatrix.format))
        else:
            ALL_FINITE = check_all_finite(spmatrix.data)

    if not ALL_FINITE:
        msg = ("%sNot all elements in array are finite. This may cause "
               "estimation problems. Consider nan conversion and replacing "
               "infinite values.")
        warnings.warn(msg % context, InputDataWarning)

    return CHANGE_DTYPE or CHANGE_FORMAT or not ALL_FINITE


def soft_check_x_y(X, y, accept_sparse=True, dtype=None,
                   force_all_finite=True, ensure_2d=True, allow_nd=True,
                   multi_output=False, ensure_min_samples=1,
                   ensure_min_features=1, y_numeric=False, estimator=None):
    """Input validation before estimation.

    Checks X and y for consistent length, and X 2d and y 1d.
    Standard input checks are only applied to y, such as checking that y
    does not have np.nan or np.inf targets. For multi-label y, set
    multi_output=True to allow 2d and sparse y.  Raises warnings if the
    dtype is object.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    accept_sparse : string, list of string or None (default=None)
        String[s] representing allowed sparse matrix formats, such as 'csc',
        'csr', etc.  None means that sparse matrix input will raise an error.
        If the input is sparse but not in the allowed format, it will be
        converted to the first listed format.

    dtype : string, type, list of types or None (default="numeric")
        Data type of result. If None, the dtype of the input is preserved.
        If "numeric", dtype is preserved unless array.dtype is object.
        If dtype is a list of types, conversion on the first type is only
        performed if the dtype of the input is not in the list.

    force_all_finite : boolean (default=True)
        Whether to raise an error on np.inf and np.nan in X. This parameter
        does not influence whether y can have np.inf or np.nan values.

    ensure_2d : boolean (default=True)
        Whether to make X at least 2d.

    allow_nd : boolean (default=False)
        Whether to allow X.ndim > 2.

    multi_output : boolean (default=False)
        Whether to allow 2-d y (array or sparse matrix). If false, y will be
        validated as a vector. y cannot have np.nan or np.inf values if
        multi_output=True.

    ensure_min_samples : int (default=1)
        Make sure that X has a minimum number of samples in its first
        axis (rows for a 2D array).

    ensure_min_features : int (default=1)
        Make sure that the 2D array has some minimum number of features
        (columns). The default value of 1 rejects empty datasets.
        This check is only enforced when X has effectively 2 dimensions or
        is originally 1D and ``ensure_2d`` is True. Setting to 0 disables
        this check.

    y_numeric : boolean (default=False)
        Whether to ensure that y has a numeric type. If dtype of y is object,
        it is converted to float64. Should only be used for regression
        algorithms.

    estimator : str or estimator instance (default=None)
        If passed, include the name of the estimator in warning messages.

    Returns
    -------
    X_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    # ------ Check X ------
    CHANGE_X = soft_check_array(X, accept_sparse, dtype,
                                force_all_finite, ensure_2d, allow_nd,
                                ensure_min_samples, ensure_min_features,
                                estimator)

    # ------ Check y ------
    if multi_output:
        CHANGE_y = soft_check_array(y, accept_sparse=['csr'],
                                    force_all_finite=force_all_finite,
                                    ensure_2d=False, dtype=dtype,
                                    estimator=estimator)
    else:
        CHANGE_y = soft_check_1d(y, y_numeric, estimator)

    # Check consistent lengths. This raises an error if test fails.
    check_consistent_length(X, y)

    return CHANGE_X or CHANGE_y


def soft_check_1d(y, y_numeric, estimator):
    """Check if y is numeric, finite and one-dimensional."""
    context = _get_context(estimator)

    CHANGE_y = _check_column_or_1d(y)

    ALL_FINITE = _check_all_finite(y)
    if not ALL_FINITE:
        CHANGE_y = True
        msg = ("%sNot all elements in array are finite. This may "
               "cause estimation problems. Consider nan conversion "
               "and replacing infinite values.")
        warnings.warn(msg % context, InputDataWarning)

    if y_numeric and y.dtype.kind == 'O':
        CHANGE_y = True
        msg = ("%sDtype of y not the expected type [dtype: %s]. "
               "Consider changing to 'float' or 'int'.")
        warnings.warn(msg % (context, y.dtype.kind), InputDataWarning)

    if CHANGE_y:
        msg = ("%sy array failed initial test. Estimation may fail. "
               "Consider converting input data to a numpy array with "
               "finite elements and no missing values.")
        warnings.warn(msg % context, InputDataWarning)

    return CHANGE_y


def _check_column_or_1d(y, context=""):
    """Check if y can be raveled."""
    CHANGE = False
    try:
        s = tuple(np.shape(y))
    except Exception as e:
        raise ValueError("%sCould not get shape of y. "
                         "y should be an ndarray or scipy sparse csr "
                         "/csc matrix of shape (n_samples, ). Got %s."
                         "Details:\n%r" % (context, type(y), e))

    if len(s) == 0:
        raise ValueError("%sy is empty: y = %r." % (context, y))

    if len(s) == 2 and s[1] == 1:
        CHANGE = True
        warnings.warn("%sA column-vector y was passed when a 1d array was"
                      " expected. Change the shape of y to "
                      "(n_samples, ), for example using ravel()." % context,
                      InputDataWarning)

    if len(s) == 2 and s[1] > 1:
        CHANGE = True
        warnings.warn("%sA matrix y was passed for as for labels. "
                      "Most estimators expect a one dimensional label vector."
                      "Consider changing the shape of y to (n_samples, )." %
                      context, InputDataWarning)

    return CHANGE


def _check_x_y(X, y):
    """Wrapper for our default arguments - relax some Scikit-learn defaults."""
    return check_X_y(X, y,
                     accept_sparse=['csr', 'csc'],  # Accept sparse csr, csc
                     order=None,             # Make no C or Fortran imposition
                     copy=False,             # Do not trigger copying
                     force_all_finite=True,  # Raise error on np.inf or np.nan
                     ensure_2d=True,         # Force 'X' do be a matrix
                     allow_nd=True,          # Allow 'X.ndim' > 2
                     multi_output=True,      # Allow 'y.shape[1]' > 1
                     warn_on_dtype=False     # Mute as 'dtype' is 'None'
                     )


def _check_array(X):
    """Wrapper for our default arguments - relax some Scikit-learn defaults."""
    return check_array(X,
                       accept_sparse=['csr', 'csc'],  # Accept sparse csr, csc
                       order=None,  # Do not enforce C or Fortran
                       copy=False,  # Do not trigger copying
                       force_all_finite=True,  # Raise error on np.inf/np.nan
                       ensure_2d=True,  # Force 'X' do be a matrix
                       allow_nd=True,  # Allow 'X.ndim' > 2
                       warn_on_dtype=False  # Mute as 'dtype' is 'None'
                       )


def check_inputs(X, y=None, check_level=0):
    r"""Pre-checks on input arrays X and y.

    Checks input data according to ``check_level`` to ensure format is roughly
    in line with what a typical estimator expects.

    If ``check_level = 0`` this test is turned off.

    Parameters
    ----------
    X : nd-array, list or sparse matrix
        Input data.

    y : nd-array, list or sparse matrix
        Labels.

    check_level : int (default = 2)
        level of strictness in checking input arrays.

            - ``check_level = 0`` no checks, returns X, y

            - ``check_level`` = 1 will raises warnings if any non-critical
              test fails. Returns boolean FAIL flag.

            - ``check_level = 2`` will impose Scikit-learn array  check,
              which converts ``X`` and ``y`` to numpy arrays and raises error
              if conversion fails.

    Returns
    ---------
    FAIL : fail flag, optional
        boolean for whether any test failed. Returned if ``check_level = 1``

    X_converted : numpy array, optional
        The converted and validated X. Returned if ``check_level = 2``

    y_converted : numpy array, optional
        The converted and validated y. Returned if ``check_level = 2``.

    random_state : object, optional
        numpy RandomState object.
    """
    if check_level == 1:
        soft_check_x_y(X, y)

    if check_level == 2:

        if y is None:
            X = _check_array(X)
        else:
            X, y = _check_x_y(X, y)

    return X, y
