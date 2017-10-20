"""ML-ENSEMBLE

Currently placeholder for validation unit testing.
"""

import numpy as np
from scipy.sparse import csr
from mlens.utils.validation import _get_context, _check_all_finite, \
    check_all_finite, _check_sparse_format, check_inputs, \
    _check_column_or_1d, soft_check_1d, soft_check_x_y, soft_check_array

from mlens.utils.exceptions import InputDataWarning
from mlens.utils.dummy import OLS

import warnings

X = np.arange(12).reshape(6, 2)
y = np.arange(6)


def test__get_context():
    """[Utils] _get_context message formatting."""

    out = _get_context(None)

    assert out is ''

    out = _get_context(OLS())
    assert out == '[ols] '


def test_check_all_finite():
    """[Utils] _check_all_finite passes finite matrix."""
    assert _check_all_finite(X)


def test_check_all_finite_nans():
    """[Utils] _check_all_finite fails on NaNs."""
    Z = X.astype('float')
    Z[0, 0] = np.nan
    assert not _check_all_finite(Z)


def test_check_all_finite_ings():
    """[Utils] _check_all_finite fails on infs."""
    Z = X.astype('float')
    Z[0, 0] = np.inf
    assert not _check_all_finite(Z)


def test_check_all_finite_sparse():
    """[Utils] check_all_finite: passes X with sparse."""
    Z = X.astype('float')
    Z[0, 0] = np.inf

    assert check_all_finite(csr.csr_matrix(X))


def test_check_all_finite_sparse_nans():
    """[Utils] check_all_finite: fails sparse X with NaNs."""
    Z = X.astype('float')
    Z[0, 0] = np.nan

    assert not check_all_finite(csr.csr_matrix(Z))


def test_check_all_finite_sparse_infs():
    """[Utils] check_all_finite: fails sparse X with infs."""
    Z = X.astype('float')
    Z[0, 0] = np.inf

    assert not check_all_finite(csr.csr_matrix(Z))


def test_check_sparse_format():
    """[Utils] _check_sparse_format: passes sparse X."""
    assert not _check_sparse_format(csr.csr_matrix(X))


def test_check_sparse_format_dtype():
    """[Utils] _check_sparse_format: flags sparse X with wrong dtype."""
    flags = np.testing.assert_warns(InputDataWarning,
                                  _check_sparse_format,
                                  csr.csr_matrix(X),
                                  dtype='float')

    assert flags


def test_check_sparse_format_dtype():
    """[Utils] _check_sparse_format: flags sparse X with wrong sparse type."""
    flags = np.testing.assert_warns(InputDataWarning,
                                  _check_sparse_format,
                                  csr.csr_matrix(X),
                                  accept_sparse=['csc'])

    assert flags


def test_check_sparse_format_finite():
    """[Utils] _check_sparse_format: flags sparse X with inf or nan."""
    Z = X.astype('float')
    Z[0, 0] = np.inf

    flags = np.testing.assert_warns(InputDataWarning,
                                    _check_sparse_format,
                                    csr.csr_matrix(Z))

    assert flags


def test_check_inputs_0():
    """[Utils] check_inputs: no checks on level = 0."""

    Z = X.astype('float')
    Z[0, 0] = np.inf

    with warnings.catch_warnings(record=True) as w:
        H, _ = check_inputs(Z, y)

        assert len(w) == 0
        assert id(H) == id(Z)
        np.testing.assert_array_equal(Z, H)


def test_check_inputs_1():
    """[Utils] check_inputs: warnings level = 1."""

    Z = X.astype('float')
    Z[0, 0] = np.inf

    H, _ = np.testing.assert_warns(InputDataWarning, check_inputs, Z, y, 1)

    assert id(H) == id(Z)
    np.testing.assert_array_equal(Z, H)


def test_check_inputs_2():
    """[Utils] check_inputs: raises on level = 2."""

    Z = X.astype('float')
    Z[0, 0] = np.inf

    np.testing.assert_raises(ValueError, check_inputs, Z, y, 2)


def test_check_column_or_1d():
    """[Utils] _check_column_or_1d: wrong dim."""
    flag = np.testing.assert_warns(InputDataWarning,_check_column_or_1d,
                                   np.array(np.ones((6, 1))))
    assert flag


def test_check_column_or_1d_matrix():
    """[Utils] _check_column_or_1d: wrong dim."""
    flag = np.testing.assert_warns(InputDataWarning,_check_column_or_1d, X)
    assert flag


def test_check_column_or_1d_no_shape():
    """[Utils] _check_column_or_1d: raises on no shape."""
    np.testing.assert_raises(ValueError, _check_column_or_1d, {})


def test_soft_check_1d():
    """[Utils] soft_check_1d: passes correct vector."""
    assert not soft_check_1d(y, False, None)


def test_soft_check_1d_dim():
    """[Utils] soft_check_1d: flags 1-dim vector."""
    flags = np.testing.assert_warns(InputDataWarning, soft_check_1d,
                                    np.ones((6, 1)),
                                    False, None)
    assert flags


def test_soft_check_1d_dtype():
    """[Utils] soft_check_1d: flags string vector."""
    flags = np.testing.assert_warns(InputDataWarning, soft_check_1d,
                                    np.array(list('abcdef'), dtype="O"),
                                    True, None)
    assert flags


def test_soft_check_array():
    """[Utils] soft_check_array: passes correct X."""

    flag = soft_check_array(X)
    assert not flag


def test_soft_check_array_dtype():
    """[Utils] soft_check_array: flags on wrong dtype."""

    flag = np.testing.assert_warns(InputDataWarning, soft_check_array, X,
                                   dtype=['float'])
    assert flag


def test_soft_check_array_dim():
    """[Utils] soft_check_array: flags on wrong dim."""

    flag = np.testing.assert_warns(InputDataWarning, soft_check_array, y)
    assert flag


def test_soft_check_array_samples():
    """[Utils] soft_check_array: raises on too few samples to build matrix."""

    np.testing.assert_raises(ValueError, soft_check_array, y[:1],
                             ensure_min_samples=10)


def test_soft_check_array_samples_2():
    """[Utils] soft_check_array: warns on fewer samples than asked for."""

    flag = np.testing.assert_warns(InputDataWarning, soft_check_array, X,
                                   ensure_min_samples=7)
    assert flag


def test_soft_check_array_samples_2():
    """[Utils] soft_check_array: warns on fewer samples than asked for."""

    flag = np.testing.assert_warns(InputDataWarning, soft_check_array, X,
                                   ensure_min_features=10)
    assert flag


def test_soft_check_array_finite():
    """[Utils] soft_check_array: flags on inf or nan."""
    Z = X.astype('float')
    Z[0, 0] = np.inf
    flag = np.testing.assert_warns(InputDataWarning, soft_check_array, Z)
    assert flag

