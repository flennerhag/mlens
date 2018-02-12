"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017-2018
:license: MIT

Scoring functions.
"""

from __future__ import division

import numpy as np


def rmse(y, p):
    r"""Root Mean Square Error.

    .. math::

        RMSE(\mathbf{y}, \mathbf{p}) = \sqrt{MSE(\mathbf{y}, \mathbf{p})},

    with

    .. math::

        MSE(\mathbf{y}, \mathbf{p}) = |S| \sum_{i \in S} (y_i - p_i)^2

    Parameters
    ----------
    y : array-like of shape [n_samples, ]
        ground truth.

    p : array-like of shape [n_samples, ]
        predicted labels.

    Returns
    -------
    z: float
        root mean squared error.
    """
    z = y - p
    return np.sqrt(np.mean(np.multiply(z, z)))


def mape(y, p):
    r"""Mean Average Percentage Error.

    .. math::

        MAPE(\mathbf{y}, \mathbf{p}) =
        |S| \sum_{i \in S} | \frac{y_i - p_i}{y_i} |

    Parameters
    ----------
    y : array-like of shape [n_samples, ]
        ground truth.

    p : array-like of shape [n_samples, ]
        predicted labels.

    Returns
    -------
    z: float
        mean average percentage error.
    """
    return np.mean(np.divide(np.abs((y - p)), np.abs(y)))


def wape(y, p):
    r"""Weighted Mean Average Percentage Error.

    .. math::

        WAPE(\mathbf{y}, \mathbf{p}) =
        \frac{\sum_{i \in S} | y_i - p_i|}{ \sum_{i \in S} |y_i|}

    Parameters
    ----------
    y : array-like of shape [n_samples, ]
        ground truth.

    p : array-like of shape [n_samples, ]
        predicted labels.

    Returns
    -------
    z: float
        weighted mean average percentage error.
    """
    return np.sum(np.abs(y - p)) / np.sum(np.abs(y))
