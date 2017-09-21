"""ML-ENSEMBLE

:author: Sebastian Flennerhag
:copyright: 2017
:license: MIT

Scoring functions.
"""

from __future__ import division
import warnings

from ..utils.exceptions import MetricWarning

import numpy as np

try:
    from collections import OrderedDict as _dict
except ImportError:
    _dict = dict


def build_scores(score_list, partitions):
    """Build a score dictionary out of a list of scores"""
    scores = {'mean': _dict(),
              'std': _dict()
              }
    tmp = _dict()

    # Collect scores per preprocessing case and estimator(s)
    for name, score in score_list:
        splitted = name.split('__')

        if partitions == 1:
            key = tuple(splitted[:-2])
        else:
            key = tuple(splitted[:-1])

        try:
            tmp[key]
        except KeyError:
            tmp[key] = list()
            scores['mean'][key] = list()
            scores['std'][key] = list()

        tmp[key].append(score)

        # Aggregate to get cross-validated mean scores
        for k, v in tmp.items():
            if not v:
                continue
            try:
                scores['mean'][k] = np.mean(v)
                scores['std'][k] = np.std(v)
            except Exception as exc:
                warnings.warn(
                    "Aggregating scores for %s failed. Raw scores:\n%r\n"
                    "Details: %r" % (k, v, exc), MetricWarning)
    return scores


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
