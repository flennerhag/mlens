#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sebastian Flennerhag
@date: 12/01/2017
"""

from __future__ import division, print_function

from mlens.preprocessing import Subset, StandardScalerDf
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

# training data
np.random.seed(100)
X = np.random.random((10, 5))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10

sc1 = StandardScalerDf()
sc2 = StandardScaler()

sub1 = Subset([1, 2])
sub2 = Subset()


def test_standard_scaler_df():
    Z = DataFrame(X)
    Zout = sc1.fit_transform(Z)
    Xout = sc2.fit_transform(X)
    Xout = DataFrame(Xout)
    assert ((Xout - Zout).abs() < 1e-10).any().any()


def test_Subset_1():
    assert sub1.fit_transform(X).shape[1] == 2


def test_Subset_2():
    assert (sub2.fit_transform(X) == X).all()
