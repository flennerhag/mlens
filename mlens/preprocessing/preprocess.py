#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML-ENSEMBLE
author: Sebastian Flennerhag
date: 10/01/2017
licence: MIT
Collection of preprocessing functions
"""

from __future__ import division, print_function

from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class StandardScalerDf(StandardScaler):
    """Wrapper around StandardScaler to preserve DataFrame representation"""
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def transform(self, X, y=None, copy=None):
        X.loc[:, :] = super().transform(X, y, copy)
        return X


class Subset(BaseEstimator, TransformerMixin):

    """
    Class for manually selecting a subset

    Parameters
    ----------
    subset : list
        list of columns indexes which can be either strings or integers
    """

    def __init__(self, subset=None):
        self.subset = subset

    def fit(self, X, y=None):

        self.is_df_ = isinstance(X, (DataFrame, Series))

        if self.subset is not None:
            self.use_loc_ = any([isinstance(x, str) for x in self.subset])

        return self

    def transform(self, X):

        if self.subset is None:
            return X
        else:
            Xt = X.copy()
            if self.is_df_ and self.use_loc_:
                Xt = Xt.loc[:, self.subset]
            elif self.is_df_:
                Xt = Xt.iloc[:, self.subset]
            else:
                Xt = Xt[:, self.subset]
            return Xt
