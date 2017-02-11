#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 13:40:02 2017

@author: Sebastian
"""

from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlens.base import (_clone_base_estimators, _clone_preprocess_cases,
                        name_estimators, name_base, check_estimators,
                        _check_names, _split_base, _check_estimators,
                        name_columns)

# meta estimator
meta = SVR()

# Create ensemble with preprocessing pipelines
base_pipelines = {'sc':
                  ([StandardScaler()],
                   [('ls', Lasso()), ('kn', KNeighborsRegressor())]),
                  'mm':
                  ([MinMaxScaler()], [SVR()]),
                  'np':
                  ([], [('rf', RandomForestRegressor(random_state=100))])}


def test_naming():

    named_meta = name_estimators([meta], 'meta-')
    named_base = name_base(base_pipelines)

    assert isinstance(named_meta, dict)
    assert isinstance(named_meta['meta-svr'], SVR)
    assert isinstance(named_base, dict)
    assert len(named_base) == 6


def test_check_names():

    preprocess = [(case, _check_names(p[0])) for case, p in
                  base_pipelines.items()]

    base_estimators = [(case, _check_names(p[1])) for case, p in
                       base_pipelines.items()]

    assert isinstance(base_estimators, list)
    assert isinstance(preprocess, list)
    assert len(base_estimators) == 3
    assert len(preprocess) == 3
    assert isinstance(base_estimators[0], tuple)
    assert isinstance(preprocess[0], tuple)


def test_clone():

    preprocess = [(case, _check_names(p[0])) for case, p in
                  base_pipelines.items()]
    base_estimators = [(case, _check_names(p[1])) for case, p in
                       base_pipelines.items()]

    base_ = _clone_base_estimators(base_estimators)
    preprocess_ = _clone_preprocess_cases(preprocess)
    base_columns_ = name_columns(base_)

    assert isinstance(preprocess_, list)
    assert isinstance(preprocess_[0], tuple)
    assert isinstance(preprocess_[0][1], list)
    assert isinstance(base_, dict)
    assert isinstance(base_['mm'], list)
    assert isinstance(base_['mm'][0], tuple)
    assert isinstance(base_columns_, list)
    assert len(base_columns_) == 4
