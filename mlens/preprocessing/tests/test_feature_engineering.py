#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 15/01/2017
"""

from __future__ import division, print_function

from mlens.preprocessing.feature_engineering import PredictionFeature
from sklearn.linear_model import Lasso

pf = PredictionFeature([Lasso()])


def test_prediction_feature():
    assert pf is not None
