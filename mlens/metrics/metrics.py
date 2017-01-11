#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 23:01:02 2017

author: Sebastian Flennerhag
date: 11/01/2017
Scoring functions not in the sklearn library
"""

import numpy as np
from sklearn.metrics import make_scorer


# Root mean square error := sqrt(mse), mse := (1/n) * sum( (y-p)**2 )
def rmse_scoring(y, p):
    return np.mean((y-p)**2)**(1/2)

rmse = make_scorer(rmse_scoring, greater_is_better=False)
