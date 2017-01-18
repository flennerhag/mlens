#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: Sebastian Flennerhag
date: 15/01/2017
licence: MIT
"""

from __future__ import division, print_function

from mlens.utils import utils
from time import time, sleep
import numpy as np
from pandas import DataFrame


# Some data
X = np.random.random((10, 5))

# An object to pickle
d = {'entry1': 'test', 'entry2': 'also_test'}


def test_slice():

    Z = utils._slice(X, None)
    assert (Z == X).all()

    D = DataFrame(X)

    Z = utils._slice(D, [1, 2, 3])

    assert D.iloc[[1, 2, 3], :].equals(Z)


def test_print_msg():

    # temp logging class to redirect print messages to a python object
    class logger():

        def __init__(self):
            self.log = []

        def write(self, msg):
            self.log.append(msg)

    l = logger()
    t0 = time()
    sleep(1)

    utils.print_time(t0, message='test', file=l)

    assert l.log[0] == 'test | 00:00:01\n'


def test_pickle():

    utils.pickle_save(d, 'd')
    test = utils.pickle_load('d')

    assert isinstance(d, dict)
    assert test['entry1'] == 'test'
    assert test['entry2'] == 'also_test'
