#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 20:41:40 2017

@author: Sebastian
"""

import numpy as np
from mlens.parallel._preprocess_functions import _preprocess_pipe
from mlens.parallel._preprocess_functions import _preprocess_fold
from mlens.parallel import preprocess_folds, preprocess_pipes
from sklearn.preprocessing import StandardScaler


# training data
np.random.seed(100)
X = np.random.random((1000, 10))

# noisy output, y = x0 * x1 + x2^2 + x3 - x4^(1/4) + e
y = X[:, 0] * X[:, 1] + X[:, 2] ** 2 + X[:, 3] - X[:, 4] ** (1 / 4)

# Change scales
X[:, 0] *= 10
X[:, 1] += 10
X[:, 2] *= 5
X[:, 3] *= 3
X[:, 4] /= 10


def test_preprocess_pipe_fun():

    out = _preprocess_pipe(X, y, X, [StandardScaler()], fit=True)
    assert out is not None
    assert len(out) == 2
    assert isinstance(out[0], np.ndarray)

    assert all([(_preprocess_pipe(X, y, X, [], fit=True)[i] == X).all() for
                i in range(2)])
    assert (_preprocess_pipe(X, y, X, [], fit=True)[1] == X).all()

    out = _preprocess_pipe(X, y, None, [StandardScaler()], fit=True,
                           p_name='test')
    assert out[-1] == 'test'


def test_preprocess_fold_fun():

    out = _preprocess_fold(X, y, (range(500), range(500, 1000)),
                           ('test', [StandardScaler()]), fit=True)
    assert out is not None
    assert len(out) == 6
    assert all([out[i].shape[1] == 10 for i in range(2)])
    assert all([out[i].shape[0] == 500 for i in range(2, 4)])
    assert out[5] == 'test'


def test_preprocess_fold():

    preprocess = [('test', [StandardScaler()])]

    data = preprocess_folds(preprocess,
                            X, y, folds=2, fit=True,
                            shuffle=False,
                            random_state=100,
                            n_jobs=-1, verbose=False)
    assert len(data) == 2
    assert len(data[0]) == 6
    assert all([isinstance(data[0][i], np.ndarray) for i in range(4)])
    assert all([data[0][i].shape == (500, 10) for i in range(0, 2)])
    assert all([data[0][i].shape == (500,) for i in range(2, 5)])
    assert isinstance(data[0][-1], str)

    data = preprocess_folds([],
                            X, y, folds=2, fit=True,
                            shuffle=False,
                            random_state=100,
                            n_jobs=-1, verbose=False)
    assert data is not None
    assert len(data) != 0
    assert (X[data[1][-1]] == data[0][0]).all()
    assert (X[data[0][-1]] == data[1][0]).all()


def test_preprocess_pipe():

    preprocess = [('test',
                   [StandardScaler(copy=True,
                                   with_mean=True, with_std=True)])]

    out = preprocess_pipes(preprocess, X, y, fit=True, dry_run=False,
                           return_estimators=True)

    pipes, Z, cases = zip(*out)
    preprocess_ = [(case, pipe) for case, pipe in
                   zip(cases, pipes)]

    assert isinstance(preprocess_, list)
    assert isinstance(preprocess_[0], tuple)
    assert preprocess_[0][0] == 'test'
    assert hasattr(preprocess_[0][1][0], 'mean_')
