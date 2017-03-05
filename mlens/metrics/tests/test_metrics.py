"""ML-ENSEMBLE

author: Sebastian Flennerhag
date: 03/03/2017
"""

from __future__ import division, print_function

import mlens.metrics as metrics
from sklearn.linear_model import Lasso

import numpy as np

SEED = 100

np.random.seed(SEED)

y = np.random.random(100)
p = np.random.random(100)
z = np.random.random(100)

X = np.vstack([p, z]).T

ls = Lasso(alpha=0.01, random_state=SEED)

def test_score_matrix():
    out = metrics.score_matrix(X, y, metrics.metrics.rmse_scoring, column_names=None, prefix=None)

    out = sorted(out)
    for i in range(len(out)):
        assert out[i] == 'preds_%i' % (i + 1)


def test_score_matrix_prefix():
    out = metrics.score_matrix(np.vstack([p, z]).T, y, metrics.metrics.rmse_scoring, prefix='test')

    out = sorted(out)
    for i in range(len(out)):
        assert out[i] == 'test-preds_%i' % (i + 1)


def test_scoring():
    ls.fit(X, y)

    scores = []
    for scorer in [metrics.rmse, metrics.wape, metrics.mape]:
        scores.append(scorer(ls, X, y))

    for score, test in zip(scores, ['-0.289228279782', '-0.534121890877', '-3.31044721206']):
        assert str(score) == test
